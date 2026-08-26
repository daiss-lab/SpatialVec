import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .model import SpatialVec
from .objective import reconstruction_loss

FIELD_KEYS = ("xy", "sdf", "occ", "boundary", "mask")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return False, torch.float32


class GeometryBankDataset(Dataset):
    def __init__(self, bank):
        self.bank = bank
        self.n_e = bank["n_e"].tolist()

    def __len__(self):
        return self.bank["xy"].shape[0]

    def __getitem__(self, index):
        item = {key: self.bank[key][index] for key in FIELD_KEYS}
        item["meta"] = self.bank["meta"][index]
        item["n_e"] = self.n_e[index]
        return item


def trim_collate(items):
    length = max(int(item["n_e"]) for item in items)
    batch = {
        key: torch.stack([item[key][:length] for item in items]) for key in FIELD_KEYS
    }
    batch["meta"] = torch.stack([item["meta"] for item in items])
    return batch


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }


def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)
    total_steps = max(total_epochs * steps_per_epoch, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / max(float(total_steps - warmup_steps), 1.0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(bank, model_cfg=None, train_cfg=None, device=None, checkpoint_path=None):
    model_cfg = ModelConfig() if model_cfg is None else model_cfg
    train_cfg = TrainConfig() if train_cfg is None else train_cfg
    device = resolve_device(device)
    use_amp, amp_dtype = resolve_amp(device)

    set_seed(train_cfg.seed)

    loader = DataLoader(
        GeometryBankDataset(bank),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=trim_collate,
        drop_last=False,
    )
    steps_per_epoch = len(loader)

    model = SpatialVec(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, train_cfg.warmup_epochs, train_cfg.epochs, steps_per_epoch
    )

    print("device=%s amp=%s dtype=%s" % (device, use_amp, amp_dtype))
    print("parameters=%d export_dim=%d" % (
        sum(p.numel() for p in model.parameters()), model.export_dim
    ))
    print("objects=%d steps_per_epoch=%d" % (len(loader.dataset), steps_per_epoch))

    best_loss = float("inf")
    best_state = None
    consecutive_nan_epochs = 0
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        started = time.time()
        collected = []
        nan_batches = 0

        for batch in tqdm(loader, desc="epoch %d/%d" % (epoch, train_cfg.epochs)):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                loss, metrics = reconstruction_loss(model, batch, train_cfg)

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()

            metrics["grad_norm"] = float(grad_norm)
            collected.append(metrics)

        if not collected:
            consecutive_nan_epochs += 1
            print("epoch %d: every batch was non-finite" % epoch)
            if consecutive_nan_epochs >= 3:
                print("stopping after 3 consecutive non-finite epochs")
                break
            continue

        consecutive_nan_epochs = 0
        summary = {
            key: float(np.mean([m[key] for m in collected])) for key in collected[0]
        }
        summary["epoch"] = epoch
        summary["nan_batches"] = nan_batches
        summary["lr"] = optimizer.param_groups[0]["lr"]
        summary["seconds"] = time.time() - started
        history.append(summary)

        print(
            "epoch %d loss=%.5f sdf=%.5f occ=%.5f boundary=%.5f "
            "occ_acc=%.4f boundary_acc=%.4f grad_norm=%.4f lr=%.6f "
            "nan_batches=%d time=%.1fs"
            % (
                epoch,
                summary["loss"],
                summary["sdf"],
                summary["occ"],
                summary["boundary"],
                summary["occ_acc"],
                summary["boundary_acc"],
                summary["grad_norm"],
                summary["lr"],
                nan_batches,
                summary["seconds"],
            )
        )

        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": best_state,
                        "model_config": model_cfg.as_dict(),
                        "train_config": train_cfg.as_dict(),
                        "epoch": epoch,
                        "loss": best_loss,
                    },
                    path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("warning: no finite epoch was recorded, exporting the untrained model")

    return model, history


@torch.no_grad()
def export_embeddings(model, bank, out_dir, device=None, batch_size=32):
    device = resolve_device(device)
    model = model.to(device).eval()

    loader = DataLoader(
        GeometryBankDataset(bank),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=trim_collate,
    )

    features = []
    for batch in tqdm(loader, desc="export"):
        batch = to_device(batch, device)
        z, _ = model.encode(
            batch["xy"],
            batch["sdf"],
            batch["occ"],
            batch["boundary"],
            batch["mask"],
            batch["meta"],
        )
        features.append(
            model.export_features(z.float(), batch["meta"].float()).cpu().numpy()
        )

    embeddings = np.concatenate(features, axis=0).astype(np.float32)

    row_to_gid = bank["row_to_gid"]
    entity_ids = np.array(sorted(row_to_gid.keys()), dtype=np.int64)
    gid_for_entity = np.array([row_to_gid[i] for i in entity_ids], dtype=np.int64)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Z_unique_geometries.npy", embeddings)
    np.savez(
        out_dir / "entity_embeddings.npz",
        entity_ids=entity_ids,
        embeddings=embeddings[gid_for_entity],
    )

    layout = {
        "z_dim": model.cfg.hidden_dim,
        "meta_dim": 17,
        "spatial_pe_dim": model.cfg.export_pe_dim,
        "final_dim": int(embeddings.shape[1]),
        "unique_geometries": int(embeddings.shape[0]),
        "total_entities": int(len(entity_ids)),
        "nan_values": int(np.isnan(embeddings).sum()),
    }
    with open(out_dir / "embedding_layout.json", "w", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2)

    print("embeddings %s -> %s" % (tuple(embeddings.shape), out_dir))
    if layout["nan_values"]:
        print("warning: %d non-finite values in the exported matrix" % layout["nan_values"])

    return embeddings, entity_ids
