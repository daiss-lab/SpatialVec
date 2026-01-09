import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import CNF2VecDataset, variable_length_collate
from .model import CNF2VecModel


def train_cnf2vec(samples_dir, batch_size=4, lr=1e-3, epochs=2, sample_ratio=0.5, seed=42, device=None):
    dataset = CNF2VecDataset(samples_dir, sample_ratio=sample_ratio, seed=seed)
    n_total = len(dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=variable_length_collate)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNF2VecModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=True)
        total_loss = 0.0

        for batch in pbar:
            xy      = batch["xy"].to(device)
            udf_gt  = batch["udf"].to(device)
            occ_gt  = batch["occ"].to(device)
            mask    = batch["mask"].to(device)

            optimizer.zero_grad()
            udf_pred, occ_logits, _ = model(xy)

            l_udf = (l1(udf_pred, udf_gt) * mask).sum() / mask.sum().clamp_min(1.0)
            l_occ = (bce(occ_logits, occ_gt) * mask).sum() / mask.sum().clamp_min(1.0)

            loss = l_udf + 0.5 * l_occ
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / max(1, len(train_loader))
        print(f"[{ep:03d}] Train Loss = {avg:.4f}")

    return model
