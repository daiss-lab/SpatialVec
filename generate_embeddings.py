import argparse
import json
from pathlib import Path

import torch

from spatialvec.bank import bank_summary, build_bank, normalize_meta
from spatialvec.config import ModelConfig, SamplingConfig, TrainConfig
from spatialvec.geometry import load_geometries
from spatialvec.pipeline import export_embeddings, train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-bank", action="store_true")

    parser.add_argument("--n0", type=int, default=SamplingConfig.n0)
    parser.add_argument("--n-min", type=int, default=SamplingConfig.n_min)
    parser.add_argument("--n-max", type=int, default=SamplingConfig.n_max)
    parser.add_argument("--lambda-ell", type=float, default=SamplingConfig.lambda_ell)
    parser.add_argument("--lambda-area", type=float, default=SamplingConfig.lambda_area)
    parser.add_argument("--lambda-bnd", type=float, default=SamplingConfig.lambda_bnd)
    parser.add_argument("--lambda-int", type=float, default=SamplingConfig.lambda_int)
    parser.add_argument("--pad-to", type=int, default=None)
    parser.add_argument("--sample-workers", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup-epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--hidden-dim", type=int, default=ModelConfig.hidden_dim)
    parser.add_argument("--n-heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--n-layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    return parser.parse_args()


def main():
    args = parse_args()

    sampling_cfg = SamplingConfig(
        n0=args.n0,
        n_min=args.n_min,
        n_max=args.n_max,
        lambda_ell=args.lambda_ell,
        lambda_area=args.lambda_area,
        lambda_bnd=args.lambda_bnd,
        lambda_int=args.lambda_int,
    )
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "geometry_bank.pt"

    if args.reuse_bank and bank_path.exists():
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        print("reused bank: %s  xy=%s" % (bank_path, tuple(bank["xy"].shape)))
    else:
        registry, skipped = load_geometries(args.input, limit=args.limit)
        print("unique geometries: %d  skipped: %d" % (len(registry), skipped))
        print("type counts: %s" % registry.type_counts())

        bank = build_bank(
            registry,
            cfg=sampling_cfg,
            seed=args.seed,
            pad_to=args.pad_to,
            workers=args.sample_workers,
            source=args.input,
        )
        torch.save(bank, bank_path)
        print("saved bank: %s  xy=%s" % (bank_path, tuple(bank["xy"].shape)))

    summary = bank_summary(bank)
    with open(out_dir / "sampling_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"config": sampling_cfg.as_dict(), "summary": summary}, handle, indent=2)
    print(json.dumps(summary, indent=2))

    bank = normalize_meta(bank)

    model, history = train(
        bank,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=args.device,
        checkpoint_path=out_dir / "best_model.pt",
    )

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    export_embeddings(
        model, bank, out_dir, device=args.device, batch_size=train_cfg.batch_size
    )


if __name__ == "__main__":
    main()
