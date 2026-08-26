import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatialvec.bank import normalize_meta
from spatialvec.config import ModelConfig, TrainConfig
from spatialvec.pipeline import export_embeddings, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup-epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig.hidden_dim)
    parser.add_argument("--n-heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--n-layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    args = parser.parse_args()

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
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    print("loaded bank: xy=%s" % (tuple(bank["xy"].shape),))
    bank = normalize_meta(bank)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
