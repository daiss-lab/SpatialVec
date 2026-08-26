import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatialvec.bank import bank_summary, build_bank
from spatialvec.config import SamplingConfig
from spatialvec.geometry import load_geometries


def add_sampling_args(parser):
    defaults = SamplingConfig()
    parser.add_argument("--n0", type=int, default=defaults.n0)
    parser.add_argument("--n-min", type=int, default=defaults.n_min)
    parser.add_argument("--n-max", type=int, default=defaults.n_max)
    parser.add_argument("--lambda-ell", type=float, default=defaults.lambda_ell)
    parser.add_argument("--lambda-area", type=float, default=defaults.lambda_area)
    parser.add_argument("--lambda-bnd", type=float, default=defaults.lambda_bnd)
    parser.add_argument("--lambda-int", type=float, default=defaults.lambda_int)
    parser.add_argument("--clip", type=float, default=defaults.clip)
    parser.add_argument(
        "--boundary-threshold", type=float, default=defaults.boundary_threshold
    )
    return parser


def sampling_config_from_args(args):
    return SamplingConfig(
        n0=args.n0,
        n_min=args.n_min,
        n_max=args.n_max,
        lambda_ell=args.lambda_ell,
        lambda_area=args.lambda_area,
        lambda_bnd=args.lambda_bnd,
        lambda_int=args.lambda_int,
        clip=args.clip,
        boundary_threshold=args.boundary_threshold,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pad-to", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", default=None)
    add_sampling_args(parser)
    args = parser.parse_args()

    cfg = sampling_config_from_args(args)

    registry, skipped = load_geometries(args.input, limit=args.limit)
    print("unique geometries: %d  skipped: %d" % (len(registry), skipped))
    print("type counts: %s" % registry.type_counts())

    bank = build_bank(
        registry,
        cfg=cfg,
        seed=args.seed,
        pad_to=args.pad_to,
        workers=args.workers,
        source=args.input,
    )

    summary = bank_summary(bank)
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out)
    print("saved bank: %s  xy=%s" % (out, tuple(bank["xy"].shape)))

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({"config": cfg.as_dict(), "summary": summary}, handle, indent=2)
        print("saved summary: %s" % summary_path)


if __name__ == "__main__":
    main()
