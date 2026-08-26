import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatialvec.config import SamplingConfig
from spatialvec.geometry import canonical_descriptors, canonicalize, load_geometries
from spatialvec.sampling import allocate, region_weights, sample_budget

TYPES = ("polygon", "polyline", "point")


def descriptors(registry, cfg):
    n = len(registry)
    ell = np.zeros(n)
    area = np.zeros(n)
    delta = np.zeros(n)
    for i in range(n):
        canon, _ = canonicalize(
            registry.coords[i], registry.gtypes[i], cfg.point_canon_scale
        )
        ell[i], area[i], delta[i] = canonical_descriptors(canon, registry.gtypes[i])
    return ell, area, delta


def budgets_for(ell, area, delta, cfg):
    return np.array(
        [sample_budget(l, a, d, cfg) for l, a, d in zip(ell, area, delta)],
        dtype=np.int64,
    )


def solve_n0(ell, area, delta, cfg, target):
    low, high = 1, 1 << 16
    while low < high:
        mid = (low + high) // 2
        probe = SamplingConfig(**dict(cfg.as_dict(), n0=mid))
        if budgets_for(ell, area, delta, probe).mean() < target:
            low = mid + 1
        else:
            high = mid
    return low


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-mean", type=float, default=1024.0)
    parser.add_argument("--n-min", type=int, default=SamplingConfig.n_min)
    parser.add_argument("--n-max", type=int, default=SamplingConfig.n_max)
    parser.add_argument("--lambda-ell", type=float, default=SamplingConfig.lambda_ell)
    parser.add_argument("--lambda-area", type=float, default=SamplingConfig.lambda_area)
    parser.add_argument("--lambda-bnd", type=float, default=SamplingConfig.lambda_bnd)
    parser.add_argument("--lambda-int", type=float, default=SamplingConfig.lambda_int)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = SamplingConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        lambda_ell=args.lambda_ell,
        lambda_area=args.lambda_area,
        lambda_bnd=args.lambda_bnd,
        lambda_int=args.lambda_int,
    )

    registry, skipped = load_geometries(args.input, limit=args.limit)
    print("unique geometries: %d  skipped: %d" % (len(registry), skipped))

    ell, area, delta = descriptors(registry, base)
    n0 = solve_n0(ell, area, delta, base, args.target_mean)
    cfg = SamplingConfig(**dict(base.as_dict(), n0=n0))

    budgets = budgets_for(ell, area, delta, cfg)
    gtypes = np.asarray(registry.gtypes)

    print("calibrated n0 = %d  mean budget = %.2f (target %.1f)" % (
        n0, budgets.mean(), args.target_mean
    ))
    print("total queries = %d  vs fixed %d = %.4f" % (
        budgets.sum(),
        int(args.target_mean) * len(budgets),
        budgets.sum() / (args.target_mean * len(budgets)),
    ))

    per_type = []
    for gtype in TYPES:
        selected = gtypes == gtype
        if not selected.any():
            continue
        share = np.zeros(3)
        for i in np.nonzero(selected)[0]:
            parts = allocate(
                int(budgets[i]),
                region_weights(ell[i], area[i], delta[i], cfg),
            )
            share += parts
        total = share.sum()
        row = {
            "type": gtype,
            "count": int(selected.sum()),
            "mean_budget": float(budgets[selected].mean()),
            "p1": float(np.percentile(budgets[selected], 1)),
            "p50": float(np.percentile(budgets[selected], 50)),
            "p99": float(np.percentile(budgets[selected], 99)),
            "max": int(budgets[selected].max()),
            "frac_boundary": float(share[0] / total),
            "frac_interior": float(share[1] / total),
            "frac_outer": float(share[2] / total),
        }
        per_type.append(row)
        print(
            "  %-9s n=%6d  N_E mean %7.1f p1 %5.0f p50 %5.0f p99 %5.0f max %5d  "
            "bnd %.3f int %.3f spc %.3f"
            % (
                row["type"], row["count"], row["mean_budget"], row["p1"],
                row["p50"], row["p99"], row["max"],
                row["frac_boundary"], row["frac_interior"], row["frac_outer"],
            )
        )

    print("clamped at n_min: %d   at n_max: %d" % (
        int((budgets == cfg.n_min).sum()), int((budgets == cfg.n_max).sum())
    ))

    result = {
        "config": cfg.as_dict(),
        "objects": int(len(budgets)),
        "mean_budget": float(budgets.mean()),
        "total_queries": int(budgets.sum()),
        "target_mean": args.target_mean,
        "per_type": per_type,
        "clamped_low": int((budgets == cfg.n_min).sum()),
        "clamped_high": int((budgets == cfg.n_max).sum()),
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("saved: %s" % out)


if __name__ == "__main__":
    main()
