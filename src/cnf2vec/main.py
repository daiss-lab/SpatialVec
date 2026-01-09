import argparse
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import torch
from tqdm.auto import tqdm

from .geometry import fix_geometry, cnf_normalize
from .sampling import make_base_grid, process_entity
from .train import train_cnf2vec


def balanced_subset(gdf, n_per_type=2000, seed=42):
    gdf = gdf[["geometry"]].copy()
    gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf.reset_index(drop=True)

    geom_types = gdf.geometry.geom_type
    idx_poly  = geom_types[geom_types.str.contains("Polygon")].index
    idx_line  = geom_types[geom_types.str.contains("Line")].index
    idx_point = geom_types[geom_types.str.contains("Point")].index

    np.random.seed(seed)
    n_poly  = min(n_per_type, len(idx_poly))
    n_line  = min(n_per_type, len(idx_line))
    n_point = min(n_per_type, len(idx_point))

    subset_ids = (
        np.random.choice(idx_poly,  n_poly,  replace=False).tolist() +
        np.random.choice(idx_line,  n_line,  replace=False).tolist() +
        np.random.choice(idx_point, n_point, replace=False).tolist()
    )
    subset_ids = np.array(sorted(subset_ids))
    gdf = gdf.iloc[subset_ids].reset_index(drop=True)
    return gdf


def generate_samples(gdf, out_samples_dir, grid_n=64, max_refine=2000, max_iter=15, tol=1e-4):
    out_samples_dir = Path(out_samples_dir)
    out_samples_dir.mkdir(parents=True, exist_ok=True)

    base_grid = make_base_grid(grid_n)

    n_ok = 0
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        ok = process_entity(
            geom=row.geometry,
            entity_id=int(idx),
            base_grid=base_grid,
            N=grid_n,
            out_samples_dir=out_samples_dir,
            fix_geometry_fn=fix_geometry,
            cnf_normalize_fn=cnf_normalize,
            max_refine=max_refine,
            max_iter=max_iter,
            tol=tol
        )
        if ok:
            n_ok += 1

    print(f"Finished sampling. Saved {n_ok}/{len(gdf)} entities into: {out_samples_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg_path", type=str, required=True)
    ap.add_argument("--layer", type=str, required=True)
    ap.add_argument("--out_root", type=str, default="outputs/nyc")

    ap.add_argument("--n_per_type", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--grid_n", type=int, default=64)
    ap.add_argument("--max_refine", type=int, default=2000)
    ap.add_argument("--max_iter", type=int, default=15)
    ap.add_argument("--tol", type=float, default=1e-4)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample_ratio", type=float, default=0.5)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPKG...")
    gdf = gpd.read_file(args.gpkg_path, layer=args.layer)
    gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf.reset_index(drop=True)

    print("Total rows after filtering:", len(gdf))
    print("Type counts:\n", gdf.geometry.geom_type.value_counts())

    print(f"\nCreating balanced subset: n_per_type={args.n_per_type}")
    gdf_sub = balanced_subset(gdf, n_per_type=args.n_per_type, seed=args.seed)
    print("Subset size:", len(gdf_sub))
    print("Subset type counts:\n", gdf_sub.geometry.geom_type.value_counts())

    print("\nSampling UDF/OCC + refinement...")
    generate_samples(
        gdf_sub,
        out_samples_dir=samples_dir,
        grid_n=args.grid_n,
        max_refine=args.max_refine,
        max_iter=args.max_iter,
        tol=args.tol
    )

    print("\nTraining model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = train_cnf2vec(
        samples_dir=str(samples_dir),
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        device=device
    )

    ckpt_path = out_root / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"\nSaved model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
