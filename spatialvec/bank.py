import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from .config import (
    MAX_ASPECT_RATIO,
    MAX_COMPACTNESS,
    MAX_FILL_RATIO,
    META_DIM,
    SamplingConfig,
)
from .geometry import canonicalize
from .sampling import boundary_aware_sample, sampling_plan

_SHARED = {}


def _sample_one(index):
    return index, boundary_aware_sample(
        _SHARED["canons"][index],
        _SHARED["gtypes"][index],
        _SHARED["cfg"],
        pad_to=_SHARED["pad_to"],
        rng=np.random.default_rng([_SHARED["seed"], index]),
    )


def canonicalize_registry(registry, cfg, progress=True):
    n = len(registry)
    canons = []
    metas = np.zeros((n, META_DIM), dtype=np.float32)
    budgets = np.zeros(n, dtype=np.int64)

    stream = range(n)
    if progress:
        stream = tqdm(stream, desc="canonicalise")

    for i in stream:
        canon, meta = canonicalize(
            registry.coords[i], registry.gtypes[i], cfg.point_canon_scale
        )
        canons.append(canon)
        metas[i] = meta
        budgets[i] = sampling_plan(canon, registry.gtypes[i], cfg)[0]

    return canons, metas, budgets


def build_bank(
    registry,
    cfg=None,
    seed=42,
    pad_to=None,
    workers=1,
    source=None,
    progress=True,
):
    cfg = SamplingConfig() if cfg is None else cfg
    canons, metas, budgets = canonicalize_registry(registry, cfg, progress=progress)

    n = len(registry)
    pad_to = int(budgets.max()) if pad_to is None else int(pad_to)
    if int(budgets.max()) > pad_to:
        raise ValueError(
            "pad_to %d is smaller than the largest budget %d"
            % (pad_to, int(budgets.max()))
        )

    xy = np.zeros((n, pad_to, 2), dtype=np.float32)
    sdf = np.zeros((n, pad_to), dtype=np.float32)
    occ = np.zeros((n, pad_to), dtype=np.float32)
    boundary = np.zeros((n, pad_to), dtype=np.float32)
    mask = np.zeros((n, pad_to), dtype=np.float32)

    _SHARED.clear()
    _SHARED.update(
        {
            "canons": canons,
            "gtypes": registry.gtypes,
            "cfg": cfg,
            "pad_to": pad_to,
            "seed": int(seed),
        }
    )

    if workers > 1:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            stream = pool.map(_sample_one, range(n), chunksize=64)
            if progress:
                stream = tqdm(stream, total=n, desc="sample")
            for index, sample in stream:
                xy[index] = sample["xy"]
                sdf[index] = sample["sdf"]
                occ[index] = sample["occ"]
                boundary[index] = sample["boundary"]
                mask[index] = sample["mask"]
    else:
        stream = range(n)
        if progress:
            stream = tqdm(stream, desc="sample")
        for index in stream:
            _, sample = _sample_one(index)
            xy[index] = sample["xy"]
            sdf[index] = sample["sdf"]
            occ[index] = sample["occ"]
            boundary[index] = sample["boundary"]
            mask[index] = sample["mask"]

    _SHARED.clear()

    return {
        "xy": torch.from_numpy(xy),
        "sdf": torch.from_numpy(sdf),
        "occ": torch.from_numpy(occ),
        "boundary": torch.from_numpy(boundary),
        "mask": torch.from_numpy(mask),
        "meta": torch.from_numpy(metas),
        "n_e": torch.from_numpy(budgets),
        "gtypes": list(registry.gtypes),
        "source_indices": torch.tensor(registry.source_indices, dtype=torch.long),
        "row_to_gid": dict(registry.row_to_gid),
        "config": dict(cfg.as_dict(), source=str(source), pad_to=pad_to, seed=int(seed)),
    }


def normalize_meta(bank):
    meta = bank["meta"].clone().float()

    meta[:, 14] = meta[:, 14].clamp(-MAX_ASPECT_RATIO, MAX_ASPECT_RATIO)
    meta[:, 15] = meta[:, 15].clamp(0.0, MAX_FILL_RATIO)
    meta[:, 16] = meta[:, 16].clamp(0.0, MAX_COMPACTNESS)

    continuous = meta[:, 3:]
    mean = continuous.mean(dim=0)
    std = continuous.std(dim=0).clamp(min=1e-6)
    meta[:, 3:] = (continuous - mean) / std

    full_mean = torch.zeros(META_DIM)
    full_std = torch.ones(META_DIM)
    full_mean[3:] = mean
    full_std[3:] = std

    bank["meta"] = meta
    bank["meta_mean"] = full_mean
    bank["meta_std"] = full_std
    return bank


def bank_summary(bank):
    budgets = bank["n_e"].numpy()
    gtypes = np.asarray(bank["gtypes"])
    mask = bank["mask"].numpy() > 0
    occ = bank["occ"].numpy()
    boundary = bank["boundary"].numpy()

    rows = []
    for gtype in ("polygon", "polyline", "point"):
        selected = gtypes == gtype
        if not selected.any():
            continue
        live = mask[selected]
        rows.append(
            {
                "type": gtype,
                "count": int(selected.sum()),
                "mean_budget": float(budgets[selected].mean()),
                "p1_budget": float(np.percentile(budgets[selected], 1)),
                "p50_budget": float(np.percentile(budgets[selected], 50)),
                "p99_budget": float(np.percentile(budgets[selected], 99)),
                "max_budget": int(budgets[selected].max()),
                "occupancy_rate": float(occ[selected][live].mean()),
                "boundary_rate": float(boundary[selected][live].mean()),
            }
        )

    return {
        "objects": int(len(budgets)),
        "pad_to": int(bank["xy"].shape[1]),
        "mean_budget": float(budgets.mean()),
        "total_queries": int(budgets.sum()),
        "live_fraction": float(mask.mean()),
        "per_type": rows,
    }
