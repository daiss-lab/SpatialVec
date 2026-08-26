import math

import numpy as np

from .config import SamplingConfig
from .geometry import canonical_descriptors, distance_to_polyline, point_in_polygon

BOUNDARY_SHELLS = np.array([0.0, 0.004, 0.012, 0.030, 0.075], dtype=np.float64)
POINT_SHELLS = np.array([0.004, 0.012, 0.030, 0.075, 0.180], dtype=np.float64)


def sample_budget(ell, area, delta, cfg=None):
    cfg = SamplingConfig() if cfg is None else cfg
    raw = cfg.n0 * (
        1.0
        + cfg.lambda_ell * float(ell)
        + cfg.lambda_area * float(delta) * math.sqrt(max(float(area), 0.0))
    )
    return int(min(cfg.n_max, max(cfg.n_min, math.ceil(raw))))


def region_weights(ell, area, delta, cfg=None):
    cfg = SamplingConfig() if cfg is None else cfg
    w_bnd = 1.0 + cfg.lambda_bnd * float(ell)
    w_int = float(delta) * (1.0 + cfg.lambda_int * math.sqrt(max(float(area), 0.0)))
    w_space = 1.0
    return w_bnd, w_int, w_space


def allocate(n_e, weights):
    w_bnd, w_int, w_space = weights
    w_total = w_bnd + w_int + w_space
    if w_total <= 0.0:
        raise ValueError("region weights must sum to a positive value")
    n_bnd = int(math.floor(n_e * w_bnd / w_total))
    n_int = int(math.floor(n_e * w_int / w_total))
    return n_bnd, n_int, n_e - n_bnd - n_int


def sampling_plan(canon, gtype, cfg=None):
    cfg = SamplingConfig() if cfg is None else cfg
    ell, area, delta = canonical_descriptors(canon, gtype)
    n_e = sample_budget(ell, area, delta, cfg)
    n_bnd, n_int, n_space = allocate(n_e, region_weights(ell, area, delta, cfg))
    return n_e, n_bnd, n_int, n_space


def sample_outer_space(n, cfg, rng):
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)
    return rng.uniform(-cfg.clip, cfg.clip, size=(n, 2)).astype(np.float32)


def sample_boundary(canon, gtype, n, cfg, rng):
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)

    if gtype == "point":
        radius = rng.choice(POINT_SHELLS, size=n)
        theta = rng.random(n) * 2.0 * math.pi
        offset = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        return offset.astype(np.float32)

    line = np.asarray(canon, dtype=np.float64)
    if line.ndim != 2:
        return sample_outer_space(n, cfg, rng)
    if gtype == "polygon":
        line = np.vstack([line, line[:1]])
    if len(line) < 2:
        return sample_outer_space(n, cfg, rng)

    lengths = np.linalg.norm(line[1:] - line[:-1], axis=1)
    total = float(lengths.sum())
    if total < 1e-9:
        return sample_outer_space(n, cfg, rng)

    segment = rng.choice(len(lengths), size=n, p=lengths / total)
    t = rng.random(n)[:, None]
    base = line[segment] * (1.0 - t) + line[segment + 1] * t

    radius = rng.choice(BOUNDARY_SHELLS, size=n)
    theta = rng.random(n) * 2.0 * math.pi
    offset = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return (base + offset).astype(np.float32)


def sample_interior(canon, n, cfg, rng):
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)

    ring = np.asarray(canon, dtype=np.float64)
    if ring.ndim != 2 or len(ring) < 3:
        return np.empty((0, 2), dtype=np.float32)

    lo = ring.min(axis=0)
    hi = ring.max(axis=0)
    if not np.all(hi > lo):
        return np.empty((0, 2), dtype=np.float32)

    kept = []
    found = 0
    drawn = 0
    limit = n * cfg.max_reject_factor

    while found < n and drawn < limit:
        size = int(min(max(n - found, 1) * 4, limit - drawn))
        candidates = rng.uniform(lo, hi, size=(size, 2))
        drawn += size
        inside = candidates[point_in_polygon(candidates, ring)]
        if len(inside):
            kept.append(inside)
            found += len(inside)

    if not kept:
        return np.empty((0, 2), dtype=np.float32)

    return np.concatenate(kept, axis=0)[:n].astype(np.float32)


def implicit_field(canon, gtype, xy, cfg):
    if len(xy) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    if gtype == "point":
        distance = np.linalg.norm(np.asarray(xy, dtype=np.float64), axis=1)
        near = (distance <= cfg.boundary_threshold).astype(np.float32)
        return distance.astype(np.float32), near, near.copy()

    if gtype == "polyline":
        distance = distance_to_polyline(xy, canon, closed=False)
        near = (distance <= cfg.boundary_threshold).astype(np.float32)
        return distance.astype(np.float32), near, near.copy()

    distance = distance_to_polyline(xy, canon, closed=True)
    inside = point_in_polygon(xy, canon)
    sdf = np.where(inside, -distance, distance).astype(np.float32)
    occupancy = inside.astype(np.float32)
    boundary = (np.abs(sdf) <= cfg.boundary_threshold).astype(np.float32)
    return sdf, occupancy, boundary


def boundary_aware_sample(canon, gtype, cfg=None, pad_to=None, rng=None):
    cfg = SamplingConfig() if cfg is None else cfg
    rng = np.random.default_rng() if rng is None else rng

    n_e, n_bnd, n_int, n_space = sampling_plan(canon, gtype, cfg)

    pad_to = n_e if pad_to is None else int(pad_to)
    if n_e > pad_to:
        raise ValueError("budget %d exceeds pad_to %d" % (n_e, pad_to))

    q_bnd = sample_boundary(canon, gtype, n_bnd, cfg, rng)

    if gtype == "polygon":
        q_int = sample_interior(canon, n_int, cfg, rng)
    else:
        q_int = np.empty((0, 2), dtype=np.float32)

    n_space = n_space + n_int - len(q_int)
    q_space = sample_outer_space(n_space, cfg, rng)

    xy = np.concatenate([q_bnd, q_int, q_space], axis=0)
    if len(xy) > n_e:
        xy = xy[:n_e]
    elif len(xy) < n_e:
        xy = np.concatenate([xy, sample_outer_space(n_e - len(xy), cfg, rng)], axis=0)

    rng.shuffle(xy, axis=0)
    sdf, occupancy, boundary = implicit_field(canon, gtype, xy, cfg)

    sample = {
        "xy": np.zeros((pad_to, 2), dtype=np.float32),
        "sdf": np.zeros(pad_to, dtype=np.float32),
        "occ": np.zeros(pad_to, dtype=np.float32),
        "boundary": np.zeros(pad_to, dtype=np.float32),
        "mask": np.zeros(pad_to, dtype=np.float32),
    }
    sample["xy"][:n_e] = xy
    sample["sdf"][:n_e] = sdf
    sample["occ"][:n_e] = occupancy
    sample["boundary"][:n_e] = boundary
    sample["mask"][:n_e] = 1.0
    sample["n_e"] = n_e
    sample["allocation"] = (n_bnd, int(len(q_int)), n_space)
    return sample
