import numpy as np
from pathlib import Path
from shapely.geometry import Point


def make_base_grid(N=64):
    xs = np.linspace(-1.0, 1.0, N)
    ys = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts, xx, yy


def udf_point_to_geom(pt: Point, geom):
    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return pt.distance(geom.boundary)
    elif gt in ("LineString", "MultiLineString"):
        return pt.distance(geom)
    elif gt == "Point":
        return pt.distance(geom)
    else:
        return pt.distance(geom)


def occupancy_point(pt: Point, geom):
    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return 1 if geom.covers(pt) else 0
    return 0


def binary_search_boundary(p0, p1, geom, max_iter=15, tol=1e-4):
    occ0 = occupancy_point(Point(float(p0[0]), float(p0[1])), geom)
    occ1 = occupancy_point(Point(float(p1[0]), float(p1[1])), geom)
    if occ0 == occ1:
        return None

    a = p0.copy()
    b = p1.copy()
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        occ_mid = occupancy_point(Point(float(mid[0]), float(mid[1])), geom)

        if np.linalg.norm(b - a) < tol:
            return mid

        if occ_mid == occ0:
            a = mid
        else:
            b = mid

    return 0.5 * (a + b)


def refine_boundaries_from_grid(grid_pts, occ_mask, N, geom,
                                max_refine=2000,
                                max_iter=15,
                                tol=1e-4):
    refined = []

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            k = idx(i, j)

            if j < N - 1:
                k2 = idx(i, j + 1)
                if occ_mask[k] != occ_mask[k2]:
                    m = binary_search_boundary(grid_pts[k], grid_pts[k2], geom, max_iter=max_iter, tol=tol)
                    if m is not None:
                        refined.append(m)

            if i < N - 1:
                k3 = idx(i + 1, j)
                if occ_mask[k] != occ_mask[k3]:
                    m = binary_search_boundary(grid_pts[k], grid_pts[k3], geom, max_iter=max_iter, tol=tol)
                    if m is not None:
                        refined.append(m)

    if len(refined) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    refined = np.vstack(refined)

    if refined.shape[0] > max_refine:
        sel = np.linspace(0, refined.shape[0] - 1, max_refine).astype(int)
        refined = refined[sel]

    return refined.astype(np.float32)


def process_entity(geom, entity_id, base_grid, N, out_samples_dir,
                   fix_geometry_fn,
                   cnf_normalize_fn,
                   max_refine=2000,
                   max_iter=15,
                   tol=1e-4):
    g_fix = fix_geometry_fn(geom)
    if g_fix is None or g_fix.is_empty:
        return False

    g_cnf = cnf_normalize_fn(g_fix)
    if g_cnf is None or g_cnf.is_empty:
        return False

    grid_pts, _, _ = base_grid

    base_udf = np.empty((grid_pts.shape[0],), dtype=np.float32)
    base_occ = np.empty((grid_pts.shape[0],), dtype=np.int8)

    for idx, (x, y) in enumerate(grid_pts):
        p = Point(float(x), float(y))
        base_udf[idx] = udf_point_to_geom(p, g_cnf)
        base_occ[idx] = occupancy_point(p, g_cnf)

    refine_pts = np.zeros((0, 2), dtype=np.float32)
    if g_cnf.geom_type in ("Polygon", "MultiPolygon"):
        refine_pts = refine_boundaries_from_grid(
            grid_pts,
            base_occ,
            N,
            g_cnf,
            max_refine=max_refine,
            max_iter=max_iter,
            tol=tol
        )

    if refine_pts.shape[0] > 0:
        refine_udf = np.empty((refine_pts.shape[0],), dtype=np.float32)
        refine_occ = np.empty((refine_pts.shape[0],), dtype=np.int8)

        for i, (x, y) in enumerate(refine_pts):
            p = Point(float(x), float(y))
            refine_udf[i] = udf_point_to_geom(p, g_cnf)
            refine_occ[i] = occupancy_point(p, g_cnf)

        XY = np.vstack([grid_pts.astype(np.float32), refine_pts.astype(np.float32)])
        UDF = np.concatenate([base_udf, refine_udf])
        OCC = np.concatenate([base_occ, refine_occ])
    else:
        XY = grid_pts.astype(np.float32)
        UDF = base_udf
        OCC = base_occ

    out_samples_dir = Path(out_samples_dir)
    out_samples_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_samples_dir / f"XY_{entity_id:04d}.npy", XY)
    np.save(out_samples_dir / f"UDF_{entity_id:04d}.npy", UDF)
    np.save(out_samples_dir / f"OCC_{entity_id:04d}.npy", OCC)

    return True
