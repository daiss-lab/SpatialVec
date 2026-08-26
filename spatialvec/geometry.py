import hashlib
import math

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon

from .config import MAX_ASPECT_RATIO, MAX_COMPACTNESS, MAX_FILL_RATIO


def geom_type_name(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return "polygon"
    if geom.geom_type in ("LineString", "MultiLineString"):
        return "polyline"
    if geom.geom_type == "Point":
        return "point"
    return "unknown"


def flatten_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
        return max(parts, key=lambda g: g.area) if parts else None
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        return max(parts, key=lambda g: g.length) if parts else None
    return geom


def geometry_to_coords(geom, gtype):
    if gtype == "point":
        return np.array([geom.x, geom.y], dtype=np.float64)
    if gtype == "polyline":
        return np.asarray(geom.coords, dtype=np.float64)
    if gtype == "polygon":
        return np.asarray(geom.exterior.coords, dtype=np.float64)
    raise ValueError("unsupported geometry type: " + str(gtype))


def polyline_length(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(coords[1:] - coords[:-1], axis=1).sum())


def polygon_area(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or len(coords) < 3:
        return 0.0
    centred = coords - coords.mean(axis=0)
    x = centred[:, 0]
    y = centred[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(query, ring):
    ring = np.asarray(ring, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if ring.ndim != 2 or len(ring) < 3 or len(query) == 0:
        return np.zeros(len(query), dtype=bool)

    xi = ring[:, 0]
    yi = ring[:, 1]
    xj = np.roll(xi, 1)
    yj = np.roll(yi, 1)
    qx = query[:, 0][:, None]
    qy = query[:, 1][:, None]

    dy = yj - yi
    dy = np.where(dy == 0.0, 1e-12, dy)
    crosses = ((yi > qy) != (yj > qy)) & (qx < (xj - xi) * (qy - yi) / dy + xi)
    return (crosses.sum(axis=1) % 2) == 1


def distance_to_polyline(query, coords, closed):
    coords = np.asarray(coords, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if len(query) == 0:
        return np.zeros(0, dtype=np.float64)
    if coords.ndim == 1:
        return np.linalg.norm(query - coords[None, :], axis=1)
    if len(coords) == 1:
        return np.linalg.norm(query - coords[0], axis=1)

    if closed:
        a = coords
        b = np.roll(coords, -1, axis=0)
    else:
        a = coords[:-1]
        b = coords[1:]

    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = query[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / denom[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(query[:, None, :] - proj, axis=2).min(axis=1)


def canonicalize(coords, gtype, point_canon_scale=2.0e-4):
    coords = np.asarray(coords, dtype=np.float64)

    if gtype == "point":
        cx = float(coords[0])
        cy = float(coords[1])
        canon = np.zeros((1, 2), dtype=np.float32)
        minx = maxx = cx
        miny = maxy = cy
        width = height = diag = float(point_canon_scale)
        area = 0.0
        length = 0.0
    else:
        centre = coords.mean(axis=0)
        cx = float(centre[0])
        cy = float(centre[1])
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        minx = float(lo[0])
        miny = float(lo[1])
        maxx = float(hi[0])
        maxy = float(hi[1])
        width = maxx - minx
        height = maxy - miny
        diag = math.sqrt(width * width + height * height)
        canon = ((coords - centre) / max(diag, 1e-9)).astype(np.float32)
        if gtype == "polygon":
            area = polygon_area(coords)
            length = polyline_length(np.vstack([coords, coords[:1]]))
        else:
            area = 0.0
            length = polyline_length(coords)

    aspect = float(
        np.clip(width / max(height, 1e-9), -MAX_ASPECT_RATIO, MAX_ASPECT_RATIO)
    )
    fill = float(min(area / max(width * height, 1e-9), MAX_FILL_RATIO))
    if gtype == "polygon":
        compactness = float(
            min(4.0 * math.pi * area / max(length * length, 1e-9), MAX_COMPACTNESS)
        )
    else:
        compactness = 0.0

    meta = np.array(
        [
            1.0 if gtype == "polygon" else 0.0,
            1.0 if gtype == "polyline" else 0.0,
            1.0 if gtype == "point" else 0.0,
            cx,
            cy,
            minx,
            miny,
            maxx,
            maxy,
            width,
            height,
            diag,
            math.log1p(max(area, 0.0)),
            math.log1p(max(length, 0.0)),
            aspect,
            fill,
            compactness,
        ],
        dtype=np.float32,
    )

    return canon, meta


def canonical_descriptors(canon, gtype):
    if gtype == "point":
        return 0.0, 0.0, 0.0

    canon = np.asarray(canon, dtype=np.float64)
    if canon.ndim != 2 or len(canon) < 2:
        return 0.0, 0.0, 0.0

    if gtype == "polygon":
        ring = np.vstack([canon, canon[:1]])
        return polyline_length(ring), polygon_area(canon), 1.0

    return polyline_length(canon), 0.0, 0.0


class GeometryRegistry:
    def __init__(self):
        self.key_to_gid = {}
        self.coords = []
        self.gtypes = []
        self.source_indices = []
        self.row_to_gid = {}

    def add(self, coords, gtype, source_index):
        key = hashlib.blake2b(
            gtype.encode() + np.ascontiguousarray(coords).tobytes(),
            digest_size=16,
        ).digest()

        gid = self.key_to_gid.get(key)
        if gid is None:
            gid = len(self.coords)
            self.key_to_gid[key] = gid
            self.coords.append(coords)
            self.gtypes.append(gtype)
            self.source_indices.append(int(source_index))

        self.row_to_gid[int(source_index)] = gid
        return gid

    def type_counts(self):
        counts = {}
        for gtype in self.gtypes:
            counts[gtype] = counts.get(gtype, 0) + 1
        return counts

    def __len__(self):
        return len(self.coords)


def load_geometries(path, limit=None, progress=True):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if limit is not None:
        gdf = gdf.iloc[:limit]

    registry = GeometryRegistry()
    skipped = 0

    stream = enumerate(gdf.geometry)
    if progress:
        from tqdm import tqdm

        stream = tqdm(stream, total=len(gdf), desc="extract")

    for index, raw in stream:
        geom = flatten_geometry(raw)
        if geom is not None and not geom.is_valid:
            geom = flatten_geometry(geom.buffer(0))
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        gtype = geom_type_name(geom)
        if gtype == "unknown":
            skipped += 1
            continue

        coords = geometry_to_coords(geom, gtype)
        if coords.size == 0:
            skipped += 1
            continue

        registry.add(coords, gtype, index)

    return registry, skipped
