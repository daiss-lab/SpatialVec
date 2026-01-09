import math
import numpy as np
from shapely.affinity import rotate, translate, scale as shp_scale
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.validation import make_valid


def fix_geometry(geom):
    """
    Try to repair invalid geometries.
    """
    try:
        g2 = make_valid(geom)
        if g2.is_empty:
            return None
        return g2
    except Exception:
        try:
            g2 = geom.buffer(0)
            if g2.is_empty:
                return None
            return g2
        except Exception:
            return None


def centroid_xy(g):
    c = g.centroid
    return float(c.x), float(c.y)


def boundary_coords(g, max_points=4096):
    """
    Extract representative boundary/coordinate samples for PCA-based orientation.
    For polygons: exterior ring coords
    For lines: vertex coords
    For points/others: centroid only
    """
    xs, ys = [], []

    if isinstance(g, Polygon):
        xs0, ys0 = g.exterior.coords.xy
        xs.extend(xs0); ys.extend(ys0)

    elif isinstance(g, MultiPolygon):
        for p in g.geoms:
            xs0, ys0 = p.exterior.coords.xy
            xs.extend(xs0); ys.extend(ys0)

    elif isinstance(g, LineString):
        xs0, ys0 = g.coords.xy
        xs.extend(xs0); ys.extend(ys0)

    elif isinstance(g, MultiLineString):
        for ln in g.geoms:
            xs0, ys0 = ln.coords.xy
            xs.extend(xs0); ys.extend(ys0)

    else:
        cx, cy = centroid_xy(g)
        xs, ys = [cx], [cy]

    arr = np.column_stack([np.asarray(xs), np.asarray(ys)])

    if arr.shape[0] > max_points:
        sel = np.linspace(0, arr.shape[0] - 1, max_points).astype(int)
        arr = arr[sel]

    return arr


def pca_theta_from_coords(arr: np.ndarray) -> float:
    """
    PCA direction of maximum variance for the coordinates -> angle in radians.
    """
    if arr.shape[0] < 2:
        return 0.0

    X = arr - arr.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v0 = Vt[0]
    return math.atan2(float(v0[1]), float(v0[0]))


def cnf_normalize(geom):
    """
    Canonical Normalized Frame (CNF):
      1) Translate to centroid at origin
      2) Rotate by PCA major axis so orientation is canonical
      3) Scale so max half-extent becomes 1 -> fits in [-1,1]^2
    """
    # 1) translate
    cx, cy = centroid_xy(geom)
    g1 = translate(geom, xoff=-cx, yoff=-cy)

    # 2) rotate
    arr = boundary_coords(g1)
    theta = pca_theta_from_coords(arr)  # radians
    g2 = rotate(g1, angle=-math.degrees(theta), origin=(0, 0), use_radians=False)

    # 3) scale
    minx, miny, maxx, maxy = g2.bounds
    half_extent = max(max(abs(minx), abs(maxx)), max(abs(miny), abs(maxy)))

    if half_extent > 0:
        g3 = shp_scale(g2, xfact=1 / half_extent, yfact=1 / half_extent, origin=(0, 0))
    else:
        g3 = g2

    return g3
