# SpatialVec (CNF2Vec pipeline)

This repository contains a runnable CNF-normalization + UDF/OCC sampling + MLP training pipeline for polymorphic geospatial objects (Point/LineString/Polygon).

## Setup

### Option A: Conda (recommended for GeoPandas on Windows)
```bash
conda create -n spatialvec python=3.10 -y
conda activate spatialvec
conda install -c conda-forge geopandas shapely -y
pip install -r requirements.txt

PYTHONPATH=src python -m cnf2vec.main \
  --gpkg_path "data/NYC_total_data.gpkg" \
  --layer "NYC_total_data" \
  --out_root "outputs/nyc" \
  --n_per_type 2000 \
  --grid_n 64 \
  --max_refine 2000 \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-3 \
  --sample_ratio 0.5


---

# 5) Now add the code files (copy/paste exactly)

## 5.1 `src/cnf2vec/geometry.py`

```bash
cat > src/cnf2vec/geometry.py << 'EOF'
import math
import numpy as np
from shapely.affinity import rotate, translate, scale as shp_scale
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.validation import make_valid


def fix_geometry(geom):
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
    if arr.shape[0] < 2:
        return 0.0
    X = arr - arr.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v0 = Vt[0]
    return math.atan2(float(v0[1]), float(v0[0]))


def cnf_normalize(geom):
    # 1) translate to centroid
    cx, cy = centroid_xy(geom)
    g1 = translate(geom, xoff=-cx, yoff=-cy)

    # 2) rotate using PCA direction of boundary/coords
    arr = boundary_coords(g1)
    theta = pca_theta_from_coords(arr)
    g2 = rotate(g1, angle=-math.degrees(theta), origin=(0, 0), use_radians=False)

    # 3) scale to fit in [-1,1] (max half-extent -> 1)
    minx, miny, maxx, maxy = g2.bounds
    half_extent = max(max(abs(minx), abs(maxx)), max(abs(miny), abs(maxy)))
    if half_extent > 0:
        g3 = shp_scale(g2, xfact=1 / half_extent, yfact=1 / half_extent, origin=(0, 0))
    else:
        g3 = g2

    return g3
