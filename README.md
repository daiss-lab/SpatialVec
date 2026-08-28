# SpatialVec: A Unified Representation Learning Framework for Geospatial Objects with Boundary-Aware Sampling

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Main dependencies are `numpy`, `torch`, `geopandas`, `shapely`, and `tqdm`.

A GPU is optional. If one is not available, the encoder runs on CPU. Mixed precision is only used when `bfloat16` is supported.

---

## Usage

### End to end

```bash
python generate_embeddings.py \
  --input data/NYC_total_data.gpkg \
  --out outputs/nyc \
  --sample-workers 8
```

Any vector format supported by GeoPandas can be used. Points, lines, and polygons are supported. For multi-part geometries, the largest part is used.

### Two stages

Sampling mainly uses the CPU, while training benefits from a GPU. They can be run separately:

```bash
python scripts/build_bank.py \
  --input data/NYC_total_data.gpkg \
  --out outputs/nyc/geometry_bank.pt \
  --workers 16 \
  --summary outputs/nyc/sampling_summary.json

python scripts/train_embeddings.py \
  --bank outputs/nyc/geometry_bank.pt \
  --out outputs/nyc
```

### Calibrating a new dataset

```bash
python scripts/calibrate.py \
  --input data/Singapore.gpkg \
  --target-mean 1024 \
  --out calibration_singapore.json
```

Use the reported `N_0` value with `--n0`.

### Slurm

```bash
INPUT=data/NYC_total_data.gpkg OUTPUT=outputs/nyc \
VENV=/path/to/venv sbatch scripts/run_embeddings.sh
```

---

## Outputs

| File                      | Description                               |
| ------------------------- | ----------------------------------------- |
| `geometry_bank.pt`        | Sampled geometry data and metadata        |
| `sampling_summary.json`   | Sampling statistics                       |
| `best_model.pt`           | Trained encoder weights and configuration |
| `training_history.json`   | Training losses and accuracies            |
| `Z_unique_geometries.npy` | Embeddings for unique geometries          |
| `entity_embeddings.npz`   | Embeddings for all input rows             |
| `embedding_layout.json`   | Layout of the final embedding             |

Duplicate geometries are sampled only once, then mapped back to the original input rows.

### Memory

With `N_max = 2048`, the NYC geometry bank uses about **2.9 GB** of memory.

Lower `--n-max` if memory usage is too high. You can also use `--pad-to` to set the query length directly.

---

## Configuration

| Flag                      |   Default | Description                  |
| ------------------------- | --------: | ---------------------------- |
| `--n0`                    |       709 | Base sampling budget         |
| `--n-min`, `--n-max`      | 256, 2048 | Minimum and maximum budgets  |
| `--lambda-ell`            |      0.25 | Boundary-length term         |
| `--lambda-area`           |      1.00 | Area term                    |
| `--lambda-bnd`            |      0.55 | Boundary sampling weight     |
| `--lambda-int`            |      0.50 | Interior sampling weight     |
| `--hidden-dim`            |       256 | Encoder hidden size          |
| `--n-heads`, `--n-layers` |      8, 6 | Encoder settings             |
| `--epochs`                |        30 | Training epochs              |
| `--batch-size`            |        32 | Training batch size          |
| `--sample-workers`        |         1 | Number of sampling processes |

The default exported embedding size is **477 dimensions**.

If GPU memory is limited, try reducing `--batch-size`.

Setting:

```text
lam_l = 0
lam_a = 0
```

gives every geometry the same sampling budget.

Setting:

```text
lam_b = 0
lam_i = 0
```

also removes shape-dependent sampling weights.

---

## Reproducibility

Each geometry is sampled using:

```python
default_rng([seed, object_index])
```

This keeps sampling results the same even when `--sample-workers` changes.

Python, NumPy, and PyTorch are also seeded using `--seed`.

Geometry calculations are performed in `float64` for better numerical stability.
