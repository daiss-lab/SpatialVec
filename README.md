# githubspaitalvec

## 3. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires numpy, torch, geopandas, shapely and tqdm. A GPU is optional; the
encoder falls back to CPU. Automatic mixed precision is enabled only when the
device supports bfloat16 — float16 is never used, because raw projected
coordinates and large Fourier products overflow its range.

---

## 4. Usage

### End to end

```bash
python generate_embeddings.py \
  --input data/NYC_total_data.gpkg \
  --out outputs/nyc \
  --sample-workers 8
```

Any vector format geopandas can read works. Polygons, polylines and points are
supported; multi-part geometries are reduced to their largest part.

### Two stages

Sampling is CPU-bound and the encoder is GPU-bound, so they are worth running
separately on a cluster.

```bash
python scripts/build_bank.py \
  --input data/NYC_total_data.gpkg \
  --out outputs/nyc/geometry_bank.pt \
  --workers 16 --summary outputs/nyc/sampling_summary.json

python scripts/train_embeddings.py \
  --bank outputs/nyc/geometry_bank.pt \
  --out outputs/nyc
```

### Calibrating a new corpus

```bash
python scripts/calibrate.py \
  --input data/Singapore.gpkg \
  --target-mean 1024 \
  --out calibration_singapore.json
```

Pass the reported `N_0` back through `--n0`.

### Slurm

```bash
INPUT=data/NYC_total_data.gpkg OUTPUT=outputs/nyc \
VENV=/path/to/venv sbatch scripts/run_embeddings.sh
```

---

## 5. Outputs

| file | contents |
|---|---|
| `geometry_bank.pt` | sampled implicit field, metadata, budgets, row-to-geometry map |
| `sampling_summary.json` | realised budget and label-coverage statistics |
| `best_model.pt` | encoder weights plus the configs used |
| `training_history.json` | per-epoch losses and accuracies |
| `Z_unique_geometries.npy` | `[n_unique, 477]` — one row per deduplicated geometry |
| `entity_embeddings.npz` | `entity_ids` and `[n_rows, 477]` — one row per input feature |
| `embedding_layout.json` | dimension breakdown of the exported vector |

Identical geometries are deduplicated before sampling; `entity_embeddings.npz`
maps every input row back to its geometry's embedding.

### Memory

The bank is padded to the largest realised budget, so at `N_max = 2048` the NYC
corpus occupies about 2.9 GB with roughly 80% live rows. Lowering `--n-max`
tightens the padding at the cost of clamping more objects; `--pad-to` sets the
query-axis length explicitly.

---

## 6. Configuration

| flag | default | meaning |
|---|---|---|
| `--n0` | 709 | base budget `N_0` |
| `--n-min`, `--n-max` | 256, 2048 | budget clamps |
| `--lambda-ell` | 0.25 | `lam_l`, boundary-length term |
| `--lambda-area` | 1.00 | `lam_a`, area term |
| `--lambda-bnd` | 0.55 | `lam_b`, boundary weight slope |
| `--lambda-int` | 0.50 | `lam_i`, interior weight slope |
| `--hidden-dim` | 256 | encoder width; export dim is `hidden + 17 + 204` |
| `--n-heads`, `--n-layers` | 8, 6 | encoder depth |
| `--epochs` | 30 | training epochs |
| `--batch-size` | 32 | lower this first if the encoder runs out of memory |
| `--sample-workers` | 1 | sampler processes; sampling is embarrassingly parallel |

Setting `lam_l = lam_a = 0` fixes the budget at `N_0` for every object, and
setting `lam_b = lam_i = 0` makes the weights shape-independent: polygons split
`(1/3, 1/3, 1/3)` and non-areal types split `(1/2, 0, 1/2)`. Together they are
the non-adaptive ablation baseline for the sampler.

---

## 7. Reproducibility

Each object is sampled from `default_rng([seed, object_index])`, so a bank is
bit-identical regardless of `--sample-workers`. Training seeds Python, numpy and
torch from `--seed`; residual nondeterminism comes only from cuDNN kernel
selection.

Query geometry is evaluated in float64 throughout: ray-casting containment,
segment distances, and the shoelace area, which is centred before summation so
it does not cancel on projected coordinates.
