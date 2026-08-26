# githubspaitalvec

Reference implementation of **SpatialVec embedding generation**: a boundary-aware
adaptive sampler that turns raw vector geometry into a sampled implicit-field
representation, and a transformer encoder that compresses that representation
into one embedding per geometry.

This repository contains the embedding-generation stage only. Downstream
evaluation (topological relation classification, distance estimation, land-use
inference, range-query selectivity) is not included.

---

## 1. Method

Every object is first mapped to a canonical frame: translated to its vertex mean
and divided by its bounding-box diagonal. All sampling and all field labels are
computed in that frame, so an object's representation is invariant to its
absolute position and scale.

### 1.1 Boundary-aware sampling

The sampler assigns each object a budget that grows with its geometric
complexity, then splits that budget across three regions whose weights also
depend on the object's shape.

**Per-object budget.** With canonical boundary length `l_E`, canonical enclosed
area `A_E`, and areal indicator `d_E` (1 for polygons, 0 otherwise):

```
N_E = min( N_max , max( N_min , ceil( N_0 * (1 + lam_l * l_E + lam_a * d_E * sqrt(A_E)) ) ) )
```

**Region weights and allocation.**

```
w_bnd   = 1 + lam_b * l_E
w_int   = d_E * (1 + lam_i * sqrt(A_E))
w_space = 1
W       = w_bnd + w_int + w_space

N_bnd   = floor( N_E * w_bnd / W )
N_int   = floor( N_E * w_int / W )
N_space = N_E - N_bnd - N_int
```

`N_space` absorbs the remainder of both floors, so the three parts sum to `N_E`
exactly. Non-areal types have `w_int = 0`, and their interior share falls
through to the outer-space region.

**Query sets.** `N_bnd` points are drawn along the outline with
length-proportional segment probability and a shell offset; `N_int` points are
drawn inside the ring by rejection sampling (polygons only); `N_space` points
are drawn uniformly from the ambient box `[-clip, clip]^2`. The union is
shuffled, and the implicit field — signed distance `phi_E`, occupancy `o_E`,
boundary indicator `b_E` — is evaluated at every query point.

Points have no outline, so their `N_bnd` share is drawn on concentric shells
around the canonical origin instead.

### 1.2 Encoder

Each query point becomes a token carrying its Fourier-encoded canonical
coordinate, its clipped signed distance, its occupancy and boundary flags, and a
one-hot distance band. Tokens are summed with a projection of the 17-dimensional
object metadata and a Fourier location encoding, then passed through a
pre-norm transformer encoder. Nine pooled views of the token sequence — CLS,
mean, max, attention-pooled boundary, transformed boundary, mid-band, far-band,
metadata and location — are fused into the code `z`.

Training is reconstruction only: the decoder must recover `phi_E`, `o_E` and
`b_E` at every query point from `z` and the per-token state, under an
inverse-distance weighting that concentrates the distance term near the
boundary. The exported vector is `[z | meta | location Fourier features]`,
477 dimensions at the default `hidden_dim = 256`.

### 1.3 Variable budgets, fixed tensors

Objects carry different budgets, so the bank pads the query axis and marks live
rows with `mask`. The mask is threaded through the encoder as a key-padding
mask, through every pooling branch as a normaliser, and through the loss as a
per-token weight — a batch is additionally trimmed to the longest live sequence
it actually contains, so padding costs no attention.

---

## 2. Calibration

`lam_l`, `lam_a`, `N_min` and `N_max` set the *shape* of the budget curve; `N_0`
sets its level. `scripts/calibrate.py` solves for the `N_0` that puts the corpus
mean budget on a target, which makes an adaptive-versus-fixed comparison
compute-matched: the rule redistributes queries rather than buying more.

On the 59,933-object NYC corpus, targeting the fixed scheme's `N = 1024`:

| parameter | value |
|---|---|
| `N_0` | 709 |
| `N_min`, `N_max` | 256, 2048 |
| `lam_l`, `lam_a` | 0.25, 1.00 |
| `lam_b`, `lam_i` | 0.55, 0.50 |

**Budget distribution**

| type | n | mean `N_E` | p1 | p50 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| polygon | 20,000 | 1476.9 | 1352 | 1466 | 1664 | 2048 |
| polyline | 20,000 | 887.4 | 887 | 887 | 895 | 973 |
| point | 19,933 | 709.0 | 709 | 709 | 709 | 709 |
| **all** | **59,933** | **1024.8** | 709 | 887 | 1629 | 2048 |

Total 61,417,759 queries against 61,371,392 for a flat `N = 1024`: **+0.08%**.
One object reaches the `N_max` clamp; none reach `N_min`.

**Allocation shares**

| type | boundary | interior | outer space |
|---|---:|---:|---:|
| polygon | 0.499 | 0.279 | 0.222 |
| polyline | 0.608 | — | 0.392 |
| point | 0.499 | — | 0.501 |

`lam_b = 0.55` is chosen so the polygon boundary share sits on 0.50, matching the
fixed `(0.50, 0.25, 0.25)` split it replaces; `lam_i = 0.50` shifts roughly three
points from outer space to the interior on polygons. Points are unchanged.
Polylines move about eleven points toward the boundary, which is where a
two-vertex segment's information actually is.

### Where the rule does and does not discriminate

The adaptive term does real work *within* the polygon class (1352 to 2048 across
the distribution). NYC polylines are overwhelmingly two-vertex road segments
(`l_E` mean 1.003, p99 1.047) and points have `l_E = A_E = 0` by construction, so
both sit at an essentially constant budget on this corpus. A corpus with long
multi-vertex polylines — coastlines, rivers, transit lines — would widen the
polyline spread. Recalibrate `N_0` per corpus with `scripts/calibrate.py`.

---

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
