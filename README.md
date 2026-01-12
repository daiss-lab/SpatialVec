# SpatialVec (CNF2Vec pipeline)

SpatialVec is a research-oriented pipeline for learning geometry-aware representations of geospatial objects (points, polylines, and polygons) using a Canonical Normalized Frame (CNF) formulation.

The project implements an end-to-end workflow that:
- normalizes raw GIS geometries into a canonical coordinate frame,
- samples unsigned distance (UDF) and occupancy (OCC) fields on a regular grid with adaptive boundary refinement,
- trains a neural network to learn stable, geometry-aware features.

The repository is designed for reproducible GeoAI research, not for hosting datasets or pretrained models.

---

## Repository structure

SpatialVec/
- src/cnf2vec/        (core implementation: geometry, sampling, model, training)
- scripts/            (one-command run scripts)
- data/               (local datasets, not tracked)
- outputs/            (generated samples and model checkpoints, not tracked)
- requirements.txt
- README.md

Note:  
The data and outputs directories are intentionally empty on GitHub.  
They are populated automatically when the pipeline is run locally.

---

## Setup

Recommended (Conda – best for GeoPandas on Windows):

conda create -n spatialvec python=3.10 -y  
conda activate spatialvec  
conda install -c conda-forge geopandas shapely -y  
pip install -r requirements.txt  

Alternative (pip only):

pip install -r requirements.txt  

---

## Data

This repository does **not** include any datasets to keep the codebase lightweight and reproducible.

The experiments in this project use the **NYC geospatial dataset** originally released with the *GeoNeuralRepresentation* project. The dataset is publicly available [here](https://github.com/chuchen2017/GeoNeuralRepresentation/blob/master/data/NYC_total_data.gpkg). Place the downloaded NYC_total_data.gpkg file under the data/ directory.


---
## Running the pipeline

From the repository root, run:

PYTHONPATH=src python -m cnf2vec.main  
--gpkg_path data/NYC_total_data.gpkg  
--layer NYC_total_data  
--out_root outputs/nyc  
--n_per_type 2000  
--grid_n 64  
--max_refine 2000  
--epochs 10  
--batch_size 4  
--lr 1e-3  
--sample_ratio 0.5  

Or simply run:

./scripts/run_train_nyc.sh

---

## What happens when you run it

Running the pipeline executes the full workflow automatically:

1. Loads geospatial objects from the input GeoPackage  
2. Filters and balances points, lines, and polygons  
3. Applies canonical normalization (translation, rotation, scaling)  
4. Samples UDF and OCC values on a regular grid  
5. Refines samples near object boundaries using adaptive bisection  
6. Saves sampled data to outputs/<run_name>/samples/  
7. Trains a neural network on the sampled representations  
8. Saves a model checkpoint to outputs/<run_name>/model.pt  

No intermediate manual steps are required.

---

## Outputs

All generated artifacts are stored locally under outputs:

outputs/nyc/
- samples/        (XY, UDF, OCC numpy arrays)
- model.pt        (trained model checkpoint)

These files are not committed to GitHub and are expected to differ across runs.

---

## Design philosophy

- Clear separation of geometry processing, sampling, modeling, and training  
- Fully reproducible, end-to-end pipeline  
- Dataset-agnostic (any GeoPackage with point, line, and polygon geometries)  
- Research-friendly and easy to extend  

  

---

## License / Usage

This repository is intended for academic and research use.  
Please ensure you have the appropriate rights to use any dataset you provide locally.
