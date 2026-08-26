from .config import ModelConfig, SamplingConfig, TrainConfig
from .bank import build_bank, bank_summary, normalize_meta
from .geometry import canonical_descriptors, canonicalize, load_geometries
from .model import SpatialVec
from .pipeline import export_embeddings, train
from .sampling import (
    allocate,
    boundary_aware_sample,
    implicit_field,
    region_weights,
    sample_boundary,
    sample_budget,
    sample_interior,
    sample_outer_space,
    sampling_plan,
)

__all__ = [
    "ModelConfig",
    "SamplingConfig",
    "TrainConfig",
    "SpatialVec",
    "allocate",
    "bank_summary",
    "boundary_aware_sample",
    "build_bank",
    "canonical_descriptors",
    "canonicalize",
    "export_embeddings",
    "implicit_field",
    "load_geometries",
    "normalize_meta",
    "region_weights",
    "sample_boundary",
    "sample_budget",
    "sample_interior",
    "sample_outer_space",
    "sampling_plan",
    "train",
]

__version__ = "1.0.0"
