from dataclasses import dataclass

META_DIM = 17
SPATIAL_META_SLICE = slice(3, 9)
N_SPATIAL_FEATS = 6

MAX_ASPECT_RATIO = 100.0
MAX_FILL_RATIO = 10.0
MAX_COMPACTNESS = 1.5

GEOMETRY_TYPES = ("polygon", "polyline", "point")


@dataclass(frozen=True)
class SamplingConfig:
    n0: int = 709
    n_min: int = 256
    n_max: int = 2048
    lambda_ell: float = 0.25
    lambda_area: float = 1.00
    lambda_bnd: float = 0.55
    lambda_int: float = 0.50
    clip: float = 1.15
    boundary_threshold: float = 0.025
    point_canon_scale: float = 2.0e-4
    max_reject_factor: int = 200

    def __post_init__(self):
        if self.n0 <= 0:
            raise ValueError("n0 must be positive")
        if not 0 < self.n_min <= self.n_max:
            raise ValueError("require 0 < n_min <= n_max")
        for name in ("lambda_ell", "lambda_area", "lambda_bnd", "lambda_int"):
            if getattr(self, name) < 0.0:
                raise ValueError(name + " must be non-negative")
        if self.clip <= 0.0:
            raise ValueError("clip must be positive")
        if self.boundary_threshold <= 0.0:
            raise ValueError("boundary_threshold must be positive")
        if self.max_reject_factor < 1:
            raise ValueError("max_reject_factor must be at least 1")

    def as_dict(self):
        return {
            "n0": self.n0,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "lambda_ell": self.lambda_ell,
            "lambda_area": self.lambda_area,
            "lambda_bnd": self.lambda_bnd,
            "lambda_int": self.lambda_int,
            "clip": self.clip,
            "boundary_threshold": self.boundary_threshold,
            "point_canon_scale": self.point_canon_scale,
            "max_reject_factor": self.max_reject_factor,
        }


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.10
    canon_freqs: tuple = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    loc_freqs: tuple = tuple(float(2 ** k) for k in range(17))
    near_threshold: float = 0.06
    mid_threshold: float = 0.22

    @property
    def export_pe_dim(self):
        return N_SPATIAL_FEATS * 2 * len(self.loc_freqs)

    @property
    def export_dim(self):
        return self.hidden_dim + META_DIM + self.export_pe_dim

    def as_dict(self):
        return {
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "canon_freqs": list(self.canon_freqs),
            "loc_freqs": list(self.loc_freqs),
            "near_threshold": self.near_threshold,
            "mid_threshold": self.mid_threshold,
            "export_dim": self.export_dim,
        }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    sdf_weight: float = 1.00
    occ_weight: float = 0.55
    boundary_weight: float = 0.70
    code_weight: float = 1.0e-4
    sdf_loss_eps: float = 0.02
    sdf_loss_clamp: float = 10.0
    num_workers: int = 0
    seed: int = 42

    def as_dict(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "grad_clip": self.grad_clip,
            "sdf_weight": self.sdf_weight,
            "occ_weight": self.occ_weight,
            "boundary_weight": self.boundary_weight,
            "code_weight": self.code_weight,
            "sdf_loss_eps": self.sdf_loss_eps,
            "sdf_loss_clamp": self.sdf_loss_clamp,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }
