import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class CNF2VecDataset(Dataset):
    """
    Loads per-entity XY/UDF/OCC saved by the sampling step.
    """
    def __init__(self, samples_dir, entity_ids=None, sample_ratio=1.0, seed=42):
        self.samples_dir = Path(samples_dir)
        random.seed(seed); np.random.seed(seed)

        all_ids = sorted(int(p.stem.split("_")[1]) for p in self.samples_dir.glob("XY_*.npy"))
        self.entity_ids = all_ids if entity_ids is None else sorted(entity_ids)
        self.sample_ratio = float(sample_ratio)

        self.rows = [
            dict(
                id=eid,
                xy=self.samples_dir / f"XY_{eid:04d}.npy",
                udf=self.samples_dir / f"UDF_{eid:04d}.npy",
                occ=self.samples_dir / f"OCC_{eid:04d}.npy",
            )
            for eid in self.entity_ids
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        XY = np.load(item["xy"]).astype(np.float32)
        UDF = np.load(item["udf"]).astype(np.float32)
        OCC = np.load(item["occ"]).astype(np.float32)

        n_total = XY.shape[0]
        n_keep = max(1, int(n_total * self.sample_ratio))
        if n_keep < n_total:
            sel = np.random.choice(n_total, n_keep, replace=False)
            XY, UDF, OCC = XY[sel], UDF[sel], OCC[sel]

        return {
            "entity_id": item["id"],
            "xy": torch.from_numpy(XY),
            "udf": torch.from_numpy(UDF),
            "occ": torch.from_numpy(OCC),
        }


def variable_length_collate(batch):
    B = len(batch)
    Ns = [b["xy"].shape[0] for b in batch]
    maxN = max(Ns)

    xy   = torch.zeros(B, maxN, 2, dtype=torch.float32)
    udf  = torch.zeros(B, maxN, dtype=torch.float32)
    occ  = torch.zeros(B, maxN, dtype=torch.float32)
    mask = torch.zeros(B, maxN, dtype=torch.float32)
    ids  = torch.tensor([b["entity_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["xy"].shape[0]
        xy[i, :n]   = b["xy"]
        udf[i, :n]  = b["udf"]
        occ[i, :n]  = b["occ"]
        mask[i, :n] = 1.0

    return {"xy": xy, "udf": udf, "occ": occ, "mask": mask, "ids": ids}
