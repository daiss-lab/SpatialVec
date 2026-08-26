import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import META_DIM, N_SPATIAL_FEATS, SPATIAL_META_SLICE, ModelConfig


class FourierEncoding(nn.Module):
    def __init__(self, in_dim, freqs):
        super().__init__()
        self.register_buffer(
            "freqs", torch.tensor(freqs, dtype=torch.float32) * math.pi
        )
        self.out_dim = in_dim * (1 + 2 * len(freqs))

    def forward(self, x):
        x = x.float()
        projected = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, torch.sin(projected).flatten(-2), torch.cos(projected).flatten(-2)],
            dim=-1,
        )


class SpatialLocationEncoder(nn.Module):
    def __init__(self, out_dim, freqs):
        super().__init__()
        self.pe = FourierEncoding(N_SPATIAL_FEATS, freqs)
        self.proj = nn.Sequential(
            nn.Linear(self.pe.out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, spatial_feats):
        return self.proj(self.pe(spatial_feats))


def spatial_export_pe(meta, freqs):
    feats = meta[:, SPATIAL_META_SLICE].float()
    scaled = torch.tensor(freqs, dtype=torch.float32, device=feats.device) * math.pi
    projected = feats.unsqueeze(-1) * scaled
    return torch.cat(
        [torch.sin(projected).flatten(-2), torch.cos(projected).flatten(-2)], dim=-1
    )


class SpatialVec(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = ModelConfig() if cfg is None else cfg
        self.cfg = cfg

        hidden = cfg.hidden_dim
        self.pe = FourierEncoding(2, cfg.canon_freqs)
        token_dim = self.pe.out_dim + 1 + 1 + 1 + 3 + 1

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.meta_proj = nn.Sequential(
            nn.Linear(META_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.spatial_loc_encoder = SpatialLocationEncoder(hidden, cfg.loc_freqs)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg.n_heads,
            dim_feedforward=hidden * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.boundary_pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.boundary_transform = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.z_fuse = nn.Sequential(
            nn.Linear(hidden * 9, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.sdf_head = nn.Linear(hidden, 1)
        self.occ_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)

    @property
    def export_dim(self):
        return self.cfg.export_dim

    def encode(self, xy, sdf, occ, boundary, mask, meta):
        batch, n_query, _ = xy.shape

        canon_pe = self.pe(xy)
        meta_h = self.meta_proj(meta)
        spatial_h = self.spatial_loc_encoder(meta[:, SPATIAL_META_SLICE])

        observed = torch.ones(batch, n_query, 1, device=xy.device)
        sdf_clip = torch.clamp(sdf, -1.0, 1.0).unsqueeze(-1)

        abs_sdf = sdf.abs()
        band = torch.zeros_like(sdf, dtype=torch.long)
        band[
            (abs_sdf > self.cfg.near_threshold) & (abs_sdf <= self.cfg.mid_threshold)
        ] = 1
        band[abs_sdf > self.cfg.mid_threshold] = 2
        band_onehot = F.one_hot(band, num_classes=3).float()

        tokens = torch.cat(
            [
                canon_pe,
                sdf_clip,
                occ.unsqueeze(-1),
                boundary.unsqueeze(-1),
                band_onehot,
                observed,
            ],
            dim=-1,
        )

        h = self.token_proj(tokens)
        h = h + meta_h.unsqueeze(1) + spatial_h.unsqueeze(1)

        cls = self.cls.expand(batch, -1, -1)
        h_all = torch.cat([cls, h], dim=1)

        cls_mask = torch.ones(batch, 1, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        h_all = self.encoder(h_all, src_key_padding_mask=full_mask == 0)

        cls_h = h_all[:, 0]
        pt_h = h_all[:, 1:]

        m = mask.unsqueeze(-1)
        mean_h = (pt_h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        max_h = pt_h.masked_fill(m == 0, -1e4).max(dim=1).values

        bnd_logits = self.boundary_pool_attn(pt_h).squeeze(-1)
        bnd_logits = bnd_logits.masked_fill(boundary <= 0.5, -1e4)
        bnd_logits = bnd_logits.masked_fill(mask == 0, -1e4)
        bnd_alpha = F.softmax(bnd_logits, dim=1)
        boundary_pool_h = (pt_h * bnd_alpha.unsqueeze(-1)).sum(1)

        bnd_mask = ((boundary > 0.5) & (mask > 0)).float().unsqueeze(-1)
        bnd_tokens = self.boundary_transform(pt_h) * bnd_mask
        boundary_enc_h = bnd_tokens.sum(1) / bnd_mask.sum(1).clamp_min(1)

        mid_w = ((band == 1) & (mask > 0)).float().unsqueeze(-1)
        far_w = ((band == 2) & (mask > 0)).float().unsqueeze(-1)
        mid_h = (pt_h * mid_w).sum(1) / mid_w.sum(1).clamp_min(1)
        far_h = (pt_h * far_w).sum(1) / far_w.sum(1).clamp_min(1)

        z = self.z_fuse(
            torch.cat(
                [
                    cls_h,
                    mean_h,
                    max_h,
                    boundary_pool_h,
                    boundary_enc_h,
                    mid_h,
                    far_h,
                    meta_h,
                    spatial_h,
                ],
                dim=-1,
            )
        )

        return z, pt_h

    def reconstruct(self, pt_h, z):
        n_query = pt_h.shape[1]
        z_expand = z.unsqueeze(1).expand(-1, n_query, -1)
        recon_h = self.recon_decoder(torch.cat([pt_h, z_expand], dim=-1))
        return (
            self.sdf_head(recon_h).squeeze(-1),
            self.occ_head(recon_h).squeeze(-1),
            self.boundary_head(recon_h).squeeze(-1),
        )

    def export_features(self, z, meta):
        return torch.cat(
            [z.float(), meta.float(), spatial_export_pe(meta, self.cfg.loc_freqs)],
            dim=-1,
        )

    def forward(self, batch):
        z, pt_h = self.encode(
            batch["xy"],
            batch["sdf"],
            batch["occ"],
            batch["boundary"],
            batch["mask"],
            batch["meta"],
        )
        sdf_pred, occ_logits, boundary_logits = self.reconstruct(pt_h, z)
        return sdf_pred, occ_logits, boundary_logits, z
