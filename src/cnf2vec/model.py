import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim=2, num_freqs=8, include_input=True):
        super().__init__()
        self.include_input = include_input
        freq_bands = (2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)) * math.pi
        self.register_buffer("freq_bands", freq_bands)
        out_dim = (in_dim if include_input else 0) + 2 * in_dim * num_freqs
        self.out_dim = out_dim

    def forward(self, x):
        xb = x[..., None, :] * self.freq_bands[:, None]  # (..., L, 2)
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        pe = torch.cat([sin, cos], dim=-1).flatten(-2)
        if self.include_input:
            pe = torch.cat([x, pe], dim=-1)
        return pe


class CNF2VecModel(nn.Module):
    """
    Inputs:  (x,y)
    Outputs:
      udf_pred (B,N)
      occ_logits (B,N)
      hidden features (B,N,H)
    """
    def __init__(self, in_dim=2, hidden_dim=256, layers=5, dropout=0.1, use_pe=True, pe_freqs=8):
        super().__init__()
        self.use_pe = use_pe
        if use_pe:
            self.pe = PositionalEncoding(in_dim, pe_freqs)
            in_dim = self.pe.out_dim

        net = []
        last = in_dim
        for _ in range(layers):
            net += [nn.Linear(last, hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout)]
            last = hidden_dim

        self.mlp = nn.Sequential(*net)
        self.udf_head = nn.Linear(hidden_dim, 1)
        self.occ_head = nn.Linear(hidden_dim, 1)

    def forward(self, xy):
        B, N, _ = xy.shape
        x = xy.reshape(B * N, -1)
        if self.use_pe:
            x = self.pe(x)
        h = self.mlp(x)
        udf = F.relu(self.udf_head(h)).view(B, N)
        occ = self.occ_head(h).view(B, N)
        h = h.view(B, N, -1)
        return udf, occ, h
