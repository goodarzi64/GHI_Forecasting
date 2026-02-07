from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .GST_Utilities import row_normalize, symmetry_normalize, topk_row
import torch.nn as nn


class GeoGeometry:
    def __init__(self, df_geo, device: str = "cpu") -> None:
        self.df_geo = df_geo
        self.device = device
        self._build()

    def _build(self) -> None:
        try:
            from pyproj import Geod
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyproj is required for GeoGeometry. Install with `pip install pyproj`."
            ) from exc

        n = self.df_geo.shape[0]
        geod = Geod(ellps="WGS84")

        dist = np.zeros((n, n), dtype=np.float32)
        theta = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            lat1, lon1 = self.df_geo.iloc[i]["latitude"], self.df_geo.iloc[i]["longitude"]
            for j in range(n):
                if i == j:
                    continue
                lat2, lon2 = self.df_geo.iloc[j]["latitude"], self.df_geo.iloc[j]["longitude"]
                az12, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
                dist[i, j] = dist_m / 1000.0
                theta[i, j] = np.radians(az12)

        self.dist_matrix = torch.tensor(dist, dtype=torch.float32, device=self.device)
        self.theta_matrix = torch.tensor(theta, dtype=torch.float32, device=self.device)


class DistanceKernel:
    def __init__(self, dist_matrix: torch.Tensor, sigma: torch.Tensor | float | None = None) -> None:
        self.dist_matrix = dist_matrix
        self.sigma = sigma or self._estimate_sigma()

    def _estimate_sigma(self) -> torch.Tensor:
        mask = torch.triu(torch.ones_like(self.dist_matrix), diagonal=1).bool()
        return torch.std(self.dist_matrix[mask])

    def compute(self, self_loops: bool = False) -> torch.Tensor:
        A = torch.exp(- (self.dist_matrix ** 2) / (2 * self.sigma ** 2))
        if not self_loops:
            A.fill_diagonal_(0)
        return A


def build_static_adjacency(
    df_geo,
    device: str = "cpu",
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build distance/theta matrices and common static adjacency variants.

    Returns a dict with:
      - dist_matrix
      - theta_matrix
      - A_raw
      - A_topk
      - A_row_norm
      - A_sym_norm
    """
    geo = GeoGeometry(df_geo, device=device)
    dist_matrix = geo.dist_matrix
    theta_matrix = geo.theta_matrix

    kernel = DistanceKernel(dist_matrix)
    A_raw = kernel.compute(self_loops=self_loops)

    return {
        "dist_matrix": dist_matrix,
        "theta_matrix": theta_matrix,
        "A_raw": A_raw,
        "A_topk": topk_row(A_raw, k=k, sym=topk_sym),
        "A_row_norm": row_normalize(A_raw),
        "A_sym_norm": symmetry_normalize(A_raw),
    }


class WindAdjacency(nn.Module):
    """
    Build A_wind(t) from static distance/bearing + time-varying wind dir/speed.
    Supports batched input: wind_feats [B, N, F].
    Produces row-stochastic adjacency per batch.
    Important: A_wind[i,j] is how much node i influences node j along wind.
    """

    def __init__(
        self,
        D_ij: torch.Tensor,
        Theta_ij: torch.Tensor,
        R: float = 150.0,
        lambda_theta: float = 1.0,
        cone_half_angle: float | None = None,
        wind_speed_pos: int = 0,
        wind_dir_pos: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer("D_ij", D_ij)         # [N,N]
        self.register_buffer("Theta_ij", Theta_ij) # [N,N]

        self.R = float(R)
        self.lambda_theta = float(lambda_theta)
        self.cone_half_angle = cone_half_angle

        self.wind_speed_pos = wind_speed_pos
        self.wind_dir_pos = wind_dir_pos

    @staticmethod
    def angdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # works with broadcasting (batched or non-batched)
        return (a - b + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(
        self,
        wind_feats: torch.Tensor,
        sparse: bool = False,
        k: int = 5,
        self_loops: bool = False,
    ) -> torch.Tensor:
        # ---------------------------
        # Handle shapes
        # ---------------------------
        if wind_feats.dim() == 2:
            # [N,F] → [1,N,F], later squeeze back
            wind_feats = wind_feats.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, _F = wind_feats.shape

        # ---------------------------
        # Extract wind components
        # ---------------------------
        wind_speed = wind_feats[..., self.wind_speed_pos]  # [B,N]
        wind_dir = wind_feats[..., self.wind_dir_pos]      # [B,N]

        # Convert meteorological (from) → movement (to)
        wind_to = (wind_dir + torch.pi) % (2 * torch.pi)    # [B,N]

        # Expand to [B,N,N]
        dir_mat = wind_to.unsqueeze(-1).expand(B, N, N)     # rows = i

        # Static fields to [B,N,N]
        D_ij = self.D_ij.unsqueeze(0).expand(B, N, N)
        Theta_ij = self.Theta_ij.unsqueeze(0).expand(B, N, N)

        # ---------------------------
        # Alignment term
        # ---------------------------
        # Theta_ij - wind_to[i] → positive if j is downstream from i according to wind
        ang = self.angdiff(Theta_ij, dir_mat)
        align = torch.cos(ang).clamp(min=0.0)               # [B,N,N]

        # ---------------------------
        # Base weight (distance + alignment)
        # ---------------------------
        base = torch.exp(-D_ij / self.R) * torch.exp(align / self.lambda_theta)

        # ---------------------------
        # Cone restriction (optional)
        # ---------------------------
        if self.cone_half_angle is not None:
            cone_mask = (ang.abs() <= self.cone_half_angle)
            base = base * cone_mask.float()

        # ---------------------------
        # Multiply by wind speed_i
        # ---------------------------
        base = base * wind_speed.unsqueeze(-1)              # [B,N,1] → [B,N,N]

        # ---------------------------
        # self loop connection
        # ---------------------------
        if not self_loops:
            base.diagonal(dim1=-2, dim2=-1).zero_()

        # ---------------------------
        # Sparse or dense?
        # ---------------------------
        if sparse:
            A = []
            for b in range(B):
                A_b = topk_row(base[b], k=k, sym=False)     # returns [N,N]
                A.append(A_b)
            A = torch.stack(A, dim=0)                       # [B,N,N]
        else:
            A = base / (base.sum(dim=-1, keepdim=True) + 1e-8)

        # ---------------------------
        # If original input was unbatched → squeeze
        # ---------------------------
        if squeeze:
            A = A[0]

        return A
