from __future__ import annotations

import torch


# ============================================================
# ============= Normalization/Sparsification functions =======
# ============================================================
def row_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_sum = A.sum(dim=-1, keepdim=True)
    return A / (row_sum + eps)


def normalize(t: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    mean = t.mean(dim=dim, keepdim=True)
    std = t.std(dim=dim, unbiased=False, keepdim=True).clamp(min=eps)
    return (t - mean) / std


def minmax_normalize(t: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    t_min = t.min(dim=dim, keepdim=True).values
    t_max = t.max(dim=dim, keepdim=True).values

    # avoid division-by-zero if t_max == t_min
    denom = (t_max - t_min).clamp(min=eps)

    return (t - t_min) / denom


def symmetry_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    d = A.sum(dim=-1)
    d_inv_sqrt = torch.pow(d + eps, -0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


# ============================================================
# ============= useful functions =============================
# ============================================================
def gaussian_kernel(distance: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    return torch.exp(- (distance ** 2) / (2 * sigma ** 2))


def directed_dirichlet_energy(
    Z: torch.Tensor,
    A_directed: torch.Tensor,
    nodewise: bool = False,
) -> torch.Tensor:
    """
    Compute directional (asymmetric) Dirichlet energy.

    Parameters
    ----------
    Z : [N, D] node embeddings
    A_directed : [N, N] directed adjacency matrix
    nodewise : bool
        If True, return per-node energy; otherwise scalar for the whole graph.

    Returns
    -------
    energy : scalar or [N] tensor
        Dirichlet energy.
    """
    # Compute pairwise squared differences ||z_i - z_j||^2
    pd = ((Z.unsqueeze(1) - Z.unsqueeze(0)) ** 2).sum(-1)  # [N, N]

    # Weight by adjacency
    weighted_pd = A_directed * pd  # [N, N]

    if nodewise:
        # Node-wise energy: sum over outgoing edges
        node_energy = 0.5 * weighted_pd.sum(dim=1)  # [N]
        return node_energy

    # Scalar energy: sum over all edges
    energy = 0.5 * weighted_pd.sum()
    return energy


def spatial_dirichlet_energy(
    Z: torch.Tensor,
    A_sym: torch.Tensor,
    nodewise: bool = False,
) -> torch.Tensor:
    """
    Compute symmetric Dirichlet energy (normalized by degree).

    Parameters
    ----------
    Z : [N, D] node embeddings
    A_sym : [N, N] symmetric adjacency matrix
    nodewise : bool
        If True, return per-node energy; otherwise scalar for the whole graph.

    Returns
    -------
    energy : scalar or [N] tensor
        Dirichlet energy.
    """
    # Compute Laplacian
    I = torch.eye(A_sym.size(0), device=A_sym.device, dtype=A_sym.dtype)
    L = I - A_sym  # normalized Laplacian

    if nodewise:
        # Node-wise energy: (L*Z) · Z per row
        node_energy = torch.sum((L @ Z) * Z, dim=1)  # [N]
        return node_energy

    # Scalar energy: trace(Z^T L Z)
    energy = torch.trace(Z.T @ (L @ Z))
    return energy


def pairwise_energy(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    X: [B, W, N, F]
    A: [B, W, N, N]
    Returns: E: [B, W, N]
    Computes nodewise directed Dirichlet energy:
        E[n] = sum_m A[n,m] * ||X[n] - X[m]||^2
    Fully vectorized.
    """
    # ||X||^2 term
    X2 = (X * X).sum(dim=-1, keepdim=True)         # [B,W,N,1]
    # X·Xᵀ term
    XY = X @ X.transpose(-1, -2)                   # [B,W,N,N]
    # squared distances
    D2 = X2 + X2.transpose(-1, -2) - 2 * XY        # [B,W,N,N]

    # multiply by adjacency
    E = (A * D2).sum(dim=-1)                       # [B,W,N]
    return E


def embedding_variance(Z: torch.Tensor) -> torch.Tensor:
    """Encourage spread in embedding space."""
    mean_z = Z.mean(dim=0, keepdim=True)
    return ((Z - mean_z) ** 2).sum() / Z.shape[0]


def topk_row(A: torch.Tensor, k: int, sym: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Top-k refinement for adjacency matrices.
    Supports A: [N, N] or A: [B, N, N].

    Args:
        A: adjacency matrix (2D or 3D)
        k: number of neighbors to keep
        sym: enforce symmetry (A ← (A+Aᵀ)/2) before normalization
        eps: numerical stability

    Returns:
        Refined adjacency of the same shape as A
    """
    if A.ndim == 2:
        # Add batch dimension: [1, N, N]
        A = A.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    B, N, N2 = A.shape
    assert N == N2, "Adjacency must be square"

    k = min(k, N)

    # ---- Top-k per row ----
    vals, idx = torch.topk(A, k, dim=2)      # [B, N, k]
    out = torch.zeros_like(A)                # [B, N, N]
    out.scatter_(2, idx, vals)               # put top-k values in place

    # ---- Symmetry optional ----
    if sym:
        out = 0.5 * (out + out.transpose(1, 2))  # [B,N,N]
        out = symmetry_normalize(out, eps)
    else:
        out = row_normalize(out, eps)

    # ---- Remove batch dimension if input was 2D ----
    if squeeze_back:
        out = out.squeeze(0)

    return out


def corrcoef(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + eps)
