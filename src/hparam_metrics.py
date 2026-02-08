import torch


def share_metric_temporal(x_t, z_t, eps=1e-9):
    """
    Compare representation diversity before and after temporal encoding.

    Parameters
    ----------
    x_t : torch.Tensor, [B, W, N, F]
        Raw temporal window input (before encoder).
    z_t : torch.Tensor, [B, N, D]
        Encoded node embeddings (after encoder).

    Returns
    -------
    dict with:
        - share_input : float  → diversity before encoding
        - share_embed : float  → diversity after encoding
        - delta       : float  → share_embed - share_input
        - details     : list of per-batch diagnostic dicts
        - mean_details: dict with mean values of eff_rank, cos_div, rel_var
    """

    B, W, N, F = x_t.shape
    _, _, D = z_t.shape

    # -------------------------------------------------------------
    # === Core metric for one batch ===
    # -------------------------------------------------------------
    def _share_core(Z, Z_ref=None):
        """Compute share metric for [N, D]."""
        N, D = Z.shape
        Zc = Z - Z.mean(dim=0, keepdim=True)

        # ---- 1. Effective rank (normalized spectral entropy) ----
        S = torch.linalg.svdvals(Zc)
        s2 = (S ** 2) + eps
        ps = s2 / s2.sum()
        H = -(ps * (ps + eps).log()).sum()
        eff_rank = torch.exp(H)
        eff_rank_norm = float((eff_rank / D).clamp(0, 1))

        # ---- 2. Cosine diversity ----
        Zn = torch.nn.functional.normalize(Zc, dim=1, eps=1e-8)
        sim = Zn @ Zn.T
        mean_cos = sim[~torch.eye(N, dtype=torch.bool, device=Z.device)].mean()
        cos_div = float((1 - mean_cos).clamp(0, 1))

        # ---- 3. Relative variance ----
        var_dim = Zc.var(dim=0, unbiased=False).mean()
        if Z_ref is not None:
            Zr = Z_ref - Z_ref.mean(dim=0, keepdim=True)
            var_ref = Zr.var(dim=0, unbiased=False).mean()
            rel_var = float((var_dim / (var_ref + eps)).clamp(0, 1))
        else:
            rel_var = float((var_dim / (var_dim + 1e-6)).clamp(0, 1))

        share = (eff_rank_norm + cos_div + rel_var) / 3.0
        return share, eff_rank_norm, cos_div, rel_var

    # -------------------------------------------------------------
    # === Compute metrics batch-wise ===
    # -------------------------------------------------------------
    x_flat = x_t.mean(dim=1).reshape(-1, N, F)  # [B, N, F]
    z_flat = z_t.reshape(-1, N, D)  # [B, N, D]

    share_in_list, share_out_list, details = [], [], []
    r_in_list, c_in_list, v_in_list = [], [], []
    r_out_list, c_out_list, v_out_list = [], [], []

    for b in range(B):
        s_in, r_in, c_in, v_in = _share_core(x_flat[b])
        s_out, r_out, c_out, v_out = _share_core(z_flat[b], Z_ref=x_flat[b])

        share_in_list.append(s_in)
        share_out_list.append(s_out)

        r_in_list.append(r_in)
        c_in_list.append(c_in)
        v_in_list.append(v_in)

        r_out_list.append(r_out)
        c_out_list.append(c_out)
        v_out_list.append(v_out)

        details.append(
            {
                "eff_rank_in": r_in,
                "cos_div_in": c_in,
                "rel_var_in": v_in,
                "eff_rank_out": r_out,
                "cos_div_out": c_out,
                "rel_var_out": v_out,
            }
        )

    share_input = float(torch.tensor(share_in_list).mean())
    share_embed = float(torch.tensor(share_out_list).mean())
    delta = share_embed - share_input

    # --- Mean of component metrics across batches ---
    mean_details = {
        "eff_rank_in": float(torch.tensor(r_in_list).mean()),
        "cos_div_in": float(torch.tensor(c_in_list).mean()),
        "rel_var_in": float(torch.tensor(v_in_list).mean()),
        "eff_rank_out": float(torch.tensor(r_out_list).mean()),
        "cos_div_out": float(torch.tensor(c_out_list).mean()),
        "rel_var_out": float(torch.tensor(v_out_list).mean()),
    }

    return {
        "share_input": share_input,
        "share_embed": share_embed,
        "delta": delta,
        "details": details,
        "mean_details": mean_details,
    }
