import torch
from src.GST_Utils import minmax_normalize, topk_row

_STATIC_EXPAND_CACHE = {}


def _get_cached_static_batch(A_static: torch.Tensor, B: int, N: int, dev: torch.device):
    """
    Cache expanded static adjacency [B,N,N] when batch/node size is stable.
    Cache key includes tensor identity/version and shape/device to avoid stale reuse.
    """
    key = (
        int(A_static.data_ptr()),
        int(getattr(A_static, "_version", 0)),
        tuple(A_static.shape),
        B,
        N,
        dev.type,
        dev.index,
        A_static.dtype,
    )
    cached = _STATIC_EXPAND_CACHE.get(key)
    if cached is not None:
        return cached

    A_dev = A_static.to(dev)
    A_s = A_dev.unsqueeze(0).expand(B, -1, -1)

    # Keep cache bounded: static graph setup usually uses one/few keys.
    if len(_STATIC_EXPAND_CACHE) > 8:
        _STATIC_EXPAND_CACHE.clear()
    _STATIC_EXPAND_CACHE[key] = A_s
    return A_s


def build_mixed_adjacency(
    e_window,  # [B, W, N, Fe]
    w_window,  # [B, W, N, Fw]
    g_window,  # [B, W, N, Fg] or None
    node_const,  # [N, Fc] or None
    Gater,
    WindKernel,
    A_static,  # [N,N]
    Embedor,  # frozen
    temperature=0.5,
    topk_each=5,
    ablation_config=None,
):
    """
    Option B:
    - ctx_tensor: concatenated tensor → Gater
    - ctx_dict  : named components → alignment & analysis
    """

    if ablation_config is None:
        ablation_config = dict(
            use_static=True,
            use_dynamic=True,
            use_wind=True,
        )

    dev = e_window.device
    dtype = e_window.dtype
    B, W, N, _ = e_window.shape

    # ============================================================
    # 1) CLOUD EMBEDDINGS (NO GRAD)
    # ============================================================
    with torch.no_grad():
        e_proc = Embedor.apply_cloud_embedding(e_window).to(dev)
        g_proc = (
            Embedor.apply_cloud_embedding(g_window).to(dev)
            if g_window is not None
            else None
        )

    g_last = g_proc[:, -1] if g_proc is not None else None

    # ============================================================
    # 2) NODE CONSTANTS
    # ============================================================
    if node_const is None:
        node_const_tensor = torch.zeros(B, N, 0, device=dev)
    else:
        node_const_tensor = node_const.unsqueeze(0).expand(B, N, -1).to(dev)

    # ============================================================
    # 3) DYNAMIC ADJACENCY (NO GRAD)
    # ============================================================
    if ablation_config["use_dynamic"]:
        with torch.no_grad():
            Z_last = Embedor.encode_window(e_proc)  # [B,N,D]

        S = Z_last @ Z_last.transpose(-1, -2)  # [B,N,N]
        A_dyn = torch.softmax(S / temperature, dim=-1)
        A_dyn.diagonal(dim1=-2, dim2=-1).zero_()
        A_dyn = topk_row(A_dyn, topk_each)
    else:
        A_dyn = torch.zeros(B, N, N, device=dev)

    # ============================================================
    # 4) STATIC ADJACENCY
    # ============================================================
    if ablation_config["use_static"]:
        A_s = _get_cached_static_batch(A_static, B, N, dev)
    else:
        A_s = torch.zeros(B, N, N, device=dev)

    # ============================================================
    # 5) ENERGY CORE (ALL LAGS, VECTORIZED)
    # Reuse pairwise squared distances once for all graph types.
    # ============================================================
    X2 = (e_proc * e_proc).sum(dim=-1, keepdim=True)  # [B,W,N,1]
    XY = e_proc @ e_proc.transpose(-1, -2)  # [B,W,N,N]
    D2 = X2 + X2.transpose(-1, -2) - 2 * XY  # [B,W,N,N]

    E_static_window = (A_s.unsqueeze(1) * D2).sum(dim=-1)  # [B,W,N]
    E_dyn_window = (A_dyn.unsqueeze(1) * D2).sum(dim=-1)  # [B,W,N]

    # ============================================================
    # 6) WIND ADJACENCY + ENERGY (STREAMED)
    # Avoid materializing full [B,W,N,N] wind tensor.
    # ============================================================
    if ablation_config["use_wind"]:
        E_wind_list = []
        A_wind = None
        for t in range(W):
            Aw_t = WindKernel(w_window[:, t], sparse=True, k=topk_each).to(
                device=dev, dtype=dtype
            )  # [B,N,N]
            E_wind_list.append((Aw_t * D2[:, t]).sum(dim=-1))  # [B,N]
            A_wind = Aw_t  # keep last lag adjacency

        E_wind_window = torch.stack(E_wind_list, dim=1)  # [B,W,N]
    else:
        E_wind_window = torch.zeros(B, W, N, device=dev, dtype=dtype)
        A_wind = torch.zeros(B, N, N, device=dev, dtype=dtype)

    # ============================================================
    # 7) CONTEXT FEATURES (NAMED)
    # ============================================================
    ctx_dict = {}

    ctx_dict["E_static_last"] = minmax_normalize(E_static_window[:, -1].unsqueeze(-1))
    ctx_dict["E_dyn_last"] = minmax_normalize(E_dyn_window[:, -1].unsqueeze(-1))
    ctx_dict["E_wind_last"] = minmax_normalize(E_wind_window[:, -1].unsqueeze(-1))

    ctx_dict["Var_E_static"] = minmax_normalize(
        E_static_window.var(dim=1, unbiased=False).unsqueeze(-1)
    )
    ctx_dict["Var_E_dyn"] = minmax_normalize(
        E_dyn_window.var(dim=1, unbiased=False).unsqueeze(-1)
    )
    ctx_dict["Var_E_wind"] = minmax_normalize(
        E_wind_window.var(dim=1, unbiased=False).unsqueeze(-1)
    )

    ctx_dict["Deg_static"] = minmax_normalize(A_s.sum(dim=-1, keepdim=True))
    ctx_dict["Deg_dyn"] = minmax_normalize(A_dyn.sum(dim=-1, keepdim=True))
    ctx_dict["Deg_wind"] = minmax_normalize(A_wind.sum(dim=-1, keepdim=True))

    # ---- non-cloud g variance ----
    if g_proc is None or g_proc.shape[-1] <= 3:
        ctx_dict["Var_g"] = torch.zeros(B, N, 1, device=dev)
    else:
        g_noncloud = g_proc[..., :-3]
        ctx_dict["Var_g"] = minmax_normalize(g_noncloud.var(dim=1, unbiased=False))

    # ============================================================
    # 8) CONCATENATED CONTEXT TENSOR
    # ============================================================
    ctx_tensor = torch.cat(
        [
            node_const_tensor,
            g_last if g_last is not None else torch.zeros(B, N, 0, device=dev),
            ctx_dict["E_static_last"],
            ctx_dict["E_dyn_last"],
            ctx_dict["E_wind_last"],
            ctx_dict["Var_E_static"],
            ctx_dict["Var_E_dyn"],
            ctx_dict["Var_E_wind"],
            ctx_dict["Deg_static"],
            ctx_dict["Deg_dyn"],
            ctx_dict["Deg_wind"],
            ctx_dict["Var_g"],
        ],
        dim=-1,
    )

    # ============================================================
    # 9) GATER
    # ============================================================
    pi = Gater(ctx_tensor)  # [B,N,3]

    mask = torch.tensor(
        [
            ablation_config["use_static"],
            ablation_config["use_dynamic"],
            ablation_config["use_wind"],
        ],
        device=dev,
        dtype=pi.dtype,
    )

    pi = pi * mask
    pi = pi / (pi.sum(-1, keepdim=True) + 1e-12)

    # ============================================================
    # 10) MIX ADJACENCY
    # ============================================================
    A_mix = pi[..., 0:1] * A_s + pi[..., 1:2] * A_dyn + pi[..., 2:3] * A_wind

    A_mix = topk_row(A_mix, topk_each)

    return A_mix, {
        "pi": pi,
        "ctx": ctx_tensor,
        "ctx_dict": ctx_dict,
        "A_static": A_s,
        "A_dynamic": A_dyn,
        "A_wind": A_wind,
    }
