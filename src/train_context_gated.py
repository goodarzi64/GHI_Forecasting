from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from src.GST_Utils import (
    compute_mi,
    corrcoef,
    gearys_C,
    morans_I,
    pairwise_edge_energy,
    dense_batch_to_block_sparse
)
from src.checkpoint_utils import load_checkpoint, save_checkpoint
from src.mixed_adj import build_mixed_adjacency

# -------------------------------------------------------------------------------
# Evaluation function for context-gated mixed adjacency models.
# -------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_context_gated(
    model,
    Embedor,
    Gater,
    WindKernel,
    loader,
    A_static,
    temperature=0.5,
    topk_each=12,
    ablation_config=None,
    device=None,
    compute_layer_metrics=False,
    node_const=None,
):
    # Evaluation mode: disable training-time behavior (dropout, BN updates, grads).
    model.eval()
    Embedor.eval()
    Gater.eval()
    WindKernel.eval()
    device = device or next(model.parameters()).device
    use_amp = (torch.device(device).type == "cuda")

    # Containers are initialized lazily because horizon H is inferred at runtime.
    horizon_true, horizon_pred, horizon_losses, layer_metrics = None, None, None, None
    A_static = A_static.to(device)

    for sample in tqdm(loader, desc="Evaluating", leave=True):
        # Unpack one mini-batch and move tensors to target device.
        x_batch, e_batch, w_batch, g_batch, y_batch = sample

        x_batch = x_batch.to(device)
        if e_batch is not None:
            try:
                e_batch = e_batch.to(device)
            except Exception:
                pass
        w_batch = w_batch.to(device)
        if g_batch is not None:
            g_batch = g_batch.to(device)
        y_batch = y_batch.to(device)

        B = x_batch.shape[0]

        # Build mixed adjacency per sample using static/dynamic/wind components.
        A_t, _ = build_mixed_adjacency(
            e_batch,
            w_batch,
            g_batch,
            node_const,
            Gater,
            WindKernel,
            A_static,
            Embedor,
            temperature=temperature,
            topk_each=topk_each,
            ablation_config=ablation_config,
        )

        if not compute_layer_metrics:
            # Fast path:
            #   run one forward on a disconnected block graph of size B*N.
            W, N, F_in = x_batch.shape[1], x_batch.shape[2], x_batch.shape[3]
            x_seq_batch = x_batch.permute(1, 0, 2, 3).reshape(W, B * N, F_in)
            ei, ew = dense_batch_to_block_sparse(A_t)
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat_bn = model(x_seq_batch, ei, ew, return_hidden=False)  # [B*N,H]
            y_hat_batch = y_hat_bn.view(B, N, -1).permute(0, 2, 1).contiguous()  # [B,H,N]
            H = y_hat_batch.shape[1]

            # Allocate per-horizon collectors once.
            if horizon_true is None:
                horizon_true = [[] for _ in range(H)]
                horizon_pred = [[] for _ in range(H)]
                horizon_losses = [[] for _ in range(H)]

            # Store predictions/targets horizon-wise for aggregated RMSE/MAE.
            for h in range(H):
                y_true_h = y_batch[:, h, :]  # [B,N]
                y_pred_h = y_hat_batch[:, h, :]  # [B,N]
                horizon_true[h].append(y_true_h.detach().cpu().numpy())
                horizon_pred[h].append(y_pred_h.detach().cpu().numpy())
                horizon_losses[h].append(torch.mean((y_pred_h - y_true_h) ** 2).item())
        else:
            # Detailed path:
            #   per-sample forward with hidden states for layer diagnostics.
            for b in range(B):
                x_seq = x_batch[b].to(device)
                y_seq = y_batch[b].to(device)
                ei, ew = dense_to_sparse(A_t[b])

                y_hat, hidden_states = model(x_seq, ei, ew, return_hidden=True)
                y_t = y_seq.transpose(0, 1).contiguous()
                H = y_t.shape[1]

                # Allocate prediction and layer-metric collectors once.
                if horizon_true is None:
                    horizon_true = [[] for _ in range(H)]
                    horizon_pred = [[] for _ in range(H)]
                    horizon_losses = [[] for _ in range(H)]
                    L = model.n_layers + 1
                    layer_metrics = {
                        name: [[] for _ in range(L)]
                        for name in ["Var", "Dir", "Moran", "Geary", "MI"]
                    }

                # Per-horizon prediction tracking.
                for h in range(H):
                    y_true_h, y_pred_h = y_t[:, h], y_hat[:, h]
                    horizon_true[h].append(y_true_h.detach().cpu().numpy())
                    horizon_pred[h].append(y_pred_h.detach().cpu().numpy())
                    horizon_losses[h].append(torch.mean((y_pred_h - y_true_h) ** 2).item())

                # Layer diagnostics: smoothness/dispersion/spatial autocorrelation and MI.
                features_list = [x_seq[-1]] + hidden_states # layer0 + every hidden layer
                for l, h_l in enumerate(features_list): # h_l shape = [N, D_l]
                    vals_var = torch.var(h_l, dim=0, unbiased=False) # variance of each feature across nodes.

                    dir_vals = torch.tensor(
                        [
                            pairwise_edge_energy(h_l[:, f], ei, ew).item()
                            for f in range(h_l.shape[1])
                        ]
                    )# Measures edge-wise roughness/smoothness of each feature across the graph.
                    mor_vals = torch.tensor(
                        [morans_I(h_l[:, f], ei, ew).item() for f in range(h_l.shape[1])]
                    )# Measures global spatial autocorrelation of each feature across the graph.
                    gea_vals = torch.tensor(
                        [gearys_C(h_l[:, f], ei, ew).item() for f in range(h_l.shape[1])]
                    )# Measures local spatial autocorrelation of each feature across the graph.

                    layer_metrics["Var"][l].append(float(vals_var.mean()))# average variance across features for layer l
                    layer_metrics["Dir"][l].append(float(dir_vals.mean()))# average pairwise edge energy across features for layer l
                    layer_metrics["Moran"][l].append(float(mor_vals.mean()))# average Moran's I across features for layer l
                    layer_metrics["Geary"][l].append(float(gea_vals.mean()))# average Geary's C across features for layer l 
                    layer_metrics["MI"][l].append(float(compute_mi(h_l, y_t, max_samples=5000)))# average mutual information between features and target across features for layer l

    # Aggregate metrics over all batches per forecast horizon.
    avg_horizon_metrics = []
    for h in range(len(horizon_true)):
        y_t_all = np.concatenate(horizon_true[h], axis=0)
        y_p_all = np.concatenate(horizon_pred[h], axis=0)
        mse = np.mean((y_t_all - y_p_all) ** 2)
        avg_horizon_metrics.append(
            {"RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(y_t_all - y_p_all))}
        )

    # Global score over all horizons and all samples.
    y_true_global = np.concatenate([np.concatenate(ht, axis=0) for ht in horizon_true], axis=0)
    y_pred_global = np.concatenate([np.concatenate(hp, axis=0) for hp in horizon_pred], axis=0)
    mse = np.mean((y_true_global - y_pred_global) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_global - y_pred_global))

    avg_layer_metrics = None
    if compute_layer_metrics:
        # Average each diagnostic across all evaluation samples.
        avg_layer_metrics = [
            {key: float(np.mean(layer_metrics[key][l])) for key in layer_metrics.keys()}
            for l in range(model.n_layers + 1)
        ]

    return {
        "Global": {"RMSE": rmse, "MAE": mae},
        "PerHorizon": avg_horizon_metrics,
        "PerLayer": avg_layer_metrics,
    }
# -------------------------------------------------------------------------------
# Main training loop with context-gated mixed adjacency and counterfactual regularization.
# -------------------------------------------------------------------------------
def train_joint_context_gated(
    model,
    Embedor,
    Gater,
    WindKernel,
    train_loader,
    device,
    A_static,
    test_loader=None,
    epochs=20,
    lr=1e-3,
    temperature=0.5,
    topk_each=5,
    checkpoint_path="checkpoint.pt",
    resume=False,
    ablation_config=None,
    layer_metrics=False,
    node_const=None,
    lambda_cf=0.05,
    lambda_align=0.01,
    huber_delta=20.0,
    cf_every_n_batches=4,
    layer_metrics_every_n_epochs=1,
):
    if ablation_config is None:
        ablation_config = {
            "use_static": True,
            "use_dynamic": True,
            "use_wind": True,
        }

    # ---------------- Parameters ----------------
    params = list(model.parameters())
    if any(ablation_config.values()):
        params += list(Gater.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.HuberLoss(delta=huber_delta)
    use_amp = (torch.device(device).type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    start_epoch = 1
    train_losses, test_metrics_history = [], []

    if resume and os.path.exists(checkpoint_path):
        start_epoch, train_losses, test_metrics_history = load_checkpoint(
            model, Embedor, Gater, optimizer, checkpoint_path, device
        )
        start_epoch += 1

    # ---------------- Freeze modules ----------------
    Embedor.eval()
    for p in Embedor.parameters():
        p.requires_grad_(False)

    try:
        WindKernel.eval()
    except Exception:
        WindKernel.train(False)

    A_static = A_static.to(device)

    # ================= Training =================
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        Gater.train()

        total_loss, n_batches = 0.0, 0

        for batch_idx, sample in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"), start=1
        ):
            x_batch, e_batch, w_batch, g_batch, y_batch = sample
            B = x_batch.size(0)

            x_batch = x_batch.to(device)  # [B, W, N, F_in]
            y_batch = y_batch.to(device)  # [B, H, N]
            w_batch = w_batch.to(device)
            if e_batch is not None:
                e_batch = e_batch.to(device)
            if g_batch is not None:
                g_batch = g_batch.to(device)

            # -------- Build adjacency --------
            # A_mix_batch: [B, N, N]
            A_mix_batch, aux = build_mixed_adjacency(
                e_batch,
                w_batch,
                g_batch,
                node_const,
                Gater,
                WindKernel,
                A_static,
                Embedor,
                temperature=temperature,
                topk_each=topk_each,
                ablation_config=ablation_config,
            )

            pi = aux["pi"]  # [B,N,G]
            ctx = aux["ctx_dict"]

            graph_names = ["static", "dynamic", "wind"]
            W, N, F_in = x_batch.shape[1], x_batch.shape[2], x_batch.shape[3]
            x_seq_batch = x_batch.permute(1, 0, 2, 3).reshape(W, B * N, F_in)
            y_target = y_batch.transpose(1, 2).contiguous()  # [B,N,H]

            # ---------------- Prediction loss ----------------
            ei, ew = dense_batch_to_block_sparse(A_mix_batch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat_bn = model(x_seq_batch, ei, ew, return_hidden=False)  # [B*N,H]
                y_hat_full = y_hat_bn.view(B, N, -1)  # [B,N,H]
                loss_pred = loss_fn(y_hat_full, y_target)
            y_hat_ref = y_hat_full.detach()

            # ---------------- Counterfactual loss ----------------
            L_cf = 0.0
            n_cf = 0

            do_cf = cf_every_n_batches > 0 and (batch_idx % cf_every_n_batches == 0)
            if do_cf:
                base_loss_b = nn.functional.huber_loss(
                    y_hat_ref, y_target, delta=huber_delta, reduction="none"
                ).mean(dim=(1, 2))
                for gi, gname in enumerate(graph_names):

                    if not ablation_config.get(f"use_{gname}", True):
                        continue

                    A_g = aux[f"A_{gname}"]
                    A_minus_batch = A_mix_batch - pi[:, :, gi : gi + 1] * A_g  # [B,N,N]
                    ei_m, ew_m = dense_batch_to_block_sparse(A_minus_batch)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        y_minus_bn = model(x_seq_batch, ei_m, ew_m, return_hidden=False)
                        y_minus = y_minus_bn.view(B, N, -1)  # [B,N,H]
                        delta_b = (
                            nn.functional.huber_loss(
                                y_minus, y_target, delta=huber_delta, reduction="none"
                            ).mean(dim=(1, 2))
                            - base_loss_b
                        )

                    L_cf += (pi[:, :, gi].mean(dim=1) * torch.relu(delta_b.detach())).mean()
                    n_cf += 1

            if n_cf > 0:
                L_cf /= n_cf

            # ---------------- Context-pi alignment ----------------
            L_align = 0.0
            n_align = 0

            if "Var_E_static" in ctx:
                v = ctx["Var_E_static"].squeeze(-1)
                L_align += corrcoef(pi[..., graph_names.index("static")], v, eps=1e-8)
                n_align += 1

            if "Var_E_dyn" in ctx:
                v = ctx["Var_E_dyn"].squeeze(-1)
                L_align -= corrcoef(pi[..., graph_names.index("dynamic")], v, eps=1e-8)
                n_align += 1

            if "Var_E_wind" in ctx:
                v = ctx["Var_E_wind"].squeeze(-1)
                L_align -= corrcoef(pi[..., graph_names.index("wind")], v, eps=1e-8)
                n_align += 1

            if n_align > 0:
                L_align /= n_align

            # ---------------- Total loss ----------------
            loss = loss_pred + lambda_cf * L_cf + lambda_align * L_align

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        train_losses.append(avg_train)

        # ---------------- Evaluation ----------------
        if test_loader is not None:
            do_layer_metrics = layer_metrics and (
                layer_metrics_every_n_epochs > 0
                and epoch % layer_metrics_every_n_epochs == 0
            )
            eval_results = evaluate_context_gated(
                model,
                Embedor,
                Gater,
                WindKernel,
                test_loader,
                A_static=A_static,
                temperature=temperature,
                topk_each=topk_each,
                ablation_config=ablation_config,
                device=device,
                compute_layer_metrics=do_layer_metrics,
                node_const=node_const,
            )
            test_metrics_history.append(eval_results)

            gm = eval_results["Global"]
            print(
                f"Epoch {epoch} | Train {avg_train:.4f} | "
                f"RMSE {gm['RMSE']:.4f}, MAE {gm['MAE']:.4f}"
            )

        save_checkpoint(
            model,
            Gater,
            optimizer,
            epoch,
            train_losses,
            test_metrics_history,
            checkpoint_path,
        )

    return {
        "train_losses": train_losses,
        "test_metrics_history": test_metrics_history,
    }
