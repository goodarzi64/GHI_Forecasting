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
)
from src.checkpoint_utils import load_checkpoint, save_checkpoint
from src.mixed_adj import build_mixed_adjacency


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
    model.eval()
    Embedor.eval()
    Gater.eval()
    WindKernel.eval()
    device = device or next(model.parameters()).device

    horizon_true, horizon_pred, horizon_losses, layer_metrics = None, None, None, None
    A_static = A_static.to(device)

    for sample in tqdm(loader, desc="Evaluating", leave=True):
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

        for b in range(B):
            x_seq = x_batch[b].to(device)
            y_seq = y_batch[b].to(device)
            ei, ew = dense_to_sparse(A_t[b])

            y_hat, hidden_states = model(x_seq, ei, ew, return_hidden=True)
            y_t = y_seq.transpose(0, 1).contiguous()
            H = y_t.shape[1]

            if horizon_true is None:
                horizon_true = [[] for _ in range(H)]
                horizon_pred = [[] for _ in range(H)]
                horizon_losses = [[] for _ in range(H)]
                if compute_layer_metrics:
                    L = model.n_layers + 1
                    layer_metrics = {
                        name: [[] for _ in range(L)]
                        for name in ["Var", "Dir", "Moran", "Geary", "MI"]
                    }

            for h in range(H):
                y_true_h, y_pred_h = y_t[:, h], y_hat[:, h]
                horizon_true[h].append(y_true_h.detach().cpu().numpy())
                horizon_pred[h].append(y_pred_h.detach().cpu().numpy())
                horizon_losses[h].append(torch.mean((y_pred_h - y_true_h) ** 2).item())

            if compute_layer_metrics:
                features_list = [x_seq[-1]] + hidden_states
                for l, h_l in enumerate(features_list):
                    vals_var = torch.var(h_l, dim=0, unbiased=False)

                    dir_vals = torch.tensor(
                        [
                            pairwise_edge_energy(h_l[:, f], ei, ew).item()
                            for f in range(h_l.shape[1])
                        ]
                    )
                    mor_vals = torch.tensor(
                        [morans_I(h_l[:, f], ei, ew).item() for f in range(h_l.shape[1])]
                    )
                    gea_vals = torch.tensor(
                        [gearys_C(h_l[:, f], ei, ew).item() for f in range(h_l.shape[1])]
                    )

                    layer_metrics["Var"][l].append(float(vals_var.mean()))
                    layer_metrics["Dir"][l].append(float(dir_vals.mean()))
                    layer_metrics["Moran"][l].append(float(mor_vals.mean()))
                    layer_metrics["Geary"][l].append(float(gea_vals.mean()))
                    layer_metrics["MI"][l].append(float(compute_mi(h_l, y_t)))

    avg_horizon_metrics = []
    for h in range(len(horizon_true)):
        y_t_all = np.concatenate(horizon_true[h], axis=0)
        y_p_all = np.concatenate(horizon_pred[h], axis=0)
        mse = np.mean((y_t_all - y_p_all) ** 2)
        avg_horizon_metrics.append(
            {"RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(y_t_all - y_p_all))}
        )

    y_true_global = np.concatenate([np.concatenate(ht, axis=0) for ht in horizon_true], axis=0)
    y_pred_global = np.concatenate([np.concatenate(hp, axis=0) for hp in horizon_pred], axis=0)
    mse = np.mean((y_true_global - y_pred_global) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_global - y_pred_global))

    avg_layer_metrics = None
    if compute_layer_metrics:
        avg_layer_metrics = [
            {key: float(np.mean(layer_metrics[key][l])) for key in layer_metrics.keys()}
            for l in range(model.n_layers + 1)
        ]

    return {
        "Global": {"RMSE": rmse, "MAE": mae},
        "PerHorizon": avg_horizon_metrics,
        "PerLayer": avg_layer_metrics,
    }


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

        for sample in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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

            # ---------------- Prediction loss ----------------
            loss_pred = 0.0
            y_hat_full = []

            for b in range(B):
                ei, ew = dense_to_sparse(A_mix_batch[b])
                # y_hat: [N, H]
                # x_batch[b]: [W, N, F_in]
                y_hat, _ = model(x_batch[b], ei, ew, return_hidden=True)

                loss_b = loss_fn(y_hat, y_batch[b].transpose(0, 1).contiguous())
                loss_pred += loss_b
                y_hat_full.append(y_hat.detach())

            loss_pred /= B

            # ---------------- Counterfactual loss ----------------
            L_cf = 0.0
            n_cf = 0

            for gi, gname in enumerate(graph_names):

                if not ablation_config.get(f"use_{gname}", True):
                    continue

                A_g = aux[f"A_{gname}"]

                for b in range(B):
                    A_minus = A_mix_batch[b] - pi[b, :, gi : gi + 1] * A_g[b]
                    ei_m, ew_m = dense_to_sparse(A_minus)

                    y_minus, _ = model(x_batch[b], ei_m, ew_m, return_hidden=True)

                    delta = loss_fn(y_minus, y_batch[b].transpose(0, 1)) - loss_fn(
                        y_hat_full[b], y_batch[b].transpose(0, 1)
                    )

                    L_cf += (pi[b, :, gi] * torch.relu(delta.detach())).mean()
                    n_cf += 1

            if n_cf > 0:
                L_cf /= n_cf

            # ---------------- Context-pi alignment ----------------
            L_align = 0.0
            n_align = 0

            if "Var_E_static" in ctx:
                v = ctx["Var_E_static"].squeeze(-1)
                L_align += corrcoef(pi[..., graph_names.index("static")], v)
                n_align += 1

            if "Var_E_dyn" in ctx:
                v = ctx["Var_E_dyn"].squeeze(-1)
                L_align -= corrcoef(pi[..., graph_names.index("dynamic")], v)
                n_align += 1

            if "Var_E_wind" in ctx:
                v = ctx["Var_E_wind"].squeeze(-1)
                L_align -= corrcoef(pi[..., graph_names.index("wind")], v)
                n_align += 1

            if n_align > 0:
                L_align /= n_align

            # ---------------- Total loss ----------------
            loss = loss_pred + lambda_cf * L_cf + lambda_align * L_align

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        train_losses.append(avg_train)

        # ---------------- Evaluation ----------------
        if test_loader is not None:
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
                compute_layer_metrics=layer_metrics,
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
