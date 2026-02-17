import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch_geometric_temporal.nn.recurrent import GConvGRU


class GraphGateNodewise(nn.Module):
    """
    Node-wise context-dependent gate over multiple adjacency types.
    Input:
        ctx: [N, in_dim] — context vector per node
    Output:
        pi: [N, n_graphs] — mixture weights for each adjacency type, per node
        Output graphs:  [static, dynamic, wind]
    """

    def __init__(self, in_dim, hidden=16, n_graphs=3, tau=1, learn_tau=False):
        super().__init__()
        self.n_graphs = n_graphs
        self.learn_tau = learn_tau

        # Learnable temperature if requested
        self.tau = nn.Parameter(torch.log(torch.tensor(tau))) if learn_tau else tau

        # MLP applied per node
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_graphs),
            nn.LayerNorm(n_graphs),  # optional: stabilize logits
        )

    def forward(self, ctx):
        """
        ctx: [N, in_dim] or [B, N, in_dim]
        Returns: [N, n_graphs] or [B, N, n_graphs]
        """
        if ctx.ndim == 2:
            # single graph: [N, in_dim] -> [N, n_graphs]
            tau = self.tau.exp() if self.learn_tau else self.tau
            logits = self.mlp(ctx) / tau
            pi = torch.softmax(logits, dim=-1)
            return pi

        if ctx.ndim == 3:
            # batched graph: [B, N, in_dim] -> [B, N, n_graphs]
            B, N, F = ctx.shape
            ctx_flat = ctx.view(B * N, F)
            tau = self.tau.exp() if self.learn_tau else self.tau
            logits_flat = self.mlp(ctx_flat) / tau
            pi = torch.softmax(logits_flat, dim=-1)
            return pi.view(B, N, self.n_graphs)

        raise ValueError(f"Unsupported ctx shape {ctx.shape}")


class RecurrentGCN(nn.Module):
    def __init__(
        self,
        node_feature_dim,  # F_in  : input feature dimension per node
        filters,  # [D1, D2, ..., DL] hidden dims per layer
        horizon=1,  # H     : number of forecasting steps
        alpha=0.5,
        dropout=0.2,
        mode="last",  # "last" or "fusion_learnable"
    ):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.horizon = horizon

        if isinstance(filters, int):
            filters = [filters]
        self.n_layers = len(filters)  # L layers

        self.recurrent_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_dim = node_feature_dim  # F_in
        for out_dim in filters:  # Dₗ
            self.recurrent_layers.append(GConvGRU(in_dim, out_dim, K=1))  # [N, Dₗ]
            self.residual_projs.append(nn.Linear(in_dim, out_dim))  # [N, Dₗ]
            in_dim = out_dim

        # One prediction head per horizon: maps [N, D_L] → [N, 1]
        self.heads = nn.ModuleList([nn.Linear(filters[-1], 1) for _ in range(horizon)])

        if self.mode == "fusion_learnable":
            self.fusion_weights = nn.Parameter(
                torch.ones(horizon, self.n_layers) / self.n_layers
            )

            # Projection: [N, Dₗ] → [N, D_L]
            self.projections = nn.ModuleList(
                [nn.Linear(d, filters[-1]) if d != filters[-1] else nn.Identity() for d in filters]
            )

    # -----------------------------------------------------------------
    # x_seq:          [W, N, F_in]
    # edge_index:     [2, E]
    # edge_weight:    [E]
    # return_hidden:  returns list of layer hidden states [N, Dₗ]
    # -----------------------------------------------------------------
    def forward(self, x_seq, edge_index, edge_weight, return_hidden=False):
        # Hidden state list: one tensor per layer → [N, Dₗ]
        hidden_states = [None] * self.n_layers

        # ----- Recurrence over window length W -----
        for w in range(x_seq.shape[0]):  # w = 0..W-1
            out = x_seq[w]  # [N, F_in] at w=0, then [N, Dₗ]

            for layer, (gru, res_proj) in enumerate(
                zip(self.recurrent_layers, self.residual_projs)
            ):
                h_prev = hidden_states[layer]  # [N, Dₗ] or None
                h_w = gru(out, edge_index, edge_weight, h_prev)  # [N, Dₗ]
                residual = res_proj(out)  # [N, Dₗ]

                h_w = (1 - self.alpha) * h_w + self.alpha * residual  # [N, Dₗ]
                h_w = Func.relu(h_w)  # [N, Dₗ]
                # h_w = norm(h_w)  # (optional)
                h_w = self.dropout(h_w)  # [N, Dₗ]

                hidden_states[layer] = h_w
                out = h_w  # feed to next GCN layer

        # ----- Horizon Predictions -----
        if self.mode == "last":
            final_hidden = hidden_states[-1]  # [N, D_L]
            outs = [head(final_hidden) for head in self.heads]  # H × [N, 1]

        elif self.mode == "fusion_learnable":
            weights = torch.softmax(self.fusion_weights, dim=-1)  # [H, L]

            outs = []
            for h, head in enumerate(self.heads):  # horizon h
                # blended: [N, D_L]
                blended = sum(
                    weights[h, l] * self.projections[l](hidden_states[l])
                    for l in range(self.n_layers)
                )
                outs.append(head(blended))  # [N, 1]

        pred = torch.cat(outs, dim=1)  # → [N, H]

        if return_hidden:
            return pred, hidden_states
        return pred
