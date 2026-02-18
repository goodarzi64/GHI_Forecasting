from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# ========== Temporal Window AutoEncoder with Optional Attention
# ============================================================
class TemporalWindowAutoEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        window: int = 6,
        conv_hidden: int = 64,
        dropout: float = 0.1,
        use_attention: bool = False,
        cloud_embed_dim: int = 3,
        mask_cloud: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            in_dim: number of input features per node
            embed_dim: latent embedding dimension for nodes
            cloud_embed_dim: dimension of learnable cloud embedding
            mask_cloud: boolean mask for cloud features in input vector
        """
        super().__init__()
        self.in_dim = in_dim
        self.window = window
        self.use_attention = use_attention
        self.cloud_embed_dim = cloud_embed_dim
        self.cloud_norm = nn.LayerNorm(cloud_embed_dim)

        if mask_cloud is None:
            raise ValueError("mask_cloud must be provided to TemporalWindowAutoEncoder")

        # Register mask for cloud features (accept torch or numpy)
        mask_cloud_t = torch.as_tensor(mask_cloud).to(torch.bool)
        self.register_buffer("mask_cloud", mask_cloud_t.detach().clone())

        # Count number of explicit cloud classes (e.g., 9)
        self.num_cloud_classes = int(self.mask_cloud.sum().item())

        # Cloud embedding replaces the one-hot block
        self.in_dim_after_embedding = self.in_dim - self.num_cloud_classes + self.cloud_embed_dim

        # ------------------------------------------------
        # Cloud embedding (for one-hot -> latent projection)
        # ------------------------------------------------
        self.use_cloud_embed = self.num_cloud_classes > 0
        if self.use_cloud_embed:
            self.cloud_emb = nn.Embedding(self.num_cloud_classes, cloud_embed_dim)

        # ---------------- Encoder ----------------
        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_dim_after_embedding, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_hidden, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ---------------- Attention Pooling ----------------
        if use_attention:
            self.attn_fc = nn.Linear(embed_dim, 1)

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, conv_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_hidden, self.in_dim_after_embedding),
        )

    # ============================================================
    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, W, N, F_in = x_t.shape

        # --- optionally embed cloud features ---
        if self.use_cloud_embed:
            x_t = self.apply_cloud_embedding(x_t)

        z_t = self.encode_window(x_t)
        x_target = x_t[:, -1, :, :]
        x_recon = self.decode_nodes(z_t)
        recon_loss = F.mse_loss(x_recon, x_target)
        return recon_loss, z_t

    # ============================================================
    def apply_cloud_embedding(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Replace one-hot cloud features with learnable latent embedding.
        Input shape: [B, W, N, F_in]
        Output: concatenated features with cloud embedding normalized
        """
        mask = self.mask_cloud.to(x_t.device)
        mask = mask[: x_t.size(-1)]  # safety for shape mismatch
        num_cloud_classes = int(mask.sum().item())

        if num_cloud_classes == 0:
            return x_t

        # --- extract one-hot cloud features ---
        cloud_feats = x_t[..., mask]  # [B, W, N, num_cloud_classes]
        cloud_idx = cloud_feats.argmax(dim=-1).long().to(x_t.device)  # [B, W, N]

        # --- get embedding ---
        cloud_emb = self.cloud_emb(cloud_idx).to(x_t.device)  # [B, W, N, cloud_embed_dim]

        # --- normalize embedding to [0,1] like other features ---
        cloud_emb = self.cloud_norm(cloud_emb)

        # --- concatenate with non-cloud features ---
        non_cloud_feats = x_t[..., ~mask].to(x_t.device)
        x_t_new = torch.cat([non_cloud_feats, cloud_emb], dim=-1)

        return x_t_new

    # ============================================================
    def encode_window(self, x: torch.Tensor) -> torch.Tensor:
        B, W, N, F_in = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, F_in, W)
        z_seq = self.encoder(x).permute(0, 2, 1)
        if self.use_attention:
            attn_scores = self.attn_fc(z_seq)
            attn_weights = torch.softmax(attn_scores, dim=1)
            z = torch.sum(attn_weights * z_seq, dim=1)
        else:
            z = z_seq.mean(dim=1)
        z = z.view(B, N, -1)
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        B, N, D = z.shape
        x_hat = self.decoder(z.reshape(B * N, D))
        return x_hat.view(B, N, -1)


# ============================================================
# ============= Sliding Window Dataset =======================
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, data: torch.Tensor, window: int) -> None:
        self.data = data
        self.window = window

    def __len__(self) -> int:
        return self.data.shape[0] - self.window

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.window]


# ============================================================
# ================ Pretraining Function ======================
# ============================================================
def pretrain_en_de_with_regularizers(
    train_tensor: torch.Tensor,
    val_tensor: Optional[torch.Tensor] = None,
    in_dim: Optional[int] = None,
    embed_dim: int = 16,
    conv_hidden: int = 128,
    window: int = 6,
    use_attention: bool = True,
    batch_size: int = 8,
    lr: float = 1e-3,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience: int = 5,
    mask_embed: Optional[torch.Tensor] = None,
    mask_cloud: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[TemporalWindowAutoEncoder, Dict[str, List[float]], int]:
    """
    Simplified temporal autoencoder pretraining:
        Loss = recon_loss
    All wind / spatial regularizers are removed.
    """
    if in_dim is None:
        in_dim = train_tensor.shape[-1]
    if mask_cloud is None:
        raise ValueError("mask_cloud must be provided to pretrain_en_de_with_regularizers")

    # -------------------- Data --------------------
    train_loader = DataLoader(
        SlidingWindowDataset(train_tensor, window),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    val_loader = None
    if val_tensor is not None:
        val_loader = DataLoader(
            SlidingWindowDataset(val_tensor, window),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

    # -------------------- Model --------------------
    model = TemporalWindowAutoEncoder(
        in_dim=in_dim,
        embed_dim=embed_dim,
        window=window,
        conv_hidden=conv_hidden,
        dropout=0.1,
        use_attention=use_attention,
        cloud_embed_dim=3,
        mask_cloud=mask_cloud,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # -------------------- Logs --------------------
    logs: Dict[str, List[float]] = {k: [] for k in ["train_total", "train_recon", "val_total", "val_recon"]}

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    # ============================================================
    # ========================= Training =========================
    # ============================================================
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_recon = 0.0
        n_batches = 0

        for x_t in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            x_t = x_t.to(device=device, dtype=torch.float32)
            emb_t = x_t[..., mask_embed] if mask_embed is not None else x_t

            optimizer.zero_grad()
            recon_loss, _z_t = model(emb_t)
            loss = recon_loss

            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        epoch_recon /= n_batches

        train_total = epoch_recon
        logs["train_total"].append(train_total)
        logs["train_recon"].append(epoch_recon)

        # ============================================================
        # ====================== Validation ==========================
        # ============================================================
        val_total = None
        if val_loader is not None:
            model.eval()
            val_recon = 0.0
            n_val = 0

            with torch.no_grad():
                for x_t in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                    x_t = x_t.to(device=device, dtype=torch.float32)
                    emb_t = x_t[..., mask_embed] if mask_embed is not None else x_t

                    recon_loss, _z_t = model(emb_t)
                    val_recon += recon_loss.item()
                    n_val += 1

            n_val = max(n_val, 1)
            val_recon /= n_val
            val_total = val_recon

            logs["val_total"].append(val_total)
            logs["val_recon"].append(val_recon)

            scheduler.step(val_total)

        # ============================================================
        # ===================== Early Stopping =======================
        # ============================================================
        improved = val_total is not None and val_total < best_val

        if improved:
            best_val = val_total
            best_epoch = epoch
            patience_counter = 0

            if save_path is not None:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "logs": logs,
                        "last_epoch": epoch,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                    },
                    save_path,
                )
        else:
            patience_counter += 1

        if verbose:
            v = f"{val_total:.4f}" if val_total is not None else "N/A"
            print(f"Epoch {epoch:02d} | Train: {train_total:.4f} | Val: {v} | Patience {patience_counter}")

        if val_total is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best={best_val:.4f} @ {best_epoch}")
            break

    # ----------------------- Save config -------------------------
    if save_path is not None:
        model_config = {
            "in_dim": in_dim,
            "embed_dim": embed_dim,
            "window": window,
            "use_attention": use_attention,
            "conv_hidden": (
                model.encoder[0].out_channels
                if hasattr(model.encoder[0], "out_channels")
                else conv_hidden
            ),
            "dropout": getattr(model.encoder[3], "p", 0.1),
        }

        checkpoint = torch.load(save_path)
        checkpoint["config"] = model_config
        torch.save(checkpoint, save_path)

        print(f"Training complete. Best model saved at {save_path} (epoch {best_epoch})")
    else:
        print(f"Training complete. Best model at epoch {best_epoch} (no file saved).")

    return model, logs, best_epoch


def load_pretrained_embedor(
    ckpt_path: str,
    mask_cloud,
    device: str = "cuda",
) -> TemporalWindowAutoEncoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = TemporalWindowAutoEncoder(
        in_dim=cfg["in_dim"],
        embed_dim=cfg["embed_dim"],
        window=cfg["window"],
        conv_hidden=cfg["conv_hidden"],
        dropout=cfg["dropout"],
        use_attention=cfg["use_attention"],
        cloud_embed_dim=cfg.get("cloud_embed_dim", 3),
        mask_cloud=mask_cloud,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded Embedor from {ckpt_path}")
    return model
