"""
RAINWISE — TFT Training Script
================================
Trains the Temporal Fusion Transformer on Gujarat historical rainfall data.

Usage:
    cd /Users/HetviSheth/rainwise/model_trainingDL
    python train_tft.py

Outputs:
    DLmodels/tft_rainfall.pth              ← best model weights
    DLmodels/tft_scalers.pkl               ← static + temporal scalers
    outputs/dl/tft_loss_curves.png         ← train/val loss
    outputs/dl/tft_feature_importance.png  ← VSN static weights
    outputs/dl/tft_predictions.png         ← actual vs predicted scatter
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Local imports ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tft_model      import TemporalFusionTransformer, QuantileLoss
from tft_dataloader import (prepare_tft_data,
                             STATIC_COLS, TEMPORAL_COLS)


# ═══════════════════════════════════════════════════
#  CONFIGURATION  — tune these for your machine
# ═══════════════════════════════════════════════════
DATA_PATH        = "../data/processed/training_dataset_gujarat_advanced_labeled.csv"
MODEL_SAVE_PATH  = "../DLmodels/tft_rainfall.pth"
SCALER_SAVE_PATH = "../DLmodels/tft_scalers.pkl"
OUTPUT_DIR       = "../outputs/dl"

# Data
NROWS       = 250_000    # rows to load (250k ≈ fast on MPS)
SEQ_LEN     = 14         # 14-day look-back window
TEST_SIZE   = 0.15
BATCH_SIZE  = 512

# Model architecture
HIDDEN_DIM      = 64
NUM_HEADS       = 4
NUM_LSTM_LAYERS = 2
DROPOUT         = 0.15

# Training
EPOCHS   = 40
LR       = 3e-4
PATIENCE = 7          # early stopping patience
QUANTILES = (0.1, 0.5, 0.9)   # P10, P50, P90

# Device
DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)


# ═══════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmse(pred, true):
    return float(torch.sqrt(torch.mean((pred - true) ** 2)))

def mae(pred, true):
    return float(torch.mean(torch.abs(pred - true)))


# ═══════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════
def train():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "═"*60)
    print("  RAINWISE — Temporal Fusion Transformer Training")
    print("═"*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Sequence   : {SEQ_LEN} days look-back")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Hidden dim : {HIDDEN_DIM}  |  Heads: {NUM_HEADS}")
    print(f"  Epochs     : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Quantiles  : {QUANTILES}")
    print("═"*60)

    # ── 1. Load Data ──────────────────────────────────────────
    print("\n📂 Preparing TFT sequences...")
    (train_loader, val_loader,
     static_scaler, temporal_scaler,
     num_static, num_temporal) = prepare_tft_data(
        file_path   = DATA_PATH,
        seq_len     = SEQ_LEN,
        nrows       = NROWS,
        test_size   = TEST_SIZE,
        batch_size  = BATCH_SIZE,
        scaler_path = SCALER_SAVE_PATH,
    )
    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Static  vars  : {num_static}  → {STATIC_COLS}")
    print(f"  Temporal vars : {num_temporal} → {TEMPORAL_COLS}")

    # ── 2. Build Model ────────────────────────────────────────
    print("\n🧠 Building Temporal Fusion Transformer...")
    model = TemporalFusionTransformer(
        num_static_vars   = num_static,
        num_temporal_vars = num_temporal,
        hidden_dim        = HIDDEN_DIM,
        num_heads         = NUM_HEADS,
        num_lstm_layers   = NUM_LSTM_LAYERS,
        dropout           = DROPOUT,
        seq_len           = SEQ_LEN,
        num_quantiles     = len(QUANTILES),
    ).to(DEVICE)

    total_params = count_params(model)
    print(f"  Total trainable parameters: {total_params:,}")

    # ── 3. Loss + Optimiser ───────────────────────────────────
    criterion = QuantileLoss(quantiles=QUANTILES)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── 4. Training Loop ──────────────────────────────────────
    print(f"\n🚀 Training on {DEVICE}...\n")
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train": [], "val": [], "val_rmse": [], "val_mae": []}
    all_static_weights = []   # for VSN interpretability plot

    total_start = time.time()

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for static_b, temporal_b, target_b in train_loader:
            static_b   = static_b.to(DEVICE)
            temporal_b = temporal_b.to(DEVICE)
            target_b   = target_b.to(DEVICE)

            optimizer.zero_grad()
            preds, _   = model(static_b, temporal_b)   # (B, 3)
            loss        = criterion(preds, target_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss       = 0.0
        all_preds_p50  = []
        all_targets    = []
        epoch_sw       = []

        with torch.no_grad():
            for static_b, temporal_b, target_b in val_loader:
                static_b   = static_b.to(DEVICE)
                temporal_b = temporal_b.to(DEVICE)
                target_b   = target_b.to(DEVICE)

                preds, sw  = model(static_b, temporal_b)   # (B, 3), (B, 4)
                loss        = criterion(preds, target_b)
                val_loss   += loss.item()

                # P50 (median) for metric computation
                all_preds_p50.append(preds[:, 1].cpu())
                all_targets.append(target_b.cpu())
                epoch_sw.append(sw.mean(dim=0).cpu())   # average weights

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        p50    = torch.cat(all_preds_p50)
        tgts   = torch.cat(all_targets)
        val_rmse_val = rmse(p50, tgts)
        val_mae_val  = mae(p50, tgts)

        history["train"].append(avg_train)
        history["val"].append(avg_val)
        history["val_rmse"].append(val_rmse_val)
        history["val_mae"].append(val_mae_val)

        # Collect average static weights for this epoch
        avg_sw = torch.stack(epoch_sw).mean(dim=0).numpy()
        all_static_weights.append(avg_sw)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ──
        log_str = (
            f"  Epoch [{epoch+1:02d}/{EPOCHS}] "
            f"| Train: {avg_train:.4f} "
            f"| Val: {avg_val:.4f} "
            f"| RMSE(P50): {val_rmse_val:.3f} mm "
            f"| MAE: {val_mae_val:.3f} mm "
            f"| LR: {current_lr:.2e}"
        )
        print(log_str)

        # ── Early Stopping + Save Best ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "val_loss":    best_val_loss,
                "val_rmse":    val_rmse_val,
                "val_mae":     val_mae_val,
                "config": {
                    "num_static":   num_static,
                    "num_temporal": num_temporal,
                    "hidden_dim":   HIDDEN_DIM,
                    "num_heads":    NUM_HEADS,
                    "lstm_layers":  NUM_LSTM_LAYERS,
                    "seq_len":      SEQ_LEN,
                    "dropout":      DROPOUT,
                },
            }, MODEL_SAVE_PATH)
            print(f"    ✅ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  🛑 Early stop at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  ✅ Training complete in {total_time/60:.1f} min")
    print(f"  🏆 Best Val Loss  : {best_val_loss:.4f}")
    print(f"  📁 Model saved   → {MODEL_SAVE_PATH}")
    print(f"  📁 Scalers saved → {SCALER_SAVE_PATH}")
    print("═"*60)

    # ── 5. Visualisations ────────────────────────────────────
    _plot_loss_curves(history)
    _plot_feature_importance(all_static_weights)
    _plot_predictions(p50.numpy(), tgts.numpy())


# ═══════════════════════════════════════════════════
#  PLOT HELPERS
# ═══════════════════════════════════════════════════
def _plot_loss_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TFT — Training Diagnostics", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(history["train"], label="Train (Quantile Loss)", color="#3B82F6", linewidth=2)
    ax.plot(history["val"],   label="Val  (Quantile Loss)",  color="#EF4444", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pinball Loss (log)")
    ax.set_title("Loss Curves")
    ax.legend(); ax.grid(alpha=0.3)

    # RMSE / MAE
    ax = axes[1]
    ax.plot(history["val_rmse"], label="Val RMSE (mm)", color="#8B5CF6", linewidth=2)
    ax.plot(history["val_mae"],  label="Val MAE  (mm)", color="#10B981", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (mm/day)")
    ax.set_title("Validation RMSE & MAE (P50 prediction)")
    ax.legend(); ax.grid(alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "tft_loss_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📈 Loss curves saved → {path}")


def _plot_feature_importance(all_static_weights):
    """Plot average Variable Selection Network weights for static features."""
    weights_arr = np.array(all_static_weights)   # (epochs, num_static)
    avg_weights = weights_arr.mean(axis=0)        # (num_static,)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"]
    bars = ax.barh(STATIC_COLS, avg_weights, color=colors, edgecolor="white")
    ax.set_xlabel("Average VSN Selection Weight", fontsize=12)
    ax.set_title("TFT: Static Feature Importance\n(Variable Selection Network)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(avg_weights) * 1.3)

    for bar, w in zip(bars, avg_weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=11, color="#1F2937")

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tft_feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Feature importance → {path}")


def _plot_predictions(preds, targets):
    """Scatter: actual vs predicted rainfall (P50)."""
    # Clip to sensible range for Gujarat
    mask = targets <= 200
    p, t = preds[mask], targets[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TFT Predictions vs Actual Rainfall (P50 / Median)",
                 fontsize=13, fontweight="bold")

    # Scatter
    ax = axes[0]
    ax.scatter(t, p, alpha=0.15, s=5, color="#3B82F6")
    lim = max(t.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual rain_mm"); ax.set_ylabel("Predicted rain_mm (P50)")
    ax.set_title("Actual vs Predicted")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # Error histogram
    ax = axes[1]
    errors = p - t
    ax.hist(errors, bins=60, color="#8B5CF6", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("Prediction Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution\n"
                 f"Mean={errors.mean():.2f}  Std={errors.std():.2f} mm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tft_predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  🎯 Predictions plot → {path}")


# ═══════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    train()
