"""
TEMPO Fine-Tuning for Carbon Flux Forecasting
================================================

Fine-tunes the pretrained TEMPO-80M model on univariate NEE data from
5 training EC tower sites, then evaluates on held-out UK-AMo and SE-Htm.

TEMPO is a univariate time series foundation model, so we extract the NEE
signal from raw site CSVs and create sliding windows (336 -> 96 timesteps).

Usage:
    python scripts/fine_tune_tempo.py
"""

import sys
import json
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "tempo_fine_tuned"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Add TEMPO repository to path
TEMPO_DIR = PROJECT_ROOT.parent / "TEMPO"
sys.path.insert(0, str(TEMPO_DIR))

from tempo.models.TEMPO import TEMPO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK = 336      # 2 weeks hourly
HORIZON = 96        # 4-day forecast
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 1.0
PATIENCE = 10
SEED = 42

TRAIN_SITES = ["FI-Lom", "GL-ZaF", "IE-Cra", "DE-Akm", "FR-LGt"]
TEST_SITES = ["UK-AMo", "SE-Htm"]

SITE_FILES = {
    "FI-Lom": "1.FI-Lom.csv",
    "GL-ZaF": "2.GL-ZaF.csv",
    "IE-Cra": "3.IE-Cra.xlsx",
    "DE-Akm": "4.DE-Akm.csv",
    "FR-LGt": "5.FR-LGt.csv",
    "UK-AMo": "6.UK-AMo.csv",
    "SE-Htm": "7.SE-Htm.csv",
}

# Zero-shot baselines (from run_zero_shot_tempo.py)
ZERO_SHOT_METRICS = {
    "UK-AMo": {"RMSE": 2.8739, "MAE": 1.7535, "R2": 0.3836},
    "SE-Htm": {"RMSE": 3.6224, "MAE": 2.4263, "R2": 0.7521},
}

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    """TEMPO uses GPT-2 token embeddings incompatible with MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading — univariate NEE from raw site files
# ---------------------------------------------------------------------------
def load_nee_series(site_name):
    """Load raw NEE time series for a site, gap-filled with ffill/bfill."""
    filepath = RAW_DIR / SITE_FILES[site_name]
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    nee = df["NEE_VUT_REF"].ffill().bfill().values.astype(np.float32)
    return nee


def create_sequences(series):
    """Create sliding-window sequences from a 1-D array."""
    X, y = [], []
    for i in range(len(series) - LOOKBACK - HORIZON + 1):
        X.append(series[i : i + LOOKBACK])
        y.append(series[i + LOOKBACK : i + LOOKBACK + HORIZON])
    return np.array(X), np.array(y)


def load_all_data():
    """Load univariate NEE sequences for training and test sites."""
    print("Loading training sites...")
    train_X_list, train_y_list = [], []
    for site in TRAIN_SITES:
        nee = load_nee_series(site)
        X, y = create_sequences(nee)
        train_X_list.append(X)
        train_y_list.append(y)
        print(f"  {site}: {len(X)} sequences ({len(nee)} timesteps)")

    train_X = np.concatenate(train_X_list)
    train_y = np.concatenate(train_y_list)

    print("\nLoading test sites...")
    test_data = {}
    for site in TEST_SITES:
        nee = load_nee_series(site)
        X, y = create_sequences(nee)
        test_data[site] = {"X": X, "y": y}
        print(f"  {site}: {len(X)} sequences ({len(nee)} timesteps)")

    return train_X, train_y, test_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tempo_model(device):
    """Load pretrained TEMPO-80M from HuggingFace."""
    cache_dir = str(PROJECT_ROOT / "models" / "checkpoints" / "tempo_zero_shot")
    model = TEMPO.load_pretrained_model(
        device=device,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir=cache_dir,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R² between arrays of shape (N, horizon)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------
def predict_batched(model, X, device, batch_size=BATCH_SIZE):
    """Run batched inference. Input X shape: (N, 336)."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(-1))  # [N, 336, 1]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            outputs, _ = model(bx, itr=0, trend=bx, season=bx, noise=bx)
            preds = outputs[:, -HORIZON:, :].cpu().numpy().squeeze(-1)
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def fine_tune(model, train_X, train_y, device):
    """
    Fine-tune TEMPO on carbon flux NEE training data.

    Uses temporal 80/20 split for train/val, AdamW optimizer,
    gradient clipping, and early stopping.
    """
    # Temporal train/val split (preserves time ordering)
    split_idx = int(len(train_X) * 0.8)
    X_trn, X_val = train_X[:split_idx], train_X[split_idx:]
    y_trn, y_val = train_y[:split_idx], train_y[split_idx:]
    print(f"  Train: {len(X_trn)} | Val: {len(X_val)} sequences")

    # Prepare tensors — TEMPO expects [B, L, 1]
    train_dataset = TensorDataset(
        torch.FloatTensor(X_trn).unsqueeze(-1),
        torch.FloatTensor(y_trn).unsqueeze(-1),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(-1),
        torch.FloatTensor(y_val).unsqueeze(-1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    train_losses = []
    val_losses = []
    stopped_epoch = EPOCHS  # will be updated if early stopping triggers

    model.train()
    t0 = time.time()

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        epoch_train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            outputs, loss_local = model(
                bx, itr=0, trend=bx, season=bx, noise=bx
            )
            outputs = outputs[:, -HORIZON:, :]
            by = by[:, -HORIZON:, :]
            loss = criterion(outputs, by)
            if loss_local is not None:
                loss = loss + 0.01 * loss_local

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train = np.mean(epoch_train_losses)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs, _ = model(bx, itr=0, trend=bx, season=bx, noise=bx)
                outputs = outputs[:, -HORIZON:, :]
                by = by[:, -HORIZON:, :]
                epoch_val_losses.append(criterion(outputs, by).item())

        avg_val = np.mean(epoch_val_losses)
        val_losses.append(avg_val)

        # Track best
        marker = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
            marker = " *"

            # Save best checkpoint
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "train_loss": avg_train,
                },
                CHECKPOINT_DIR / "best_model.pth",
            )
        else:
            wait += 1

        print(
            f"  Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Best: {best_val_loss:.4f}{marker}"
        )

        # Early stopping
        if wait >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} epochs)")
            stopped_epoch = epoch + 1
            break

    elapsed = time.time() - t0
    print(f"\n  Fine-tuning completed in {elapsed:.1f}s")
    print(f"  Best epoch: {best_epoch+1} (val_loss={best_val_loss:.6f})")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, train_losses, val_losses, best_epoch, stopped_epoch


# ---------------------------------------------------------------------------
# Learning curve plot
# ---------------------------------------------------------------------------
def plot_learning_curve(train_losses, val_losses, best_epoch, stopped_epoch):
    """Generate training/validation loss curve with annotations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=1.5)

    # Mark best epoch
    ax.axvline(
        x=best_epoch + 1, color="green", linestyle="--",
        alpha=0.7, label=f"Best Epoch ({best_epoch+1})"
    )

    # Mark early stopping if triggered
    if stopped_epoch < EPOCHS:
        ax.axvline(
            x=stopped_epoch, color="orange", linestyle=":",
            alpha=0.7, label=f"Early Stop ({stopped_epoch})"
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("TEMPO Fine-Tuning Learning Curve", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "tempo_learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Learning curve saved: {path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(ft_metrics):
    """Print fine-tuned vs zero-shot comparison table."""
    print("\n" + "=" * 64)
    print("FINE-TUNING RESULTS vs ZERO-SHOT")
    print("=" * 64)
    print(f"{'Site':<10}| {'Metric':<8}| {'Zero-Shot':>10}| "
          f"{'Fine-Tuned':>11}| {'Improvement':>12}")
    print("-" * 64)

    for site in TEST_SITES:
        zs = ZERO_SHOT_METRICS[site]
        ft = ft_metrics[site]

        for metric in ["RMSE", "MAE", "R2"]:
            zs_val = zs[metric]
            ft_val = ft[metric]

            if metric == "R2":
                # R² improvement as absolute difference
                diff = ft_val - zs_val
                imp_str = f"{diff:+.4f}"
            else:
                # RMSE/MAE improvement as percentage decrease
                pct = (zs_val - ft_val) / zs_val * 100
                imp_str = f"{pct:+.1f}%"

            print(
                f"{site:<10}| {metric:<8}| {zs_val:>10.4f}| "
                f"{ft_val:>11.4f}| {imp_str:>12}"
            )
        print("-" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("TEMPO FINE-TUNING FOR CARBON FLUX FORECASTING")
    print("=" * 64)

    device = get_device()
    print(f"\nDevice: {device}")

    # --- Load data ---
    print("\nLoading univariate NEE data...")
    train_X, train_y, test_data = load_all_data()
    print(f"\nTotal training: {len(train_X)} sequences")

    # --- Load pretrained TEMPO ---
    print("\nLoading pretrained TEMPO-80M...")
    model = load_tempo_model(device)
    print("  Model loaded successfully.")

    # --- Fine-tune ---
    print(f"\nFine-tuning (epochs={EPOCHS}, lr={LR}, patience={PATIENCE})...")
    model, train_losses, val_losses, best_epoch, stopped_epoch = fine_tune(
        model, train_X, train_y, device
    )

    # --- Learning curve ---
    print("\nGenerating learning curve plot...")
    plot_learning_curve(train_losses, val_losses, best_epoch, stopped_epoch)

    # --- Evaluate on test sites ---
    print("\nEvaluating fine-tuned model on test sites...")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    ft_metrics = {}
    for site in TEST_SITES:
        X_test = test_data[site]["X"]
        y_test = test_data[site]["y"]

        print(f"\n  Predicting {site} ({len(X_test)} samples)...")
        t0 = time.time()
        preds = predict_batched(model, X_test, device)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        metrics = compute_metrics(y_test, preds)
        ft_metrics[site] = metrics
        print(f"  {site}:  RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

        # Save predictions
        pred_path = PREDICTIONS_DIR / f"tempo_fine_tuned_preds_{site}.npy"
        np.save(pred_path, preds)
        print(f"  Predictions saved: {pred_path.name}")

    # --- Save metrics ---
    metrics_path = METRICS_DIR / "tempo_fine_tuned_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(ft_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # --- Comparison ---
    print_comparison(ft_metrics)

    print(f"\nCheckpoint: {CHECKPOINT_DIR / 'best_model.pth'}")
    print("Done.")


if __name__ == "__main__":
    main()
