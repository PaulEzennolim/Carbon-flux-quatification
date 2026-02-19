"""
TEMPO Carbon Flux Forecasting — Consolidated Script
=====================================================

Evaluates the TEMPO foundation model for cross-site NEE (Net Ecosystem Exchange)
prediction at two held-out EC tower sites: UK-AMo and SE-Htm.

Workflow:
1. Load pretrained TEMPO-80M from HuggingFace
2. Zero-shot evaluation (no training) on test sites
3. Fine-tune TEMPO on 5 training sites
4. Re-evaluate fine-tuned model on test sites
5. Compute metrics (RMSE, MAE, R²) and generate comparison plots

TEMPO is a univariate time series foundation model, so we extract the NEE
signal from raw site CSVs and create sliding windows (336→96 timesteps).
"""

import sys
import os
import warnings
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tempo.models.TEMPO import TEMPO

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK = 336   # 2 weeks of hourly data
HORIZON = 96     # 4 days ahead
SEED = 42

TRAIN_SITES = ['FI-Lom', 'GL-ZaF', 'IE-Cra', 'DE-Akm', 'FR-LGt']
TEST_SITES = ['UK-AMo', 'SE-Htm']

SITE_FILES = {
    'FI-Lom': '1.FI-Lom.csv',
    'GL-ZaF': '2.GL-ZaF.csv',
    'IE-Cra': '3.IE-Cra.xlsx',
    'DE-Akm': '4.DE-Akm.csv',
    'FR-LGt': '5.FR-LGt.csv',
    'UK-AMo': '6.UK-AMo.csv',
    'SE-Htm': '7.SE-Htm.csv',
}

RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    # TEMPO uses GPT-2 token embeddings incompatible with MPS — use CUDA or CPU
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_nee_series(site_name):
    """Load raw NEE time series for a site, gap-filled with ffill/bfill."""
    filepath = RAW_DIR / SITE_FILES[site_name]
    if filepath.suffix == '.xlsx':
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    nee = df['NEE_VUT_REF'].ffill().bfill().values.astype(np.float32)
    return nee


def create_sequences(series, lookback=LOOKBACK, horizon=HORIZON):
    """Create sliding-window sequences from a 1-D array."""
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def load_all_data():
    """Load univariate NEE sequences for all train and test sites."""
    train_X_list, train_y_list = [], []
    for site in TRAIN_SITES:
        nee = load_nee_series(site)
        X, y = create_sequences(nee)
        train_X_list.append(X)
        train_y_list.append(y)
        print(f"  {site}: {len(X)} sequences ({len(nee)} timesteps)")

    train_X = np.concatenate(train_X_list)
    train_y = np.concatenate(train_y_list)

    test_data = {}
    for site in TEST_SITES:
        nee = load_nee_series(site)
        X, y = create_sequences(nee)
        test_data[site] = {'X': X, 'y': y}
        print(f"  {site}: {len(X)} sequences ({len(nee)} timesteps)")

    return train_X, train_y, test_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tempo_model(device):
    """Load the pretrained TEMPO-80M model from HuggingFace."""
    # Use project-relative cache directory
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = str(project_root / 'checkpoints' / 'TEMPO_checkpoints')
    
    model = TEMPO.load_pretrained_model(
        device=device,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R² between arrays of shape (N, horizon)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------
def zero_shot_evaluate(model, X, y, max_samples=500):
    """
    Evaluate the pretrained TEMPO model without any fine-tuning.

    Uses model.predict() which handles a single univariate input at a time.
    We subsample to keep runtime reasonable.
    """
    n = len(X)
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    preds = []
    for i in idx:
        pred = model.predict(X[i], pred_length=HORIZON)
        preds.append(pred)

    preds = np.array(preds)
    y_sub = y[idx]
    return compute_metrics(y_sub, preds), preds, y_sub, idx


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def fine_tune(model, train_X, train_y, device, epochs=10, batch_size=64,
              lr=1e-4, val_split=0.1, patience=3, max_samples=5000):
    """
    Fine-tune TEMPO on the carbon flux training data.

    The TEMPO forward() method expects input shape [B, L, 1] (univariate)
    and returns (outputs, loss_local) where outputs is [B, pred_len, 1].
    """
    model.train()

    # Subsample if dataset is too large for CPU training
    n = len(train_X)
    if max_samples and n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        train_X = train_X[idx]
        train_y = train_y[idx]
        n = max_samples
        print(f"  Subsampled to {n} training sequences")

    # Train / validation split
    n_val = max(1, int(n * val_split))
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_tr = torch.FloatTensor(train_X[train_idx]).unsqueeze(-1)  # [N, 336, 1]
    y_tr = torch.FloatTensor(train_y[train_idx]).unsqueeze(-1)  # [N, 96, 1]
    X_val = torch.FloatTensor(train_X[val_idx]).unsqueeze(-1)
    y_val = torch.FloatTensor(train_y[val_idx]).unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-8
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            # Pass dummy STL components (same as input) so TEMPO
            # computes loss_local for the internal decomposition
            outputs, loss_local = model(
                bx, itr=0, trend=bx, season=bx, noise=bx
            )
            outputs = outputs[:, -HORIZON:, :]
            by = by[:, -HORIZON:, :]
            loss = criterion(outputs, by)
            if loss_local is not None:
                loss = loss + 0.01 * loss_local
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs, _ = model(bx, itr=0, trend=bx, season=bx, noise=bx)
                outputs = outputs[:, -HORIZON:, :]
                by = by[:, -HORIZON:, :]
                val_losses.append(criterion(outputs, by).item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"  Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {avg_train:.6f}  Val Loss: {avg_val:.6f}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluate fine-tuned model (batched forward pass)
# ---------------------------------------------------------------------------
def evaluate_finetuned(model, X, y, device, batch_size=256):
    """Evaluate the fine-tuned model using batched forward passes."""
    model.eval()
    dataset = TensorDataset(
        torch.FloatTensor(X).unsqueeze(-1),
        torch.FloatTensor(y).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            outputs, _ = model(bx, itr=0, trend=bx, season=bx, noise=bx)
            outputs = outputs[:, -HORIZON:, :]
            all_preds.append(outputs.cpu().numpy().squeeze(-1))
            all_targets.append(by.numpy().squeeze(-1))

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(targets, preds), preds, targets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(results, save_dir):
    """Generate bar charts comparing zero-shot vs fine-tuned metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['RMSE', 'MAE', 'R2']
    sites = list(results.keys())
    x = np.arange(len(sites))
    width = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        zs_vals = [results[s]['zero_shot'][metric] for s in sites]
        ft_vals = [results[s]['fine_tuned'][metric] for s in sites]

        ax.bar(x - width / 2, zs_vals, width, label='Zero-shot', color='steelblue')
        ax.bar(x + width / 2, ft_vals, width, label='Fine-tuned', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()

    fig.suptitle('TEMPO: Zero-shot vs Fine-tuned on Held-out Sites', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'tempo_metrics_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved metrics comparison plot")


def plot_forecast_samples(site, y_true, y_pred_zs, y_pred_ft,
                          idx_zs, save_dir, n_samples=3):
    """Plot example forecasts for a test site."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    sample_indices = np.linspace(0, len(y_true) - 1, n_samples, dtype=int)
    hours = np.arange(HORIZON)

    for ax, si in zip(axes, sample_indices):
        ax.plot(hours, y_true[si], 'k-', label='Observed', linewidth=1.5)
        # Find matching zero-shot prediction if available
        zs_match = np.where(idx_zs == si)[0]
        if len(zs_match) > 0:
            ax.plot(hours, y_pred_zs[zs_match[0]], 'b--', label='Zero-shot', alpha=0.8)
        ax.plot(hours, y_pred_ft[si], 'r--', label='Fine-tuned', alpha=0.8)
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('NEE')
        ax.set_title(f'{site} — Sample {si}')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / f'tempo_forecasts_{site}.png', dpi=150)
    plt.close()
    print(f"  Saved forecast samples for {site}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Using device: {device}\n")

    # --- Load data ---
    print("Loading NEE data from raw site files...")
    train_X, train_y, test_data = load_all_data()
    print(f"\nTraining: {train_X.shape[0]} sequences")
    for site in TEST_SITES:
        print(f"Test {site}: {test_data[site]['X'].shape[0]} sequences")

    # --- Load pretrained TEMPO ---
    print("\nLoading pretrained TEMPO-80M...")
    model = load_tempo_model(device)
    print("Model loaded successfully.\n")

    # --- Zero-shot evaluation ---
    print("=" * 60)
    print("ZERO-SHOT EVALUATION (no training)")
    print("=" * 60)
    zs_results = {}
    zs_preds = {}
    zs_indices = {}
    for site in TEST_SITES:
        X_test = test_data[site]['X']
        y_test = test_data[site]['y']
        print(f"\n  Evaluating {site} ({len(X_test)} samples, subsampling 500)...")
        metrics, preds, y_sub, idx = zero_shot_evaluate(model, X_test, y_test)
        zs_results[site] = metrics
        zs_preds[site] = preds
        zs_indices[site] = idx
        print(f"  {site} Zero-shot:  RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

    # --- Fine-tuning ---
    print("\n" + "=" * 60)
    print("FINE-TUNING TEMPO")
    print("=" * 60)
    print(f"\n  Training on {train_X.shape[0]} sequences from {len(TRAIN_SITES)} sites...")
    t0 = time.time()
    model = fine_tune(model, train_X, train_y, device,
                      epochs=5, max_samples=2000)
    elapsed = time.time() - t0
    print(f"  Fine-tuning completed in {elapsed:.1f}s\n")

    # --- Fine-tuned evaluation ---
    print("=" * 60)
    print("FINE-TUNED EVALUATION")
    print("=" * 60)
    ft_results = {}
    ft_preds = {}
    ft_targets = {}
    for site in TEST_SITES:
        X_test = test_data[site]['X']
        y_test = test_data[site]['y']
        metrics, preds, targets = evaluate_finetuned(model, X_test, y_test, device)
        ft_results[site] = metrics
        ft_preds[site] = preds
        ft_targets[site] = targets
        print(f"  {site} Fine-tuned: RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    combined = {}
    for site in TEST_SITES:
        combined[site] = {
            'zero_shot': zs_results[site],
            'fine_tuned': ft_results[site],
        }
        print(f"\n  {site}:")
        print(f"    Zero-shot  — RMSE: {zs_results[site]['RMSE']:.4f}, "
              f"MAE: {zs_results[site]['MAE']:.4f}, R2: {zs_results[site]['R2']:.4f}")
        print(f"    Fine-tuned — RMSE: {ft_results[site]['RMSE']:.4f}, "
              f"MAE: {ft_results[site]['MAE']:.4f}, R2: {ft_results[site]['R2']:.4f}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_comparison(combined, RESULTS_DIR)
    for site in TEST_SITES:
        plot_forecast_samples(
            site,
            ft_targets[site],
            zs_preds[site],
            ft_preds[site],
            zs_indices[site],
            RESULTS_DIR,
        )

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
