"""
TEMPO Zero-Shot Inference on Carbon Flux Test Sites
=====================================================

Runs the pretrained TEMPO-80M model (no fine-tuning) on held-out
UK-AMo and SE-Htm test sites. Uses preprocessed .npy arrays from
the multivariate data pipeline, extracting only the NEE lookback
channel for TEMPO's univariate input.

Usage:
    python scripts/run_zero_shot_tempo.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Add TEMPO repository to path
TEMPO_DIR = PROJECT_ROOT.parent / "TEMPO"
sys.path.insert(0, str(TEMPO_DIR))

from tempo.models.TEMPO import TEMPO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZON = 96        # 4-day forecast (matching y shape)
BATCH_SIZE = 32     # Batch size for inference
SEED = 42
TEST_SITES = ["UK-AMo", "SE-Htm"]

# NEE is the target variable — in the multivariate .npy arrays the
# features are [SW_IN_F, LW_IN_F, VPD_F, TA_F, PA_F, P_F, WS_F,
# G_F_MDS, LE_F_MDS, H_F_MDS, MODIS_band_1..7, DOY, TOD].
# NEE is NOT one of the 19 input features — we predict it as the
# target (y). For TEMPO zero-shot we feed the lookback portion of y
# (the NEE signal) as univariate input.
#
# However, the .npy files store X (features) and y (targets)
# separately. Since we don't have the NEE lookback in X, we
# reconstruct the univariate NEE lookback from y of the preceding
# window. Instead, we load raw NEE series and create sequences,
# matching the approach in tempo_carbon_flux.py.

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
SITE_FILES = {
    "UK-AMo": "6.UK-AMo.csv",
    "SE-Htm": "7.SE-Htm.csv",
}
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOOKBACK = 336  # 2 weeks hourly


def load_test_data():
    """
    Load univariate NEE test sequences for both sites.

    Creates sliding windows (336 -> 96) from the raw NEE signal,
    matching the same sequence count as the multivariate .npy files.
    """
    import pandas as pd

    test_data = {}
    for site in TEST_SITES:
        filepath = RAW_DIR / SITE_FILES[site]
        df = pd.read_csv(filepath)
        nee = df["NEE_VUT_REF"].ffill().bfill().values.astype(np.float32)

        X, y = [], []
        for i in range(len(nee) - LOOKBACK - HORIZON + 1):
            X.append(nee[i : i + LOOKBACK])
            y.append(nee[i + LOOKBACK : i + LOOKBACK + HORIZON])

        test_data[site] = {
            "X": np.array(X),
            "y": np.array(y),
        }
        print(f"  {site}: {len(X)} sequences ({len(nee)} timesteps)")

    return test_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tempo_model(device):
    """Load pretrained TEMPO-80M from HuggingFace."""
    cache_dir = str(PROJECT_ROOT / "models" / "checkpoints" / "tempo_zero_shot")
    print(f"  Cache directory: {cache_dir}")

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
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# Zero-shot inference (batched)
# ---------------------------------------------------------------------------
def zero_shot_predict(model, X, device, batch_size=BATCH_SIZE):
    """
    Run batched zero-shot inference with TEMPO.

    TEMPO forward() expects input shape [B, L, 1] (univariate).
    We pass dummy STL components (trend=season=noise=input) and
    extract the last HORIZON timesteps from the output.
    """
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(-1))  # [N, 336, 1]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            outputs, _ = model(bx, itr=0, trend=bx, season=bx, noise=bx)
            # Extract last HORIZON timesteps, squeeze channel dim
            preds = outputs[:, -HORIZON:, :].cpu().numpy().squeeze(-1)
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TEMPO ZERO-SHOT INFERENCE")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice: {device}")

    # --- Load test data ---
    print("\nLoading test site NEE data...")
    test_data = load_test_data()

    # --- Load model ---
    print("\nLoading pretrained TEMPO-80M...")
    model = load_tempo_model(device)
    print("  Model loaded successfully.")

    # --- Inference ---
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for site in TEST_SITES:
        X_test = test_data[site]["X"]
        y_test = test_data[site]["y"]

        print(f"\nRunning zero-shot inference on {site} "
              f"({len(X_test)} samples, batch_size={BATCH_SIZE})...")
        t0 = time.time()
        preds = zero_shot_predict(model, X_test, device)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Metrics
        metrics = compute_metrics(y_test, preds)
        all_metrics[site] = metrics
        print(f"  {site}:  RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

        # Save predictions
        pred_path = PREDICTIONS_DIR / f"tempo_zero_shot_preds_{site}.npy"
        np.save(pred_path, preds)
        print(f"  Predictions saved: {pred_path.name}")

    # --- Save metrics ---
    metrics_path = METRICS_DIR / "tempo_zero_shot_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ZERO-SHOT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Site':<10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 42)
    for site in TEST_SITES:
        m = all_metrics[site]
        print(f"{site:<10} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
