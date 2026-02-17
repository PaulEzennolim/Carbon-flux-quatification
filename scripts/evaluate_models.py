"""
Unified Model Evaluation Script
================================

Trains and evaluates all models (RF, XGBoost, LSTM, TEMPO) on the held-out
UK-AMo and SE-Htm test sites. Saves per-model metrics to results/metrics/
as JSON and prints a comparison table.

Usage:
    python scripts/evaluate_models.py                # All models
    python scripts/evaluate_models.py --skip-tempo    # Baselines only
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_models import BaselineModels, evaluate as evaluate_sklearn
from models.lstm_baseline import LSTMForecaster, train_lstm, evaluate_lstm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_SITES = ["UK-AMo", "SE-Htm"]
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_processed_data():
    """Load multivariate processed .npy arrays."""
    X_train = np.load(DATA_DIR / "train_X.npy")
    y_train = np.load(DATA_DIR / "train_y.npy")

    test_data = {}
    for site in TEST_SITES:
        test_data[site] = {
            "X": np.load(DATA_DIR / f"test_{site}_X.npy"),
            "y": np.load(DATA_DIR / f"test_{site}_y.npy"),
        }

    return X_train, y_train, test_data


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------
def run_baselines(X_train, y_train, test_data):
    """Train and evaluate RF and XGBoost baselines."""
    baselines = BaselineModels()
    results = {}

    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = baselines.train_random_forest(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s")
    results["RandomForest"] = {}
    for site in TEST_SITES:
        metrics = evaluate_sklearn(rf, test_data[site]["X"], test_data[site]["y"])
        results["RandomForest"][site] = metrics
        print(f"  {site}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}")

    print("\n--- XGBoost ---")
    t0 = time.time()
    xgb = baselines.train_xgboost(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s")
    results["XGBoost"] = {}
    for site in TEST_SITES:
        metrics = evaluate_sklearn(xgb, test_data[site]["X"], test_data[site]["y"])
        results["XGBoost"][site] = metrics
        print(f"  {site}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}")

    return results, {"RandomForest": rf, "XGBoost": xgb}


# ---------------------------------------------------------------------------
# LSTM evaluation
# ---------------------------------------------------------------------------
def run_lstm(X_train, y_train, test_data):
    """Train and evaluate the LSTM baseline."""
    print("\n--- LSTM ---")
    input_size = X_train.shape[2]
    horizon = y_train.shape[1]
    model = LSTMForecaster(input_size=input_size, horizon=horizon)

    t0 = time.time()
    model = train_lstm(model, X_train, y_train, epochs=20, batch_size=64)
    print(f"  Trained in {time.time() - t0:.1f}s")

    results = {}
    for site in TEST_SITES:
        metrics = evaluate_lstm(model, test_data[site]["X"], test_data[site]["y"])
        results[site] = metrics
        print(f"  {site}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}")

    return results, model


# ---------------------------------------------------------------------------
# TEMPO evaluation
# ---------------------------------------------------------------------------
def run_tempo():
    """Run TEMPO zero-shot and fine-tuned evaluation (univariate NEE)."""
    try:
        from models.tempo_carbon_flux import (
            load_all_data, load_tempo_model, get_device,
            zero_shot_evaluate, fine_tune, evaluate_finetuned,
            TEST_SITES as TEMPO_TEST_SITES,
        )
    except ImportError as e:
        print(f"\n  TEMPO import failed: {e}")
        print("  Ensure the TEMPO repository is available at ../TEMPO/")
        return None

    device = get_device()
    print(f"\n--- TEMPO (device: {device}) ---")

    print("  Loading univariate NEE data...")
    train_X, train_y, test_data = load_all_data()

    print("  Loading pretrained TEMPO-80M...")
    model = load_tempo_model(device)

    # Zero-shot
    print("  Zero-shot evaluation...")
    results = {}
    zs_preds_all = {}
    zs_targets_all = {}
    for site in TEMPO_TEST_SITES:
        metrics, preds, y_sub, idx = zero_shot_evaluate(
            model, test_data[site]["X"], test_data[site]["y"], max_samples=500
        )
        results[f"TEMPO-ZeroShot"] = results.get("TEMPO-ZeroShot", {})
        results["TEMPO-ZeroShot"][site] = metrics
        zs_preds_all[site] = preds
        zs_targets_all[site] = y_sub
        print(f"  {site} ZS: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}")

    # Fine-tuning
    print("  Fine-tuning TEMPO...")
    t0 = time.time()
    model = fine_tune(model, train_X, train_y, device, epochs=5, max_samples=2000)
    print(f"  Fine-tuned in {time.time() - t0:.1f}s")

    results["TEMPO-FineTuned"] = {}
    ft_preds_all = {}
    ft_targets_all = {}
    for site in TEMPO_TEST_SITES:
        metrics, preds, targets = evaluate_finetuned(
            model, test_data[site]["X"], test_data[site]["y"], device
        )
        results["TEMPO-FineTuned"][site] = metrics
        ft_preds_all[site] = preds
        ft_targets_all[site] = targets
        print(f"  {site} FT: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}")

    # Save predictions for figure generation
    preds_dir = METRICS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for site in TEMPO_TEST_SITES:
        np.save(preds_dir / f"tempo_zs_preds_{site}.npy", zs_preds_all[site])
        np.save(preds_dir / f"tempo_zs_targets_{site}.npy", zs_targets_all[site])
        np.save(preds_dir / f"tempo_ft_preds_{site}.npy", ft_preds_all[site])
        np.save(preds_dir / f"tempo_ft_targets_{site}.npy", ft_targets_all[site])

    return results


# ---------------------------------------------------------------------------
# Save and display
# ---------------------------------------------------------------------------
def save_metrics(all_results):
    """Save metrics as JSON files."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model JSON files
    for model_name, site_metrics in all_results.items():
        out = {}
        for site, metrics in site_metrics.items():
            out[site] = {k: float(v) for k, v in metrics.items()}
        path = METRICS_DIR / f"{model_name.lower().replace('-', '_')}_metrics.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {path.name}")

    # Combined summary
    summary_path = METRICS_DIR / "all_metrics.json"
    combined = {}
    for model_name, site_metrics in all_results.items():
        combined[model_name] = {}
        for site, metrics in site_metrics.items():
            combined[model_name][site] = {k: float(v) for k, v in metrics.items()}
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved {summary_path.name}")


def save_predictions(models, test_data):
    """Save baseline predictions for figure generation."""
    preds_dir = METRICS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        for site in TEST_SITES:
            X = test_data[site]["X"]
            y = test_data[site]["y"]
            X_flat = X.reshape(X.shape[0], -1)
            preds = model.predict(X_flat)
            np.save(preds_dir / f"{model_name.lower()}_preds_{site}.npy", preds)
            np.save(preds_dir / f"targets_{site}.npy", y)


def save_lstm_predictions(model, test_data):
    """Save LSTM predictions for figure generation."""
    preds_dir = METRICS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()
    for site in TEST_SITES:
        X = test_data[site]["X"]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(X)),
            batch_size=256,
        )
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                preds.append(model(bx.to(device)).cpu().numpy())
        preds = np.concatenate(preds)
        np.save(preds_dir / f"lstm_preds_{site}.npy", preds)


def print_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 78)
    print("MODEL COMPARISON TABLE")
    print("=" * 78)

    header = f"{'Model':<20} {'Site':<8} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
    print(header)
    print("-" * 78)

    for model_name in all_results:
        for site in TEST_SITES:
            m = all_results[model_name][site]
            print(f"{model_name:<20} {site:<8} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--skip-tempo", action="store_true",
                        help="Skip TEMPO evaluation (requires TEMPO repo)")
    args = parser.parse_args()

    print("=" * 60)
    print("UNIFIED MODEL EVALUATION")
    print("=" * 60)

    # Load multivariate processed data (for baselines + LSTM)
    print("\nLoading processed data...")
    X_train, y_train, test_data = load_processed_data()
    print(f"  Train: {X_train.shape}  |  "
          f"UK-AMo: {test_data['UK-AMo']['X'].shape[0]}  |  "
          f"SE-Htm: {test_data['SE-Htm']['X'].shape[0]} samples")

    all_results = {}

    # --- Baselines ---
    baseline_results, sklearn_models = run_baselines(X_train, y_train, test_data)
    all_results.update(baseline_results)

    # --- LSTM ---
    lstm_results, lstm_model = run_lstm(X_train, y_train, test_data)
    all_results["LSTM"] = lstm_results

    # --- TEMPO ---
    if not args.skip_tempo:
        tempo_results = run_tempo()
        if tempo_results:
            all_results.update(tempo_results)
    else:
        print("\n--- TEMPO skipped (--skip-tempo) ---")

    # --- Save ---
    print("\nSaving results...")
    save_metrics(all_results)
    save_predictions(sklearn_models, test_data)
    save_lstm_predictions(lstm_model, test_data)

    # --- Print table ---
    print_table(all_results)

    print(f"Results saved to {METRICS_DIR}/")


if __name__ == "__main__":
    main()
