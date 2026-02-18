"""
Baseline Model Training Script
================================

Trains Random Forest, XGBoost, and LSTM baselines for carbon flux prediction.
Evaluates on held-out UK-AMo and SE-Htm test sites, saves checkpoints,
predictions, and a CSV summary of all metrics.

Usage:
    python scripts/train_baselines.py
"""

import sys
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import joblib
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "baselines"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions" / "baselines"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_models import BaselineModels, evaluate as evaluate_sklearn
from models.lstm_baseline import LSTMForecaster, train_lstm, evaluate_lstm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_SITES = ["UK-AMo", "SE-Htm"]
SEED = 42
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 10,
        "n_jobs": -1,
        "random_state": SEED,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "LSTM": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
    },
}


# ---------------------------------------------------------------------------
# NaN imputation safety net
# ---------------------------------------------------------------------------
NAN_FEATURE_INDICES = [1, 4, 5, 6, 7]  # Features with known NaN from diagnostics


def impute_features(X, feature_indices=None, fit_values=None):
    """
    Targeted median imputation for known problematic features.
    Only imputes features 1, 4, 5, 6, 7 which have NaN.
    Features 0, 2, 3, 8-18 are clean and left untouched.

    Args:
        X: shape (n_samples, seq_len, n_features)
        feature_indices: which features to impute
        fit_values: pre-computed medians (use training medians for test data)

    Returns:
        X_clean: imputed array
        medians: computed medians (save for test imputation)
    """
    if feature_indices is None:
        feature_indices = NAN_FEATURE_INDICES
    X_clean = X.copy()
    medians = {}

    for feat_idx in feature_indices:
        feature_data = X_clean[:, :, feat_idx]

        if fit_values is None:
            median_val = np.nanmedian(feature_data)
            medians[feat_idx] = median_val
        else:
            median_val = fit_values[feat_idx]
            medians[feat_idx] = median_val

        nan_mask = np.isnan(feature_data)
        X_clean[:, :, feat_idx][nan_mask] = median_val

    return X_clean, medians


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load training and test data from .npy files."""
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
# Evaluation + prediction helpers
# ---------------------------------------------------------------------------
def evaluate_and_predict_sklearn(model, X, y):
    """Evaluate an sklearn-style model and return metrics + predictions."""
    X_flat = X.reshape(X.shape[0], -1)
    preds = model.predict(X_flat)
    metrics = evaluate_sklearn(model, X, y)
    return metrics, preds


def evaluate_and_predict_lstm(model, X, y):
    """Evaluate the LSTM and return metrics + predictions."""
    device = next(model.parameters()).device
    model.eval()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    metrics = evaluate_lstm(model, X, y)
    return metrics, preds


# ---------------------------------------------------------------------------
# Training: Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train, test_data):
    """Train and evaluate Random Forest."""
    hp = HYPERPARAMS["RandomForest"]
    log.info("  n_estimators=%d  max_depth=%d  min_samples_split=%d",
             hp["n_estimators"], hp["max_depth"], hp["min_samples_split"])

    baselines = BaselineModels()
    log.info("  Training on %d samples...", len(X_train))
    t0 = time.time()
    rf = baselines.train_random_forest(X_train, y_train)
    elapsed = time.time() - t0
    log.info("  ✓ Completed in %.1fs", elapsed)

    # Evaluate on test sites
    results = {}
    preds_dict = {}
    print(f"\n  {'Site':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for site in TEST_SITES:
        metrics, preds = evaluate_and_predict_sklearn(
            rf, test_data[site]["X"], test_data[site]["y"]
        )
        results[site] = metrics
        preds_dict[site] = preds
        print(f"  {site:<10} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} {metrics['R2']:>8.4f}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"randomforest_{TIMESTAMP}.joblib"
    joblib.dump(rf, ckpt_path)
    log.info("  ✓ Checkpoint: models/checkpoints/baselines/%s", ckpt_path.name)

    return results, preds_dict


# ---------------------------------------------------------------------------
# Training: XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, test_data):
    """Train and evaluate XGBoost."""
    hp = HYPERPARAMS["XGBoost"]
    log.info("  n_estimators=%d  learning_rate=%.3f  max_depth=%d",
             hp["n_estimators"], hp["learning_rate"], hp["max_depth"])

    baselines = BaselineModels()
    log.info("  Training on %d samples...", len(X_train))
    t0 = time.time()
    xgb = baselines.train_xgboost(X_train, y_train)
    elapsed = time.time() - t0
    log.info("  ✓ Completed in %.1fs", elapsed)

    # Evaluate on test sites
    results = {}
    preds_dict = {}
    print(f"\n  {'Site':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for site in TEST_SITES:
        metrics, preds = evaluate_and_predict_sklearn(
            xgb, test_data[site]["X"], test_data[site]["y"]
        )
        results[site] = metrics
        preds_dict[site] = preds
        print(f"  {site:<10} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} {metrics['R2']:>8.4f}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"xgboost_{TIMESTAMP}.joblib"
    joblib.dump(xgb, ckpt_path)
    log.info("  ✓ Checkpoint: models/checkpoints/baselines/%s", ckpt_path.name)

    return results, preds_dict


# ---------------------------------------------------------------------------
# Training: LSTM (with validation and progress bars)
# ---------------------------------------------------------------------------
def train_lstm_model(X_train, y_train, test_data):
    """Train and evaluate the LSTM with per-epoch validation."""
    params = HYPERPARAMS["LSTM"]
    input_size = X_train.shape[2]
    horizon = y_train.shape[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("  Architecture:  2-layer LSTM (%d units, dropout=%.1f)",
             params["hidden_size"], params["dropout"])
    log.info("  Device:        %s", device)

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        horizon=horizon,
    ).to(device)

    # Validation split (10% of training data)
    n = len(X_train)
    idx = np.random.permutation(n)
    val_size = max(1, int(0.1 * n))
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    X_trn, y_trn = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_trn), torch.FloatTensor(y_trn)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in tqdm(range(params["epochs"]), desc="  LSTM epochs"):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_dataset)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch {epoch+1:3d}/{params['epochs']}  │  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    log.info("  ✓ Completed in %.0fm %.0fs", m, s)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        log.info("  ✓ Best model restored (val_loss=%.6f)", best_val_loss)

    # Evaluate on test sites
    results = {}
    preds_dict = {}
    print(f"\n  {'Site':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for site in TEST_SITES:
        metrics, preds = evaluate_and_predict_lstm(
            model, test_data[site]["X"], test_data[site]["y"]
        )
        results[site] = metrics
        preds_dict[site] = preds
        print(f"  {site:<10} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} {metrics['R2']:>8.4f}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"lstm_{TIMESTAMP}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": params,
            "input_size": input_size,
            "horizon": horizon,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    log.info("  ✓ Checkpoint: models/checkpoints/baselines/%s", ckpt_path.name)

    return results, preds_dict


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------
def save_predictions(all_preds):
    """Save per-model, per-site prediction arrays."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, site_preds in all_preds.items():
        for site, preds in site_preds.items():
            path = PREDICTIONS_DIR / f"{model_name.lower()}_preds_{site}.npy"
            np.save(path, preds)
    log.info("Predictions saved to %s/", PREDICTIONS_DIR.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Save metrics CSV
# ---------------------------------------------------------------------------
def save_metrics_csv(all_results):
    """Save a combined CSV with all model/site metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "baseline_results.csv"

    rows = []
    for model_name, site_metrics in all_results.items():
        for site, metrics in site_metrics.items():
            rows.append({
                "Model": model_name,
                "Site": site,
                "RMSE": f"{metrics['RMSE']:.4f}",
                "MAE": f"{metrics['MAE']:.4f}",
                "R2": f"{metrics['R2']:.4f}",
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Site", "RMSE", "MAE", "R2"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Metrics saved to %s", csv_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Model':<16} {'Site':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 72)

    for model_name in all_results:
        for site in TEST_SITES:
            m = all_results[model_name][site]
            print(
                f"{model_name:<16} {site:<8} "
                f"{m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    print("=" * 80)
    print("BASELINE MODEL TRAINING FOR CARBON FLUX PREDICTION")
    print(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed:       {SEED}")
    print("=" * 80)

    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters
    hp_path = CHECKPOINT_DIR / f"hyperparameters_{TIMESTAMP}.json"
    with open(hp_path, "w") as f:
        json.dump(HYPERPARAMS, f, indent=2)

    # ── [1/6] Load data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading Data")
    print("─" * 80)
    try:
        X_train, y_train, test_data = load_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found")
        print(f"   ├─ Missing: {e}")
        print(f"   └─ Solution: Run 'python scripts/tempo_data_prep.py' first")
        sys.exit(1)

    n_train = X_train.shape[0]
    print(f"  ✓ Training data:    {n_train:>8,} sequences  "
          f"(lookback={X_train.shape[1]}, features={X_train.shape[2]})")
    for site in TEST_SITES:
        n_site = test_data[site]["X"].shape[0]
        print(f"  ✓ Test {site}:   {n_site:>8,} sequences")

    # ── [2/6] Data validation ─────────────────────────────────────────────────
    print("\n[2/6] Data Validation")
    print("─" * 80)
    log.info("Validating data quality...")
    X_train, train_medians = impute_features(X_train)
    for site in TEST_SITES:
        test_data[site]["X"], _ = impute_features(
            test_data[site]["X"], fit_values=train_medians
        )
    assert np.isnan(X_train).sum() == 0, "NaN detected in training data"
    assert all(
        np.isnan(test_data[s]["X"]).sum() == 0 for s in TEST_SITES
    ), "NaN detected in test data"
    print("  ✓ Data validation passed (0 NaN detected)")

    all_results = {}
    all_preds = {}

    # ── [3/6] Random Forest ───────────────────────────────────────────────────
    print("\n[3/6] Training Random Forest")
    print("─" * 80)
    try:
        rf_results, rf_preds = train_random_forest(X_train, y_train, test_data)
        all_results["RandomForest"] = rf_results
        all_preds["RandomForest"] = rf_preds
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")

    # ── [4/6] XGBoost ─────────────────────────────────────────────────────────
    print("\n[4/6] Training XGBoost")
    print("─" * 80)
    try:
        xgb_results, xgb_preds = train_xgboost(X_train, y_train, test_data)
        all_results["XGBoost"] = xgb_results
        all_preds["XGBoost"] = xgb_preds
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")

    # ── [5/6] LSTM ────────────────────────────────────────────────────────────
    print("\n[5/6] Training LSTM")
    print("─" * 80)
    try:
        lstm_results, lstm_preds = train_lstm_model(X_train, y_train, test_data)
        all_results["LSTM"] = lstm_results
        all_preds["LSTM"] = lstm_preds
    except Exception as e:
        print(f"  ❌ LSTM failed: {e}")

    if not all_results:
        print("\n❌ ERROR: All models failed. Exiting.")
        sys.exit(1)

    # ── [6/6] Save outputs + summary ─────────────────────────────────────────
    print("\n[6/6] Saving Outputs")
    print("─" * 80)
    save_predictions(all_preds)
    save_metrics_csv(all_results)
    for site in TEST_SITES:
        np.save(PREDICTIONS_DIR / f"targets_{site}.npy", test_data[site]["y"])
    print(f"  ✓ Targets:      results/predictions/baselines/targets_*.npy")
    print(f"  ✓ Hyperparams:  models/checkpoints/baselines/{hp_path.name}")

    print_summary(all_results)

    elapsed = time.time() - t_start
    n_ok = len(all_results)
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║  TRAINING COMPLETE" + " " * 59 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Models trained:   {n_ok}/3 successful"
          + " " * (39 - len(str(n_ok))) + "║")
    print(f"║  Total runtime:    {elapsed // 60:.0f}m {elapsed % 60:.0f}s"
          + " " * (54 - len(f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s")) + "║")
    print(f"║  Checkpoints:      models/checkpoints/baselines/" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")


if __name__ == "__main__":
    main()
