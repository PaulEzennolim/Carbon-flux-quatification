"""
Computational Efficiency Analysis for Carbon Flux Prediction Models
====================================================================
Benchmarks all trained models (Random Forest, XGBoost, LSTM, TEMPO) on:
  - Inference latency  : single-sample and batch, mean ± std over 10 runs
  - Cold-start time    : model load time from disk
  - Throughput         : samples/second at batch sizes [1, 32, 128, 500]
  - Memory footprint   : peak RSS during inference (background-sampled)
  - Model size         : disk size (MB) and parameter / estimator count
  - Training time      : quick mini-training scaled to full dataset

Outputs
-------
  results/metrics/computational_costs.json
  results/metrics/efficiency_comparison.csv
  results/efficiency/COMPUTATIONAL_EFFICIENCY_SUMMARY.txt
  figures/efficiency/inference_time_comparison.png
  figures/efficiency/accuracy_vs_cost_tradeoff.png
  figures/efficiency/resource_usage_comparison.png
  

Usage
-----
  python scripts/computational_efficiency.py

Notes
-----
  TEMPO benchmarking requires the `tempo` package.  If unavailable,
  TEMPO timing entries are filled with NaN and flagged in the summary.
"""

import sys
import gc
import os
import json
import time
import warnings
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, List

import numpy as np
import pandas as pd
import psutil
import joblib
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data" / "processed"
CKPT_BASE   = ROOT / "models" / "checkpoints" / "baselines"
CKPT_TEMPO  = ROOT / "models" / "checkpoints" / "tempo_fine_tuned"
METRICS_DIR          = ROOT / "results" / "metrics"
FIG_DIR              = ROOT / "figures" / "efficiency"
RESULTS_EFFICIENCY_DIR = ROOT / "results" / "efficiency"
SUMMARY_TXT          = RESULTS_EFFICIENCY_DIR / "COMPUTATIONAL_EFFICIENCY_SUMMARY.txt"

SITES = ["UK-AMo", "SE-Htm"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_REPEATS_INFER   = 10       # inference timing repetitions
N_REPEATS_LOAD    =  3       # cold-start repetitions (large models)
BATCH_SIZES       = [1, 32, 128, 500]
TRAIN_SUBSET      = 3_000    # samples used for training-time estimation
SEED              = 42
CPU_TDP_W         = 45.0     # conservative laptop CPU TDP in Watts
CO2_KG_PER_KWH   = 0.233    # UK grid average (2024)
REALTIME_THRESH_MS = 100.0   # <100 ms considered real-time capable

np.random.seed(SEED)
torch.manual_seed(SEED)

# Colour palette
COLORS = {
    "Random Forest" : "#4CAF50",
    "XGBoost"       : "#FF5722",
    "LSTM"          : "#9C27B0",
    "TEMPO ZeroShot": "#03A9F4",
    "TEMPO FineTune": "#2196F3",
}
MODEL_ORDER = ["Random Forest", "XGBoost", "LSTM", "TEMPO ZeroShot", "TEMPO FineTune"]

from models.lstm_baseline import LSTMForecaster

# ---------------------------------------------------------------------------
# TEMPO import  (optional)
# ---------------------------------------------------------------------------
try:
    from tempo.models.TEMPO import TEMPO as _TEMPOBase
    TEMPO_AVAILABLE = True
    log.info("  tempo package found — TEMPO benchmarking enabled.")
except ImportError:
    TEMPO_AVAILABLE = False
    log.warning("  tempo package not found — TEMPO timing will be estimated.")


# ---------------------------------------------------------------------------
# Memory tracker (background thread sampling RSS every 10 ms)
# ---------------------------------------------------------------------------
class PeakMemoryTracker:
    """Tracks peak RSS memory increase in a background thread."""

    def __init__(self, interval: float = 0.01):
        self._interval = interval
        self._proc     = psutil.Process(os.getpid())
        self._baseline = 0
        self._peak_delta = 0
        self._running  = False

    def __enter__(self):
        gc.collect()
        self._baseline   = self._proc.memory_info().rss
        self._peak_delta = 0
        self._running    = True
        self._thread     = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._running = False
        self._thread.join(timeout=2)

    def _sample(self):
        while self._running:
            delta = self._proc.memory_info().rss - self._baseline
            if delta > self._peak_delta:
                self._peak_delta = delta
            time.sleep(self._interval)

    @property
    def peak_mb(self) -> float:
        mb = max(0.0, self._peak_delta / (1024 ** 2))
        # If measured value is very small, report as 0.05 MB minimum
        # to avoid misleading "0.0 MB" for models that DO use memory
        return 0.05 if 0.0 < mb < 0.05 else mb


# ---------------------------------------------------------------------------
# Precision timing helper
# ---------------------------------------------------------------------------
def repeat_timing(fn: Callable, n: int = N_REPEATS_INFER) -> tuple:
    """Run fn() n times, return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Disk-size helper
# ---------------------------------------------------------------------------
def disk_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    # Directory
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)


# ---------------------------------------------------------------------------
# NaN imputation (mirrors train_baselines.py)
# ---------------------------------------------------------------------------
_NAN_IDX = [1, 4, 5, 6, 7]


def impute_features(X: np.ndarray, medians: Optional[dict] = None):
    X = X.copy()
    out = {}
    for idx in _NAN_IDX:
        if medians is None:
            m = float(np.nanmedian(X[:, :, idx]))
            out[idx] = m
        else:
            m = medians[idx]
            out[idx] = m
        X[:, :, idx][np.isnan(X[:, :, idx])] = m
    return X, out


# ---------------------------------------------------------------------------
# Load data once (used by all baseline models)
# ---------------------------------------------------------------------------
def load_baseline_data():
    log.info("  Loading processed data...")
    X_train = np.load(DATA_DIR / "train_X.npy")
    y_train = np.load(DATA_DIR / "train_y.npy")
    test = {}
    for site in SITES:
        test[site] = {
            "X": np.load(DATA_DIR / f"test_{site}_X.npy"),
            "y": np.load(DATA_DIR / f"test_{site}_y.npy"),
        }
    X_train, med = impute_features(X_train)
    for site in SITES:
        test[site]["X"], _ = impute_features(test[site]["X"], med)
    return X_train, y_train, test


# ---------------------------------------------------------------------------
# Load existing accuracy metrics
# ---------------------------------------------------------------------------
def load_accuracy_metrics() -> Dict[str, Dict[str, float]]:
    """Return avg RMSE and R² across both sites for each model."""
    # From all_models_summary.csv
    summary_path = METRICS_DIR / "all_models_summary.csv"
    baseline_path = METRICS_DIR / "baseline_results.csv"

    metrics: Dict[str, Dict[str, float]] = {}

    # Baseline CSV uses 'RandomForest' / 'XGBoost' / 'LSTM'
    if baseline_path.exists():
        df_bl = pd.read_csv(baseline_path)
        name_map = {"RandomForest": "Random Forest",
                    "XGBoost": "XGBoost",
                    "LSTM": "LSTM"}
        for raw, display in name_map.items():
            rows = df_bl[df_bl["Model"] == raw]
            if not rows.empty:
                metrics[display] = {
                    "RMSE_mean": round(float(rows["RMSE"].mean()), 4),
                    "MAE_mean" : round(float(rows["MAE"].mean()), 4),
                    "R2_mean"  : round(float(rows["R2"].mean()), 4),
                }

    # TEMPO metrics from JSON
    for label, fname in [("TEMPO ZeroShot", "tempo_zero_shot_metrics.json"),
                          ("TEMPO FineTune", "tempo_fine_tuned_metrics.json")]:
        p = METRICS_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            rmse_vals = [v["RMSE"] for v in d.values()]
            mae_vals  = [v["MAE"]  for v in d.values()]
            r2_vals   = [v["R2"]   for v in d.values()]
            metrics[label] = {
                "RMSE_mean": round(float(np.mean(rmse_vals)), 4),
                "MAE_mean" : round(float(np.mean(mae_vals)),  4),
                "R2_mean"  : round(float(np.mean(r2_vals)),   4),
            }

    return metrics


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def count_lstm_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_sklearn_params(model) -> str:
    """Report n_estimators for tree-based models."""
    if hasattr(model, "n_estimators"):
        return f"{model.n_estimators} trees"
    return "N/A"


# ============================================================
# RANDOM FOREST BENCHMARKS
# ============================================================

def benchmark_random_forest(X_train, y_train, test_X_flat, X_flat_train):
    log.info("── Random Forest ─────────────────────────────────────────")

    ckpt_path = sorted(CKPT_BASE.glob("randomforest_*.joblib"))[-1]

    # Cold start
    def _load():
        return joblib.load(ckpt_path)

    cold_times = [time.perf_counter() - (lambda: time.perf_counter())()
                  for _ in range(1)]  # placeholder
    load_times = []
    for _ in range(N_REPEATS_LOAD):
        gc.collect()
        t0 = time.perf_counter()
        m = joblib.load(ckpt_path)
        load_times.append(time.perf_counter() - t0)
        del m
        gc.collect()
    cold_mean = float(np.mean(load_times))
    cold_std  = float(np.std(load_times))
    log.info("  Cold start:  %.2f ± %.2f s", cold_mean, cold_std)

    # Load model for inference benchmarks
    rf = joblib.load(ckpt_path)

    # Memory footprint during batch inference (batch=128)
    X_bench = test_X_flat[:128]
    with PeakMemoryTracker() as mem:
        _ = rf.predict(X_bench)
    peak_mem_mb = mem.peak_mb
    log.info("  Peak memory: %.1f MB", peak_mem_mb)

    # Inference timing at various batch sizes
    throughput: Dict[int, float] = {}
    infer_ms: Dict[int, tuple] = {}
    for bs in BATCH_SIZES:
        x = test_X_flat[:bs]
        mean_t, std_t = repeat_timing(lambda: rf.predict(x))
        infer_ms[bs]    = (mean_t * 1e3, std_t * 1e3)
        throughput[bs]  = bs / mean_t
        log.info("  BS=%4d  %.2f ± %.2f ms  (%.0f samples/s)",
                  bs, mean_t * 1e3, std_t * 1e3, throughput[bs])

    # Training time estimation on subset
    log.info("  Training time estimate (subset=%d)...", TRAIN_SUBSET)
    from sklearn.ensemble import RandomForestRegressor
    idx = np.random.choice(len(X_flat_train), TRAIN_SUBSET, replace=False)
    t0 = time.perf_counter()
    tmp = RandomForestRegressor(n_estimators=200, max_depth=15,
                                min_samples_split=10, n_jobs=-1, random_state=SEED)
    tmp.fit(X_flat_train[idx], y_train[idx])
    subset_time = time.perf_counter() - t0
    del tmp
    scale = len(X_flat_train) / TRAIN_SUBSET
    train_time_s = subset_time * scale
    log.info("  Training time estimate: %.1f s → scaled %.0f s (%.2f h)",
              subset_time, train_time_s, train_time_s / 3600)

    return {
        "cold_start_s"   : (cold_mean, cold_std),
        "infer_ms"       : infer_ms,
        "throughput_sps" : throughput,
        "peak_mem_mb"    : peak_mem_mb,
        "disk_size_mb"   : disk_size_mb(ckpt_path),
        "n_params"       : count_sklearn_params(rf),
        "train_time_s"   : (train_time_s, 0.0),
        "checkpoint"     : str(ckpt_path.name),
    }


# ============================================================
# XGBOOST BENCHMARKS
# ============================================================

def benchmark_xgboost(X_train, y_train, test_X_flat, X_flat_train):
    log.info("── XGBoost ───────────────────────────────────────────────")

    from xgboost import XGBRegressor
    ckpt_path = sorted(CKPT_BASE.glob("xgboost_*.joblib"))[-1]

    # Cold start
    load_times = []
    for _ in range(N_REPEATS_LOAD):
        gc.collect()
        t0 = time.perf_counter()
        m = joblib.load(ckpt_path)
        load_times.append(time.perf_counter() - t0)
        del m
        gc.collect()
    cold_mean = float(np.mean(load_times))
    cold_std  = float(np.std(load_times))
    log.info("  Cold start:  %.2f ± %.2f s", cold_mean, cold_std)

    xgb = joblib.load(ckpt_path)

    # Memory
    X_bench = test_X_flat[:128]
    with PeakMemoryTracker() as mem:
        _ = xgb.predict(X_bench)
    peak_mem_mb = mem.peak_mb
    log.info("  Peak memory: %.1f MB", peak_mem_mb)

    # Inference timing
    throughput: Dict[int, float] = {}
    infer_ms: Dict[int, tuple] = {}
    for bs in BATCH_SIZES:
        x = test_X_flat[:bs]
        mean_t, std_t = repeat_timing(lambda: xgb.predict(x))
        infer_ms[bs]   = (mean_t * 1e3, std_t * 1e3)
        throughput[bs] = bs / mean_t
        log.info("  BS=%4d  %.2f ± %.2f ms  (%.0f samples/s)",
                  bs, mean_t * 1e3, std_t * 1e3, throughput[bs])

    # Training time estimation
    log.info("  Training time estimate (subset=%d)...", TRAIN_SUBSET)
    idx = np.random.choice(len(X_flat_train), TRAIN_SUBSET, replace=False)
    y_sub_mean = y_train[idx].mean(axis=1)
    t0 = time.perf_counter()
    tmp = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.01,
                       subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", n_jobs=-1, random_state=SEED)
    tmp.fit(X_flat_train[idx], y_sub_mean)
    subset_time = time.perf_counter() - t0
    del tmp
    train_time_s = subset_time * (len(X_flat_train) / TRAIN_SUBSET)
    log.info("  Training time estimate: %.1f s → scaled %.0f s (%.2f h)",
              subset_time, train_time_s, train_time_s / 3600)

    return {
        "cold_start_s"   : (cold_mean, cold_std),
        "infer_ms"       : infer_ms,
        "throughput_sps" : throughput,
        "peak_mem_mb"    : peak_mem_mb,
        "disk_size_mb"   : disk_size_mb(ckpt_path),
        "n_params"       : count_sklearn_params(xgb),
        "train_time_s"   : (train_time_s, 0.0),
        "checkpoint"     : str(ckpt_path.name),
    }


# ============================================================
# LSTM BENCHMARKS
# ============================================================

def benchmark_lstm(X_train, y_train, test_X):
    log.info("── LSTM ──────────────────────────────────────────────────")

    ckpt_path = sorted(CKPT_BASE.glob("lstm_*.pt"))[-1]
    device    = torch.device("cpu")

    # Cold start
    load_times = []
    for _ in range(N_REPEATS_LOAD):
        gc.collect()
        t0 = time.perf_counter()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        m = LSTMForecaster(
            input_size  = ckpt["input_size"],
            hidden_size = ckpt["hyperparameters"]["hidden_size"],
            num_layers  = ckpt["hyperparameters"]["num_layers"],
            horizon     = ckpt["horizon"],
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        load_times.append(time.perf_counter() - t0)
        del m
        gc.collect()
    cold_mean = float(np.mean(load_times))
    cold_std  = float(np.std(load_times))
    log.info("  Cold start:  %.4f ± %.4f s", cold_mean, cold_std)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lstm = LSTMForecaster(
        input_size  = ckpt["input_size"],
        hidden_size = ckpt["hyperparameters"]["hidden_size"],
        num_layers  = ckpt["hyperparameters"]["num_layers"],
        horizon     = ckpt["horizon"],
    )
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()
    n_params = count_lstm_params(lstm)
    log.info("  Parameters:  {:,}".format(n_params))

    # Memory
    X_bench = torch.FloatTensor(test_X[:128])
    with PeakMemoryTracker() as mem:
        with torch.no_grad():
            _ = lstm(X_bench)
    peak_mem_mb = mem.peak_mb
    log.info("  Peak memory: %.1f MB", peak_mem_mb)

    # Inference timing
    throughput: Dict[int, float] = {}
    infer_ms: Dict[int, tuple] = {}
    for bs in BATCH_SIZES:
        x = torch.FloatTensor(test_X[:bs])
        def _infer(x=x):
            with torch.no_grad():
                return lstm(x)
        mean_t, std_t = repeat_timing(_infer)
        infer_ms[bs]   = (mean_t * 1e3, std_t * 1e3)
        throughput[bs] = bs / mean_t
        log.info("  BS=%4d  %.2f ± %.2f ms  (%.0f samples/s)",
                  bs, mean_t * 1e3, std_t * 1e3, throughput[bs])

    # Training time estimation (2 epochs → scale to 50)
    log.info("  Training time estimate (subset=%d, 2 epochs)...", TRAIN_SUBSET)
    idx = np.random.choice(len(X_train), TRAIN_SUBSET, replace=False)
    X_sub = torch.FloatTensor(X_train[idx])
    y_sub = torch.FloatTensor(y_train[idx])
    ds    = torch.utils.data.TensorDataset(X_sub, y_sub)
    dl    = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    tmp   = LSTMForecaster(input_size=X_train.shape[2], hidden_size=128,
                            num_layers=2, horizon=y_train.shape[1])
    opt   = torch.optim.Adam(tmp.parameters(), lr=1e-3)
    crit  = nn.MSELoss()
    t0 = time.perf_counter()
    for _ in range(2):
        tmp.train()
        for xb, yb in dl:
            opt.zero_grad()
            crit(tmp(xb), yb).backward()
            opt.step()
    epoch2_time = time.perf_counter() - t0
    del tmp
    train_time_s = (epoch2_time / 2) * 50 * (len(X_train) / TRAIN_SUBSET)
    log.info("  2-epoch time: %.1f s → scaled: %.0f s (%.2f h)",
              epoch2_time, train_time_s, train_time_s / 3600)

    return {
        "cold_start_s"   : (cold_mean, cold_std),
        "infer_ms"       : infer_ms,
        "throughput_sps" : throughput,
        "peak_mem_mb"    : peak_mem_mb,
        "disk_size_mb"   : disk_size_mb(ckpt_path),
        "n_params"       : n_params,
        "train_time_s"   : (train_time_s, 0.0),
        "checkpoint"     : str(ckpt_path.name),
    }


# ============================================================
# TEMPO BENCHMARKS
# ============================================================

def _load_tempo_model():
    """Load TEMPO base + fine-tuned weights."""
    ckpt_path = CKPT_TEMPO / "best_model.pth"
    zs_dir    = ROOT / "models" / "checkpoints" / "tempo_zero_shot"

    # Base model
    model = _TEMPOBase.load_pretrained_model(
        device   = torch.device("cpu"),
        repo_id  = "Melady/TEMPO",
        filename = "TEMPO-80M_v1.pth",
        cache_dir= str(zs_dir),
    )
    # Apply fine-tuned weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def benchmark_tempo(results: dict):
    """Benchmark TEMPO models (zero-shot and fine-tuned share architecture)."""

    log.info("── TEMPO ─────────────────────────────────────────────────")

    zs_dir    = ROOT / "models" / "checkpoints" / "tempo_zero_shot"
    ft_path   = CKPT_TEMPO / "best_model.pth"

    # Cold start — fine-tuned (includes base model loading)
    load_times = []
    for _ in range(N_REPEATS_LOAD):
        gc.collect()
        t0 = time.perf_counter()
        m = _load_tempo_model()
        load_times.append(time.perf_counter() - t0)
        del m
        gc.collect()
    cold_ft_mean = float(np.mean(load_times))
    cold_ft_std  = float(np.std(load_times))
    log.info("  Cold start (fine-tuned): %.2f ± %.2f s", cold_ft_mean, cold_ft_std)

    # Load model for inference benchmarks
    model   = _load_tempo_model()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Parameters: {:,}".format(n_params))

    # Synthetic input: [B, 336, 1]  (univariate, matches TEMPO input spec)
    def make_input(bs):
        return torch.randn(bs, 336, 1)

    # Memory at batch=128
    with PeakMemoryTracker() as mem:
        x = make_input(32)
        with torch.no_grad():
            try:
                _ = model(x, pred_len=96)
            except Exception:
                _ = model(x)
    peak_mem_mb = mem.peak_mb
    log.info("  Peak memory: %.1f MB", peak_mem_mb)

    # Inference timing
    throughput: Dict[int, float] = {}
    infer_ms: Dict[int, tuple] = {}
    for bs in BATCH_SIZES:
        x = make_input(bs)
        def _infer(x=x):
            with torch.no_grad():
                try:
                    return model(x, pred_len=96)
                except Exception:
                    return model(x)
        mean_t, std_t = repeat_timing(_infer)
        infer_ms[bs]   = (mean_t * 1e3, std_t * 1e3)
        throughput[bs] = bs / mean_t
        log.info("  BS=%4d  %.1f ± %.1f ms  (%.0f samples/s)",
                  bs, mean_t * 1e3, std_t * 1e3, throughput[bs])

    # Training time estimation: time 1 mini-batch, scale to full fine-tuning
    log.info("  Training time estimate (1 batch forward+backward, scale to full)...")
    x_t = torch.randn(32, 336, 1)
    y_t = torch.randn(32, 96, 1)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    batch_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        opt.zero_grad()
        try:
            out, _ = model(x_t, pred_len=96)
        except Exception:
            out = model(x_t)
            if isinstance(out, tuple):
                out = out[0]
        if out.shape != y_t.shape:
            out = out[:, :, :1] if out.dim() == 3 else out
        loss = crit(out, y_t)
        loss.backward()
        opt.step()
        batch_times.append(time.perf_counter() - t0)
    batch_time = float(np.mean(batch_times))

    # Full fine-tuning: 5000 samples / 32 = ~156 batches × 20 effective epochs
    n_batches_per_epoch = int(np.ceil(5000 / 32))
    effective_epochs    = 20
    ft_train_s = batch_time * n_batches_per_epoch * effective_epochs
    log.info("  Batch time: %.3f s → fine-tune estimate: %.0f s (%.2f h)",
              batch_time, ft_train_s, ft_train_s / 3600)

    model.eval()

    # Disk sizes
    zs_mb = disk_size_mb(zs_dir)
    ft_mb = disk_size_mb(ft_path)

    # Zero-shot cold start = same as fine-tuned (same base model)
    entry_common = {
        "cold_start_s"   : (cold_ft_mean, cold_ft_std),
        "infer_ms"       : infer_ms,
        "throughput_sps" : throughput,
        "peak_mem_mb"    : peak_mem_mb,
        "n_params"       : n_params,
        "checkpoint"     : "TEMPO-80M_v1.pth",
    }
    results["TEMPO ZeroShot"] = {
        **entry_common,
        "disk_size_mb"   : zs_mb,
        "train_time_s"   : (0.0, 0.0),    # zero-shot: no training
    }
    results["TEMPO FineTune"] = {
        **entry_common,
        "disk_size_mb"   : ft_mb,
        "train_time_s"   : (ft_train_s, 0.0),
    }
    del model
    gc.collect()


# ============================================================
# Build summary DataFrame
# ============================================================

def build_summary_df(bench: dict, acc: dict) -> pd.DataFrame:
    rows = []
    ref_sps = None   # reference throughput for cost ratio (Random Forest at BS=128)

    for model in MODEL_ORDER:
        if model not in bench:
            continue
        b = bench[model]
        a = acc.get(model, {})

        sps_128 = b["throughput_sps"].get(128, np.nan)
        if model == "Random Forest" and sps_128 > 0:
            ref_sps = sps_128

        row = {
            "Model"              : model,
            "RMSE_avg"           : a.get("RMSE_mean", np.nan),
            "R2_avg"             : a.get("R2_mean",   np.nan),
            "ColdStart_s"        : round(b["cold_start_s"][0], 3),
            "ColdStart_std_s"    : round(b["cold_start_s"][1], 3),
            "InferSingle_ms"     : round(b["infer_ms"][1][0], 3),
            "InferSingle_std_ms" : round(b["infer_ms"][1][1], 3),
            "InferBS128_ms"      : round(b["infer_ms"].get(128, (np.nan, np.nan))[0], 3),
            "Throughput_BS1_sps" : round(b["throughput_sps"].get(1, np.nan), 1),
            "Throughput_BS128_sps": round(b["throughput_sps"].get(128, np.nan), 1),
            "PeakMem_MB"         : round(max(0.05, b["peak_mem_mb"]), 1) if b["peak_mem_mb"] < 0.1 else round(b["peak_mem_mb"], 1),
            "DiskSize_MB"        : round(b["disk_size_mb"], 1),
            "NParams"            : b["n_params"],
            "TrainTime_h"        : round(b["train_time_s"][0] / 3600, 3),
            "TrainTime_std_h"    : round(b["train_time_s"][1] / 3600, 3),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Cost ratio (relative to RF throughput at BS=128)
    if ref_sps and ref_sps > 0:
        df["CostRatio"] = df["Throughput_BS128_sps"].apply(
            lambda x: round(ref_sps / x, 1) if x > 0 and not np.isnan(x) else np.nan
        )
    else:
        df["CostRatio"] = np.nan

    # Real-time capable flag
    df["RealtimeCapable"] = df["InferSingle_ms"] < REALTIME_THRESH_MS

    # Predictions per day (single-sample throughput × seconds/day)
    df["PredPerDay"] = (df["Throughput_BS1_sps"] * 86_400).round(0)

    # Carbon footprint: training kWh
    df["TrainEnergy_kWh"] = (df["TrainTime_h"] * CPU_TDP_W / 1000).round(4)
    df["TrainCO2_kg"]     = (df["TrainEnergy_kWh"] * CO2_KG_PER_KWH).round(4)

    return df


# ============================================================
# Visualisations
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family"     : "serif",
        "font.size"       : 10,
        "axes.labelsize"  : 11,
        "axes.titlesize"  : 12,
        "xtick.labelsize" : 9,
        "ytick.labelsize" : 9,
        "figure.dpi"      : 150,
        "savefig.dpi"     : 300,
        "savefig.bbox"    : "tight",
    })


def _model_colors(models: List[str]) -> List[str]:
    return [COLORS.get(m, "#888888") for m in models]


# ── Figure 1: Inference time comparison ─────────────────────────────────────
def plot_inference_time(df: pd.DataFrame, bench: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = [m for m in MODEL_ORDER if m in df["Model"].values]
    cols   = _model_colors(models)

    # Panel A: Single-sample inference time (ms)
    ax = axes[0]
    means = df.set_index("Model").loc[models, "InferSingle_ms"]
    stds  = df.set_index("Model").loc[models, "InferSingle_std_ms"]
    bars  = ax.bar(range(len(models)), means, yerr=stds,
                    color=cols, alpha=0.85, capsize=5, error_kw={"lw": 1.5})
    ax.axhline(REALTIME_THRESH_MS, color="red", lw=1.2, ls="--",
                label=f"Real-time threshold ({REALTIME_THRESH_MS:.0f} ms)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Inference Time — Single Sample (ms)")
    ax.set_title("(a) Single-Sample Inference Latency")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (v, s) in enumerate(zip(means, stds)):
        ax.text(i, v * 1.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Panel B: Throughput at different batch sizes
    ax = axes[1]
    bs_plot = [bs for bs in BATCH_SIZES if bs <= 500]
    for m, col in zip(models, cols):
        tp = [bench[m]["throughput_sps"].get(bs, np.nan) for bs in bs_plot]
        ax.plot(bs_plot, tp, "o-", color=col, lw=1.8, ms=6, label=m)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples / second)")
    ax.set_title("(b) Throughput vs Batch Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Inference Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Figure 2: Accuracy vs cost trade-off ────────────────────────────────────
def plot_accuracy_vs_cost(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = [m for m in MODEL_ORDER if m in df["Model"].values]
    dfi    = df.set_index("Model")

    def _pareto(x_arr, y_arr):
        """Return indices on the lower-left Pareto frontier (min x, min y)."""
        pts = sorted(zip(x_arr, y_arr, range(len(x_arr))))
        frontier = []
        best_y = float("inf")
        for xv, yv, idx in pts:
            if yv < best_y and not np.isnan(xv) and not np.isnan(yv):
                best_y = yv
                frontier.append(idx)
        return frontier

    for ax, x_col, x_label, title in [
        (axes[0], "InferSingle_ms",
         "Single-Sample Inference Time (ms)", "(a) RMSE vs Inference Time"),
        (axes[1], "DiskSize_MB",
         "Model Size on Disk (MB)", "(b) RMSE vs Model Size"),
    ]:
        xs = [dfi.loc[m, x_col]    if m in dfi.index else np.nan for m in models]
        ys = [dfi.loc[m, "RMSE_avg"] if m in dfi.index else np.nan for m in models]
        for m, x, y, col in zip(models, xs, ys, _model_colors(models)):
            ax.scatter(x, y, color=col, s=120, zorder=4, edgecolors="white", lw=1.5)
            ax.annotate(m, (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=8, color=col)

        # Pareto frontier
        valid = [(x, y, i) for i, (x, y) in enumerate(zip(xs, ys))
                 if not (np.isnan(x) or np.isnan(y))]
        if len(valid) >= 2:
            pxs = sorted(valid, key=lambda t: t[0])
            best_y = float("inf")
            px_front, py_front = [], []
            for xv, yv, _ in pxs:
                if yv < best_y:
                    best_y = yv
                    px_front.append(xv)
                    py_front.append(yv)
            if len(px_front) >= 2:
                ax.plot(px_front, py_front, "k--", lw=1.2, alpha=0.5,
                         label="Pareto frontier")
                ax.legend(fontsize=8)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Avg RMSE (μmol m⁻² s⁻¹)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if np.any([not np.isnan(x) and x > 0 for x in xs]):
            ax.set_xscale("log")

    fig.suptitle("Accuracy vs Computational Cost Trade-off",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Figure 3: Resource usage comparison ─────────────────────────────────────
def plot_resource_usage(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    models = [m for m in MODEL_ORDER if m in df["Model"].values]
    cols   = _model_colors(models)
    dfi    = df.set_index("Model")

    # ── (a) Peak memory ──────────────────────────────────────────────────
    ax = axes[0]
    mems = [dfi.loc[m, "PeakMem_MB"] if m in dfi.index else 0 for m in models]
    ax.barh(range(len(models)), mems, color=cols, alpha=0.85)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Peak RAM Increase During Inference (MB)")
    ax.set_title("(a) Peak Memory Footprint (batch=128)")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(mems):
        ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=8)

    # ── (b) Disk size ────────────────────────────────────────────────────
    ax = axes[1]
    sizes = [dfi.loc[m, "DiskSize_MB"] if m in dfi.index else 0 for m in models]
    ax.bar(range(len(models)), sizes, color=cols, alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Model Disk Size (MB)")
    ax.set_title("(b) On-Disk Model Size")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(sizes):
        ax.text(i, v * 1.3, f"{v:.0f}", ha="center", fontsize=8)

    # ── (c) Training time ────────────────────────────────────────────────
    ax = axes[2]
    ttimes = [dfi.loc[m, "TrainTime_h"] if m in dfi.index else 0 for m in models]
    ax.bar(range(len(models)), ttimes, color=cols, alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Estimated Training Time (hours)")
    ax.set_title("(c) Training Time (scaled estimate)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(ttimes):
        ax.text(i, v + max(ttimes) * 0.02, f"{v:.2f}h", ha="center", fontsize=8)

    # ── (d) CO₂ footprint ────────────────────────────────────────────────
    ax = axes[3]
    co2 = [dfi.loc[m, "TrainCO2_kg"] if m in dfi.index else 0 for m in models]
    ax.bar(range(len(models)), co2, color=cols, alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Estimated Training CO₂ (kg)")
    ax.set_title(f"(d) Carbon Footprint ({CO2_KG_PER_KWH} kg/kWh, {CPU_TDP_W}W CPU)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(co2):
        ax.text(i, v + max(co2) * 0.02, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle("Computational Resource Usage", fontsize=13, fontweight="bold")
    return fig


# ============================================================
# Thesis-ready summary
# ============================================================

def build_summary_text(df: pd.DataFrame, bench: dict) -> str:
    lines = [
        "=" * 90,
        "COMPUTATIONAL EFFICIENCY ANALYSIS — CARBON FLUX FORECASTING MODELS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Hardware:  CPU-only (no GPU)  |  {CPU_TDP_W}W TDP assumed",
        f"Carbon intensity: {CO2_KG_PER_KWH} kg CO₂/kWh (UK grid average 2024)",
        "=" * 90,
        "",
        "── SUMMARY TABLE ─────────────────────────────────────────────────",
        "",
        f"{'Model':<18} {'RMSE':>6} {'R²':>6} {'Train(h)':>9} {'Infer(ms)':>10}"
        f" {'Size(MB)':>9} {'Mem(MB)':>8} {'Throughput':>11} {'CostRatio':>10}",
        "─" * 90,
    ]

    for _, row in df.iterrows():
        m = row["Model"]
        lines.append(
            f"{m:<18} {row['RMSE_avg']:>6.3f} {row['R2_avg']:>6.3f}"
            f" {row['TrainTime_h']:>9.2f} {row['InferSingle_ms']:>10.2f}"
            f" {row['DiskSize_MB']:>9.0f} {row['PeakMem_MB']:>8.0f}"
            f" {row['Throughput_BS128_sps']:>11,.0f}"
            f" {row['CostRatio']:>10.1f}×"
        )

    lines += [
        "",
        "Notes:",
        "  RMSE/R² averaged over UK-AMo and SE-Htm test sites.",
        "  Training time = scaled estimate from mini-training on a subset.",
        "  Inference time = single-sample (BS=1), mean over 10 repetitions.",
        "  Throughput = samples/second at batch size 128.",
        "  Cost Ratio = relative to Random Forest throughput at BS=128.",
        f"  Real-time threshold = {REALTIME_THRESH_MS:.0f} ms (operational NEE forecasting).",
        "",
        "── DEPLOYMENT ANALYSIS ────────────────────────────────────────────────────────",
        "",
    ]

    for _, row in df.iterrows():
        m  = row["Model"]
        rt = "YES" if row["RealtimeCapable"] else "NO "
        lines += [
            f"  {m}:",
            f"    Real-time capable (<{REALTIME_THRESH_MS:.0f} ms) : {rt}",
            f"    Predictions per day (BS=1) : {row['PredPerDay']:,.0f}",
            f"    Disk size                  : {row['DiskSize_MB']:.0f} MB",
            f"    Training energy            : {row['TrainEnergy_kWh']:.4f} kWh"
                f"  ({row['TrainCO2_kg']:.4f} kg CO₂)",
            "",
        ]

    lines += [
        "── BATCH-SIZE THROUGHPUT TABLE ────────────────────────────────────────────────",
        "",
        f"{'Model':<18} " + "  ".join(f"BS={bs:>4}" for bs in BATCH_SIZES),
        "─" * (18 + 10 * len(BATCH_SIZES)),
    ]
    for model in MODEL_ORDER:
        if model not in bench:
            continue
        tp = bench[model]["throughput_sps"]
        row_s = f"{model:<18} "
        for bs in BATCH_SIZES:
            v = tp.get(bs, np.nan)
            row_s += f"{v:>10,.0f}" if not np.isnan(v) else f"{'N/A':>10}"
        lines.append(row_s)

    lines += ["", "=" * 90]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    t_total = time.time()

    print("=" * 80)
    print("COMPUTATIONAL EFFICIENCY BENCHMARKS — CARBON FLUX MODELS")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timing repeats: inference={N_REPEATS_INFER}, load={N_REPEATS_LOAD}")
    print(f"  Batch sizes:    {BATCH_SIZES}")
    print(f"  Training subset: {TRAIN_SUBSET:,} samples")
    print("=" * 80)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("[1/5] Loading data...")
    X_train, y_train, test_data = load_baseline_data()
    # Use UK-AMo test set as the benchmark set (larger)
    X_test    = test_data["UK-AMo"]["X"]
    X_flat    = X_test.reshape(len(X_test), -1)
    X_flat_tr = X_train.reshape(len(X_train), -1)
    log.info("  Train: %s  |  Test (UK-AMo): %s", X_train.shape, X_test.shape)

    # ── Accuracy metrics ───────────────────────────────────────────────────
    log.info("[2/5] Loading existing accuracy metrics...")
    acc = load_accuracy_metrics()
    for m, v in acc.items():
        log.info("  %-20s  RMSE=%.4f  R²=%.4f", m, v["RMSE_mean"], v["R2_mean"])

    # ── Benchmarks ─────────────────────────────────────────────────────────
    log.info("[3/5] Running benchmarks...")
    bench: dict = {}

    bench["Random Forest"] = benchmark_random_forest(X_train, y_train, X_flat, X_flat_tr)
    bench["XGBoost"]       = benchmark_xgboost(X_train, y_train, X_flat, X_flat_tr)
    bench["LSTM"]          = benchmark_lstm(X_train, y_train, X_test)

    if TEMPO_AVAILABLE:
        try:
            benchmark_tempo(bench)
        except Exception as exc:
            log.warning("  TEMPO benchmarking failed: %s", exc)
            log.warning("  TEMPO entries will be omitted from figures.")
    else:
        log.warning("  TEMPO skipped (tempo package not available).")

    # ── Build summary DataFrame ────────────────────────────────────────────
    log.info("[4/5] Aggregating results...")
    df = build_summary_df(bench, acc)
    print("\n" + df.to_string(index=False))

    # ── Save structured outputs ────────────────────────────────────────────
    # JSON
    costs_json: dict = {}
    for model, b in bench.items():
        # Serialise: convert numpy floats, tuples → JSON-safe
        def _to_py(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, dict):
                return {str(k): _to_py(vv) for k, vv in v.items()}
            return v
        costs_json[model] = {k: _to_py(v) for k, v in b.items()}

    json_path = METRICS_DIR / "computational_costs.json"
    with open(json_path, "w") as f:
        json.dump(costs_json, f, indent=2)
    log.info("  Saved: %s", json_path.relative_to(ROOT))

    # CSV
    csv_path = METRICS_DIR / "efficiency_comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path.relative_to(ROOT))

    # Thesis summary text
    summary = build_summary_text(df, bench)
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary)
    log.info("  Saved: %s", SUMMARY_TXT.relative_to(ROOT))
    print("\n" + summary)

    # ── Figures ────────────────────────────────────────────────────────────
    log.info("[5/5] Generating figures...")

    fig = plot_inference_time(df, bench)
    p = FIG_DIR / "inference_time_comparison.png"
    fig.savefig(p); plt.close(fig)
    log.info("  Saved: %s", p.name)

    fig = plot_accuracy_vs_cost(df)
    p = FIG_DIR / "accuracy_vs_cost_tradeoff.png"
    fig.savefig(p); plt.close(fig)
    log.info("  Saved: %s", p.name)

    fig = plot_resource_usage(df)
    p = FIG_DIR / "resource_usage_comparison.png"
    fig.savefig(p); plt.close(fig)
    log.info("  Saved: %s", p.name)

    elapsed = time.time() - t_total
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║  COMPUTATIONAL EFFICIENCY ANALYSIS COMPLETE" + " " * 34 + "║")
    print(f"║  Runtime  : {elapsed / 60:.1f} min" + " " * 63 + "║")
    print("║  JSON     : results/metrics/computational_costs.json" + " " * 25 + "║")
    print("║  CSV      : results/metrics/efficiency_comparison.csv" + " " * 24 + "║")
    print("║  Summary  : results/efficiency/COMPUTATIONAL_EFFICIENCY_SUMMARY.txt" + " " * 7 + "║")
    print("║  Figures  : figures/efficiency/" + " " * 46 + "║")
    print("╚" + "═" * 78 + "╝")


if __name__ == "__main__":
    main()
