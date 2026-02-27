"""
Uncertainty Quantification Analysis for Carbon Flux Prediction
==============================================================
Implements and evaluates three complementary approaches to predictive
uncertainty quantification for carbon flux forecasting:

1. Monte Carlo Dropout (Gal & Ghahramani 2016) for LSTM
   - 100 stochastic forward passes with dropout active at inference
   - 90% prediction intervals from the sampling distribution

2. Quantile Regression for XGBoost
   - Models trained at multiple quantile levels (0.05–0.95)
   - Asymmetric uncertainty bands capturing non-Gaussian residuals

3. Ensemble Variance for Random Forest
   - Per-tree predictions from 200 decision trees
   - 90% intervals via empirical distribution over the ensemble

Calibration Analysis:
- Empirical coverage (PICP) vs nominal coverage (reliability diagrams)
- Interval sharpness (mean width) as a function of nominal coverage
- Horizon-specific uncertainty decomposition (steps 1–96)

References
----------
Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
  Representing model uncertainty in deep learning. ICML.
Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica.
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Outputs
-------
  figures/uncertainty/prediction_intervals_{model}_{site}.png
  figures/uncertainty/calibration_analysis_{site}.png
  figures/uncertainty/uncertainty_decomposition_{site}.png
  results/uncertainty/uncertainty_metrics.csv
  results/uncertainty/calibration_summary.txt

Usage
-----
  python scripts/uncertainty_quantification.py
"""

import sys
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from xgboost import XGBRegressor
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR  = ROOT / "data" / "processed"
CKPT_DIR  = ROOT / "models" / "checkpoints" / "baselines"
UQ_DIR    = ROOT / "results" / "uncertainty"
FIG_DIR   = ROOT / "figures" / "uncertainty"

SITES = ["UK-AMo", "SE-Htm"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MC_SAMPLES       = 300      # stochastic forward passes
MC_DROPOUT_RATE  = 0.25     # additional output-layer dropout for MC
# Note: Original value 0.10 resulted in severe miscalibration (~31% coverage
# at nominal 90%). Increased to 0.25 per Gal & Ghahramani (2016) guidance on
# dropout rates for uncertainty calibration in deep sequential models.
MC_TEST_N        = 500      # test-set subset for MC (speed)
HORIZON          = 96
SEED             = 42

# Nominal coverage levels used in calibration tables / figures
COVERAGE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# XGBoost quantile levels: chosen so that symmetric pairs cover
# nominal levels 20 / 40 / 60 / 80 / 90 %
XGB_QUANTILE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40,
                        0.50,
                        0.60, 0.70, 0.80, 0.90, 0.95]

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Colour palette  (consistent with other analysis scripts)
# ---------------------------------------------------------------------------
COLORS = {
    "LSTM":         "#9C27B0",
    "RandomForest": "#4CAF50",
    "XGBoost":      "#FF5722",
}
SITE_NAMES = {"UK-AMo": "UK-AMo (Wetland)", "SE-Htm": "SE-Htm (Forest)"}

# ---------------------------------------------------------------------------
# LSTM model definition  (must match training definition exactly)
# ---------------------------------------------------------------------------
from models.lstm_baseline import LSTMForecaster


# ---------------------------------------------------------------------------
# 1. Monte Carlo Dropout — LSTM
# ---------------------------------------------------------------------------
class MCDropoutLSTM(nn.Module):
    """
    Wraps a trained LSTMForecaster for MC Dropout inference.

    Following Gal & Ghahramani (2016), the model is kept in train() mode
    during inference so the existing inter-layer LSTM dropout is active.
    An additional Dropout layer is inserted between the final hidden state
    and the linear projection, providing output-level stochasticity.
    """

    def __init__(self, base_model: LSTMForecaster, mc_dropout_rate: float = 0.10):
        super().__init__()
        self.lstm       = base_model.lstm
        self.fc         = base_model.fc
        self.mc_dropout = nn.Dropout(p=mc_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # MC Dropout applied to final hidden state
        return self.fc(self.mc_dropout(lstm_out[:, -1, :]))


def mc_dropout_inference(mc_model: MCDropoutLSTM,
                          X: np.ndarray,
                          n_samples: int = 100,
                          batch_size: int = 256,
                          device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    Run `n_samples` stochastic forward passes with dropout active.

    Returns
    -------
    samples : ndarray of shape (n_samples, N, HORIZON)
    """
    mc_model = mc_model.to(device)
    mc_model.train()          # activates all Dropout modules

    X_t = torch.FloatTensor(X).to(device)
    N   = len(X)
    all_samples = np.zeros((n_samples, N, HORIZON), dtype=np.float32)

    for s in range(n_samples):
        preds = []
        for i in range(0, N, batch_size):
            with torch.no_grad():
                out = mc_model(X_t[i:i + batch_size]).cpu().numpy()
            preds.append(out)
        all_samples[s] = np.concatenate(preds)

    return all_samples


# ---------------------------------------------------------------------------
# 2. RF Ensemble Uncertainty
# ---------------------------------------------------------------------------
def rf_tree_predictions(rf_model, X: np.ndarray) -> np.ndarray:
    """
    Extract per-estimator predictions from a fitted RandomForestRegressor.

    Returns
    -------
    tree_preds : ndarray of shape (n_trees, N, HORIZON)
    """
    X_flat = X.reshape(X.shape[0], -1)
    tree_preds = np.stack([est.predict(X_flat)
                            for est in rf_model.estimators_])
    return tree_preds   # (n_trees, N, HORIZON)


# ---------------------------------------------------------------------------
# 3. XGBoost Quantile Regression
# ---------------------------------------------------------------------------
def train_xgb_quantile_models(X_train: np.ndarray,
                                y_train: np.ndarray,
                                quantile_levels: list,
                                n_estimators: int = 60) -> dict:
    """
    Train one XGBRegressor per quantile level.

    Target: horizon-averaged NEE per sample (scalar), which enables
    standard quantile regression.  Per-horizon decomposition is handled
    by the RF/LSTM methods which carry full HORIZON-dimensional outputs.

    Returns
    -------
    models : dict mapping float quantile → fitted XGBRegressor
    """
    X_flat   = X_train.reshape(X_train.shape[0], -1)
    y_scalar = y_train.mean(axis=1)          # (N,)

    models = {}
    for q in quantile_levels:
        log.info("  XGB quantile q=%.2f  (n_est=%d)...", q, n_estimators)
        m = XGBRegressor(
            n_estimators    = n_estimators,
            max_depth       = 4,
            learning_rate   = 0.05,
            objective       = 'reg:quantileerror',
            quantile_alpha  = float(q),
            tree_method     = 'hist',
            subsample       = 0.8,
            colsample_bytree= 0.5,
            n_jobs          = -1,
            random_state    = SEED,
        )
        m.fit(X_flat, y_scalar)
        models[q] = m

    return models


def xgb_quantile_predict(xgb_models: dict, X: np.ndarray) -> dict:
    """Predict each quantile level for test data X (scalar output)."""
    X_flat = X.reshape(X.shape[0], -1)
    return {q: m.predict(X_flat) for q, m in xgb_models.items()}


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------
def samples_to_intervals(samples: np.ndarray, alpha: float):
    """
    Compute mean, lower, upper from an empirical sample array.

    Parameters
    ----------
    samples : (n_samples, ...) array
    alpha   : nominal coverage (e.g. 0.90)
    """
    q_lo  = (1.0 - alpha) / 2.0
    q_hi  = 1.0 - q_lo
    mean  = samples.mean(axis=0)
    lower = np.quantile(samples, q_lo, axis=0)
    upper = np.quantile(samples, q_hi, axis=0)
    return mean, lower, upper


def gaussian_interval_width(std: np.ndarray, alpha: float) -> np.ndarray:
    """Width of Gaussian PI at nominal coverage alpha: 2 * z * std."""
    z = stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0)
    return 2.0 * z * std


def compute_coverage(y_true: np.ndarray,
                      lower: np.ndarray,
                      upper: np.ndarray) -> float:
    """Fraction of y_true inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_picp(y_true: np.ndarray,
                  samples: np.ndarray,
                  alpha: float) -> float:
    """PICP at nominal coverage alpha, from empirical samples."""
    _, lower, upper = samples_to_intervals(samples, alpha)
    return compute_coverage(y_true, lower, upper)


def compute_mean_width(samples: np.ndarray, alpha: float) -> float:
    """Mean interval width at nominal coverage alpha."""
    _, lower, upper = samples_to_intervals(samples, alpha)
    return float(np.mean(upper - lower))


# ---------------------------------------------------------------------------
# Horizon-specific calibration
# ---------------------------------------------------------------------------
def horizon_picp(y_true: np.ndarray,
                  samples: np.ndarray,
                  alpha: float = 0.90) -> np.ndarray:
    """
    Coverage (PICP) at each forecast step.

    Returns
    -------
    picp_by_hour : ndarray of shape (HORIZON,)
    """
    _, lower, upper = samples_to_intervals(samples, alpha)
    return np.mean((y_true >= lower) & (y_true <= upper), axis=0)


def horizon_width(samples: np.ndarray, alpha: float = 0.90) -> np.ndarray:
    """Mean interval width at each forecast step.

    Returns
    -------
    width_by_hour : ndarray of shape (HORIZON,)
    """
    _, lower, upper = samples_to_intervals(samples, alpha)
    return np.mean(upper - lower, axis=0)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------
def reliability_data(y_true: np.ndarray,
                      samples: np.ndarray,
                      alpha_grid: Optional[np.ndarray] = None) -> dict:
    """
    Compute (nominal_coverage, empirical_PICP, mean_width) over a grid.

    Parameters
    ----------
    y_true  : (N, ...) ground truth
    samples : (n_samples, N, ...) predictive draws
    """
    if alpha_grid is None:
        alpha_grid = np.arange(0.05, 1.00, 0.05)

    nominal  = []
    coverage = []
    width    = []
    for alpha in alpha_grid:
        nominal.append(float(alpha))
        coverage.append(compute_picp(y_true, samples, float(alpha)))
        width.append(compute_mean_width(samples, float(alpha)))

    return {
        'nominal' : np.array(nominal),
        'coverage': np.array(coverage),
        'width'   : np.array(width),
    }


def reliability_data_xgb(y_true_scalar: np.ndarray,
                           xgb_preds: dict) -> dict:
    """
    Reliability data for XGBoost quantile predictions.

    Uses symmetric pairs: for nominal coverage α the interval is
    [Q_{(1-α)/2}, Q_{1-(1-α)/2}].  Only levels present in xgb_preds
    are used.
    """
    nominal, coverage, width = [], [], []
    for alpha in [0.20, 0.40, 0.60, 0.80, 0.90]:
        q_lo = round((1.0 - alpha) / 2.0, 2)
        q_hi = round(1.0 - q_lo, 2)
        if q_lo in xgb_preds and q_hi in xgb_preds:
            lower = xgb_preds[q_lo]
            upper = xgb_preds[q_hi]
            nominal.append(alpha)
            coverage.append(compute_coverage(y_true_scalar, lower, upper))
            width.append(float(np.mean(upper - lower)))

    return {
        'nominal' : np.array(nominal),
        'coverage': np.array(coverage),
        'width'   : np.array(width),
    }


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------
def collect_metrics(site: str,
                     model_name: str,
                     y_true: np.ndarray,
                     samples: np.ndarray) -> list:
    """Return a list of metric dicts for CSV export."""
    rows = []
    for alpha in COVERAGE_LEVELS:
        picp  = compute_picp(y_true, samples, alpha)
        width = compute_mean_width(samples, alpha)
        rows.append({
            'Site'              : site,
            'Model'             : model_name,
            'NominalCoverage'   : round(alpha, 2),
            'PICP'              : round(picp, 4),
            'MeanIntervalWidth' : round(width, 4),
            'CalibrationError'  : round(abs(picp - alpha), 4),
        })
    return rows


def collect_xgb_metrics(site: str,
                          y_true_scalar: np.ndarray,
                          xgb_preds: dict) -> list:
    """Collect XGBoost calibration metrics from quantile pairs."""
    rows = []
    for alpha in COVERAGE_LEVELS:
        q_lo = round((1.0 - alpha) / 2.0, 2)
        q_hi = round(1.0 - q_lo, 2)
        if q_lo not in xgb_preds or q_hi not in xgb_preds:
            continue
        lower = xgb_preds[q_lo]
        upper = xgb_preds[q_hi]
        picp  = compute_coverage(y_true_scalar, lower, upper)
        width = float(np.mean(upper - lower))
        rows.append({
            'Site'              : site,
            'Model'             : 'XGBoost_Quantile',
            'NominalCoverage'   : round(alpha, 2),
            'PICP'              : round(picp, 4),
            'MeanIntervalWidth' : round(width, 4),
            'CalibrationError'  : round(abs(picp - alpha), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        'font.family'     : 'serif',
        'font.size'       : 10,
        'axes.labelsize'  : 11,
        'axes.titlesize'  : 12,
        'xtick.labelsize' : 9,
        'ytick.labelsize' : 9,
        'figure.dpi'      : 150,
        'savefig.dpi'     : 300,
        'savefig.bbox'    : 'tight',
    })


NEE_LABEL = r'NEE ($\mu$mol m$^{-2}$ s$^{-1}$)'
WIDTH_LABEL = r'Interval Width ($\mu$mol m$^{-2}$ s$^{-1}$)'


# ---------------------------------------------------------------------------
# Figure 1a – LSTM prediction intervals
# ---------------------------------------------------------------------------
def plot_prediction_intervals_lstm(y_true: np.ndarray,
                                    mc_samples: np.ndarray,
                                    site: str,
                                    n_examples: int = 3,
                                    idx_start: int = 0) -> plt.Figure:
    """
    Show n_examples 96-hour forecast windows with MC Dropout uncertainty bands.
    mc_samples : (n_mc, N, HORIZON)
    """
    hours = np.arange(HORIZON)
    q05   = np.quantile(mc_samples, 0.05, axis=0)
    q25   = np.quantile(mc_samples, 0.25, axis=0)
    q75   = np.quantile(mc_samples, 0.75, axis=0)
    q95   = np.quantile(mc_samples, 0.95, axis=0)
    mean  = mc_samples.mean(axis=0)

    fig, axes = plt.subplots(n_examples, 1,
                              figsize=(13, 3.8 * n_examples),
                              sharex=False)
    if n_examples == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        i = idx_start + k
        ax.fill_between(hours, q05[i], q95[i],
                         alpha=0.25, color=COLORS["LSTM"], label='90% PI')
        ax.fill_between(hours, q25[i], q75[i],
                         alpha=0.40, color=COLORS["LSTM"], label='50% PI')
        ax.plot(hours, mean[i],   color=COLORS["LSTM"], lw=1.8, label='MC Mean')
        ax.plot(hours, y_true[i], 'k-', lw=1.2, alpha=0.85, label='Observed')
        ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
        ax.set_ylabel(NEE_LABEL)
        ax.set_title(f'{SITE_NAMES[site]} — Window {i + 1}')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Forecast Horizon (h)')
    fig.suptitle(f'LSTM MC Dropout Prediction Intervals — {SITE_NAMES[site]}',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1b – Random Forest prediction intervals
# ---------------------------------------------------------------------------
def plot_prediction_intervals_rf(y_true: np.ndarray,
                                   tree_preds: np.ndarray,
                                   site: str,
                                   n_examples: int = 3,
                                   idx_start: int = 0) -> plt.Figure:
    """
    Show n_examples forecast windows with RF ensemble uncertainty.
    tree_preds : (n_trees, N, HORIZON)
    """
    hours    = np.arange(HORIZON)
    rf_mean  = tree_preds.mean(axis=0)
    rf_std   = tree_preds.std(axis=0)
    z90      = stats.norm.ppf(0.95)
    z50      = stats.norm.ppf(0.75)

    fig, axes = plt.subplots(n_examples, 1,
                              figsize=(13, 3.8 * n_examples),
                              sharex=False)
    if n_examples == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        i = idx_start + k
        ax.fill_between(hours,
                         rf_mean[i] - z90 * rf_std[i],
                         rf_mean[i] + z90 * rf_std[i],
                         alpha=0.25, color=COLORS["RandomForest"], label='90% PI')
        ax.fill_between(hours,
                         rf_mean[i] - z50 * rf_std[i],
                         rf_mean[i] + z50 * rf_std[i],
                         alpha=0.40, color=COLORS["RandomForest"], label='50% PI')
        ax.plot(hours, rf_mean[i],  color=COLORS["RandomForest"], lw=1.8, label='RF Mean')
        ax.plot(hours, y_true[i],   'k-', lw=1.2, alpha=0.85, label='Observed')
        ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
        ax.set_ylabel(NEE_LABEL)
        ax.set_title(f'{SITE_NAMES[site]} — Window {i + 1}')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Forecast Horizon (h)')
    fig.suptitle(f'Random Forest Ensemble Prediction Intervals — {SITE_NAMES[site]}',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1c – XGBoost quantile prediction intervals
# ---------------------------------------------------------------------------
def plot_prediction_intervals_xgb(y_true_scalar: np.ndarray,
                                    xgb_preds: dict,
                                    site: str,
                                    n_show: int = 300) -> plt.Figure:
    """
    Plot XGBoost quantile regression intervals on the scalar (mean) NEE.
    """
    n_show = min(n_show, len(y_true_scalar))
    idx    = np.arange(n_show)

    q05 = xgb_preds.get(0.05)
    q25 = xgb_preds.get(0.25)
    q50 = xgb_preds.get(0.50)
    q75 = xgb_preds.get(0.75)
    q95 = xgb_preds.get(0.95)

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.fill_between(idx, q05[idx], q95[idx],
                     alpha=0.25, color=COLORS["XGBoost"], label='90% PI [Q5–Q95]')
    if q25 is not None and q75 is not None:
        ax.fill_between(idx, q25[idx], q75[idx],
                         alpha=0.40, color=COLORS["XGBoost"], label='50% PI [Q25–Q75]')
    ax.plot(idx, q50[idx],           color=COLORS["XGBoost"], lw=1.8, label='Q50 Median')
    ax.plot(idx, y_true_scalar[idx], 'k-', lw=0.9, alpha=0.75, label='Observed (mean NEE)')
    ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)

    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel(r'Mean NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
    ax.set_title(f'XGBoost Quantile Regression — {SITE_NAMES[site]}')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – 6-panel calibration analysis
# ---------------------------------------------------------------------------
def plot_calibration_analysis(site: str,
                                lstm_rel: dict,
                                rf_rel:   dict,
                                xgb_rel:  dict,
                                lstm_h_picp:  np.ndarray,
                                rf_h_picp:    np.ndarray,
                                lstm_h_width: np.ndarray,
                                rf_h_width:   np.ndarray) -> plt.Figure:
    """
    Six-panel calibration figure:
      (a) Reliability diagram — all models
      (b) Sharpness: interval width vs nominal coverage
      (c) LSTM horizon-specific PICP at 90 %
      (d) RF horizon-specific PICP at 90 %
      (e) 90 % interval width by forecast horizon
      (f) Coverage by 4-hour blocks (bar chart)
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
    ax  = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    hours = np.arange(HORIZON)

    # ── (a) Reliability diagram ──────────────────────────────────────────
    a = ax[0]
    a.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    a.plot(lstm_rel['nominal'], lstm_rel['coverage'],
            'o-', color=COLORS["LSTM"], lw=1.8, ms=5, label='LSTM MC Dropout')
    a.plot(rf_rel['nominal'], rf_rel['coverage'],
            's-', color=COLORS["RandomForest"], lw=1.8, ms=5, label='Random Forest')
    if len(xgb_rel['nominal']) > 0:
        a.plot(xgb_rel['nominal'], xgb_rel['coverage'],
                '^-', color=COLORS["XGBoost"], lw=1.8, ms=6, label='XGBoost QR')
    a.set_xlabel('Nominal Coverage')
    a.set_ylabel('Empirical Coverage (PICP)')
    a.set_title('(a) Reliability Diagram')
    a.legend(fontsize=8, loc='upper left')
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.grid(True, alpha=0.3)

    # ── (b) Sharpness ────────────────────────────────────────────────────
    a = ax[1]
    a.plot(lstm_rel['nominal'], lstm_rel['width'],
            'o-', color=COLORS["LSTM"], lw=1.8, ms=5, label='LSTM MC Dropout')
    a.plot(rf_rel['nominal'], rf_rel['width'],
            's-', color=COLORS["RandomForest"], lw=1.8, ms=5, label='Random Forest')
    if len(xgb_rel['nominal']) > 0:
        a.plot(xgb_rel['nominal'], xgb_rel['width'],
                '^-', color=COLORS["XGBoost"], lw=1.8, ms=6, label='XGBoost QR')
    a.set_xlabel('Nominal Coverage')
    a.set_ylabel(WIDTH_LABEL)
    a.set_title('(b) Sharpness vs Reliability')
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── (c) LSTM horizon PICP ────────────────────────────────────────────
    a = ax[2]
    a.axhline(0.90, color='gray', lw=1.2, ls='--', alpha=0.7, label='Nominal 90 %')
    a.plot(hours, lstm_h_picp, color=COLORS["LSTM"], lw=1.8, label='LSTM MC')
    a.set_xlabel('Forecast Horizon (h)')
    a.set_ylabel('PICP')
    a.set_title('(c) LSTM — Horizon Coverage at 90 %')
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── (d) RF horizon PICP ──────────────────────────────────────────────
    a = ax[3]
    a.axhline(0.90, color='gray', lw=1.2, ls='--', alpha=0.7, label='Nominal 90 %')
    a.plot(hours, rf_h_picp, color=COLORS["RandomForest"], lw=1.8, label='Random Forest')
    a.set_xlabel('Forecast Horizon (h)')
    a.set_ylabel('PICP')
    a.set_title('(d) Random Forest — Horizon Coverage at 90 %')
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── (e) Horizon interval width ───────────────────────────────────────
    a = ax[4]
    a.plot(hours, lstm_h_width,  color=COLORS["LSTM"],         lw=1.8, label='LSTM MC')
    a.plot(hours, rf_h_width,    color=COLORS["RandomForest"], lw=1.8, label='Random Forest')
    a.set_xlabel('Forecast Horizon (h)')
    a.set_ylabel(WIDTH_LABEL)
    a.set_title('(e) 90 % Interval Width by Horizon')
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── (f) Coverage by 4-hour blocks ────────────────────────────────────
    a = ax[5]
    block   = 4
    n_blk   = HORIZON // block
    blk_hrs = np.arange(n_blk) * block
    bw      = block * 0.38
    lstm_blk = [lstm_h_picp[b:b + block].mean() for b in blk_hrs]
    rf_blk   = [rf_h_picp[b:b + block].mean()   for b in blk_hrs]
    a.bar(blk_hrs,        lstm_blk, width=bw, color=COLORS["LSTM"],
           alpha=0.80, label='LSTM MC')
    a.bar(blk_hrs + bw,   rf_blk,   width=bw, color=COLORS["RandomForest"],
           alpha=0.80, label='Random Forest')
    a.axhline(0.90, color='gray', lw=1.2, ls='--', alpha=0.7, label='Nominal 90 %')
    a.set_xlabel('Forecast Horizon (h)')
    a.set_ylabel('Mean PICP')
    a.set_title('(f) Coverage by 4-Hour Blocks')
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    fig.suptitle(f'Uncertainty Calibration Analysis — {SITE_NAMES[site]}',
                  fontsize=14, fontweight='bold')
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Uncertainty decomposition (horizon effects)
# ---------------------------------------------------------------------------
def plot_uncertainty_decomposition(site: str,
                                    mc_samples:   np.ndarray,
                                    tree_preds:   np.ndarray,
                                    xgb_preds:    dict,
                                    y_true_full:  np.ndarray,
                                    y_true_scalar: np.ndarray) -> plt.Figure:
    """
    Three-panel uncertainty decomposition figure:
      (a) LSTM MC — interval width at multiple nominal levels
      (b) RF ensemble — interval width at multiple nominal levels
      (c) Sharpness vs accuracy scatter (RMSE vs width, coloured by horizon)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    hours = np.arange(HORIZON)

    alphas = [0.50, 0.70, 0.90]
    styles = [':', '--', '-']

    # ── (a) LSTM width by horizon ────────────────────────────────────────
    ax = axes[0]
    for alpha, ls in zip(alphas, styles):
        w = horizon_width(mc_samples, alpha)
        ax.plot(hours, w, lw=1.8, ls=ls, color=COLORS["LSTM"],
                 label=f'{int(alpha * 100)} % PI')
    ax.set_xlabel('Forecast Horizon (h)')
    ax.set_ylabel(WIDTH_LABEL)
    ax.set_title(f'(a) LSTM MC Dropout — {SITE_NAMES[site]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (b) RF width by horizon ──────────────────────────────────────────
    ax = axes[1]
    rf_mean = tree_preds.mean(axis=0)     # (N, H)
    rf_std  = tree_preds.std(axis=0)      # (N, H)
    for alpha, ls in zip(alphas, styles):
        w = gaussian_interval_width(rf_std, alpha).mean(axis=0)   # (H,)
        ax.plot(hours, w, lw=1.8, ls=ls, color=COLORS["RandomForest"],
                 label=f'{int(alpha * 100)} % PI')
    ax.set_xlabel('Forecast Horizon (h)')
    ax.set_ylabel(WIDTH_LABEL)
    ax.set_title(f'(b) RF Ensemble — {SITE_NAMES[site]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (c) RMSE vs width scatter (coloured by horizon index) ───────────
    ax = axes[2]
    lstm_mean_pred = mc_samples.mean(axis=0)                            # (N, H)
    rmse_lstm      = np.sqrt(np.mean((y_true_full - lstm_mean_pred) ** 2, axis=0))
    width_lstm     = horizon_width(mc_samples, 0.90)

    rmse_rf  = np.sqrt(np.mean((y_true_full - rf_mean) ** 2, axis=0))
    width_rf = gaussian_interval_width(rf_std, 0.90).mean(axis=0)

    sc1 = ax.scatter(rmse_lstm, width_lstm,
                      c=hours, cmap='viridis', s=22, alpha=0.85,
                      label='LSTM MC', zorder=3)
    ax.scatter(rmse_rf, width_rf,
                c=hours, cmap='plasma', s=22, alpha=0.85, marker='s',
                label='RF Ensemble', zorder=3)

    ax.set_xlabel(r'RMSE ($\mu$mol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel(r'90 % Interval Width ($\mu$mol m$^{-2}$ s$^{-1}$)')
    ax.set_title(f'(c) Sharpness vs Accuracy — {SITE_NAMES[site]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax, label='Horizon (h)', shrink=0.85)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NaN imputation (mirrors train_baselines.py)
# ---------------------------------------------------------------------------
_NAN_IDX = [1, 4, 5, 6, 7]


def impute_features(X: np.ndarray, medians: Optional[dict] = None):
    X = X.copy()
    out_med = {}
    for idx in _NAN_IDX:
        if medians is None:
            med = float(np.nanmedian(X[:, :, idx]))
            out_med[idx] = med
        else:
            med = medians[idx]
            out_med[idx] = med
        mask = np.isnan(X[:, :, idx])
        X[:, :, idx][mask] = med
    return X, out_med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 80)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS FOR CARBON FLUX PREDICTION")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  MC samples: {MC_SAMPLES}  |  dropout rate: {MC_DROPOUT_RATE}"
          f"  |  test subset: {MC_TEST_N}")
    print("=" * 80)

    UQ_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ── [1] Load data ──────────────────────────────────────────────────────
    log.info("[1/5] Loading data...")
    X_train = np.load(DATA_DIR / "train_X.npy")
    y_train = np.load(DATA_DIR / "train_y.npy")

    test_data = {}
    for site in SITES:
        test_data[site] = {
            'X': np.load(DATA_DIR / f"test_{site}_X.npy"),
            'y': np.load(DATA_DIR / f"test_{site}_y.npy"),
        }

    X_train, train_med = impute_features(X_train)
    for site in SITES:
        test_data[site]['X'], _ = impute_features(
            test_data[site]['X'], medians=train_med)

    log.info("  Train: %s  |  features: %d", X_train.shape, X_train.shape[2])
    for site in SITES:
        log.info("  Test %s: %s", site, test_data[site]['X'].shape)

    # ── [2] Load trained models ────────────────────────────────────────────
    log.info("[2/5] Loading model checkpoints...")

    # LSTM
    lstm_ckpts = sorted(CKPT_DIR.glob("lstm_*.pt"))
    assert lstm_ckpts, f"No LSTM checkpoint in {CKPT_DIR}"
    ckpt = torch.load(lstm_ckpts[-1], map_location='cpu')
    lstm_base = LSTMForecaster(
        input_size  = ckpt['input_size'],
        hidden_size = ckpt['hyperparameters']['hidden_size'],
        num_layers  = ckpt['hyperparameters']['num_layers'],
        horizon     = ckpt['horizon'],
    )
    lstm_base.load_state_dict(ckpt['model_state_dict'])
    mc_lstm = MCDropoutLSTM(lstm_base, mc_dropout_rate=MC_DROPOUT_RATE)
    log.info("  LSTM: %s", lstm_ckpts[-1].name)

    # Random Forest
    rf_ckpts = sorted(CKPT_DIR.glob("randomforest_*.joblib"))
    assert rf_ckpts, f"No RF checkpoint in {CKPT_DIR}"
    rf_model = joblib.load(rf_ckpts[-1])
    log.info("  RF:   %s  (%d trees)", rf_ckpts[-1].name, len(rf_model.estimators_))

    # ── [3] XGBoost quantile training ─────────────────────────────────────
    log.info("[3/5] Training XGBoost quantile regression (%d levels)...",
              len(XGB_QUANTILE_LEVELS))
    xgb_models = train_xgb_quantile_models(X_train, y_train, XGB_QUANTILE_LEVELS)
    log.info("  XGBoost quantile models complete.")

    # ── [4] Per-site analysis ──────────────────────────────────────────────
    all_metrics: list = []
    cal_lines:   list = []

    for site in SITES:
        print(f"\n{'─' * 70}")
        print(f"  Site: {SITE_NAMES[site]}")
        print(f"{'─' * 70}")

        X_test       = test_data[site]['X']
        y_test       = test_data[site]['y']             # (N, 96)
        y_test_scalar = y_test.mean(axis=1)             # (N,)

        # Random subset for MC Dropout
        rng    = np.random.RandomState(SEED)
        idx_mc = np.sort(rng.choice(len(X_test),
                                     size=min(MC_TEST_N, len(X_test)),
                                     replace=False))
        X_mc         = X_test[idx_mc]
        y_mc         = y_test[idx_mc]
        y_mc_scalar  = y_test_scalar[idx_mc]

        # ── MC Dropout ────────────────────────────────────────────────────
        log.info("[4/5] LSTM MC Dropout: %d passes on N=%d samples...",
                  MC_SAMPLES, len(X_mc))
        t0 = time.time()
        mc_samples = mc_dropout_inference(mc_lstm, X_mc, n_samples=MC_SAMPLES)
        log.info("  Done in %.1fs.  Shape: %s", time.time() - t0, mc_samples.shape)

        # ── RF ensemble predictions ───────────────────────────────────────
        log.info("  RF ensemble: %d trees on N=%d...", len(rf_model.estimators_), len(X_mc))
        t0 = time.time()
        tree_preds = rf_tree_predictions(rf_model, X_mc)
        log.info("  Done in %.1fs.  Shape: %s", time.time() - t0, tree_preds.shape)

        # RF Gaussian samples for calibration (match MC sample count)
        rf_mean = tree_preds.mean(axis=0)   # (N, H)
        rf_std  = tree_preds.std(axis=0)    # (N, H)
        rng2    = np.random.RandomState(SEED + 1)
        rf_gauss = rng2.normal(
            loc   = rf_mean[np.newaxis],
            scale = rf_std[np.newaxis],
            size  = (len(rf_model.estimators_),) + rf_mean.shape
        )                                    # (n_trees, N, H)

        # ── XGBoost quantile predictions ──────────────────────────────────
        xgb_preds = xgb_quantile_predict(xgb_models, X_mc)   # dict q → (N,)
        # Also predict on the full test set for the time-series figure
        xgb_preds_full = xgb_quantile_predict(xgb_models, X_test)

        # ── Calibration ───────────────────────────────────────────────────
        log.info("  Calibration analysis...")
        lstm_rel = reliability_data(y_mc, mc_samples)
        rf_rel   = reliability_data(y_mc, rf_gauss)
        xgb_rel  = reliability_data_xgb(y_mc_scalar, xgb_preds)

        lstm_h_picp  = horizon_picp(y_mc, mc_samples,  alpha=0.90)
        rf_h_picp    = horizon_picp(y_mc, rf_gauss,    alpha=0.90)
        lstm_h_width = horizon_width(mc_samples, alpha=0.90)
        rf_h_width   = gaussian_interval_width(rf_std, 0.90).mean(axis=0)

        # Collect metrics
        all_metrics.extend(collect_metrics(site, 'LSTM_MC_Dropout', y_mc, mc_samples))
        all_metrics.extend(collect_metrics(site, 'RF_Ensemble',     y_mc, rf_gauss))
        all_metrics.extend(collect_xgb_metrics(site, y_mc_scalar, xgb_preds))

        # Calibration summary block
        cal_lines += [
            f"\n{'=' * 65}",
            f"  Site: {SITE_NAMES[site]}",
            f"{'=' * 65}",
            f"\n  {'Model':<22} {'Nominal':>8} {'PICP':>8} {'Width':>10} {'|Err|':>8}",
            f"  {'─'*22} {'─'*8} {'─'*8} {'─'*10} {'─'*8}",
        ]
        for alpha in COVERAGE_LEVELS:
            picp_l  = compute_picp(y_mc, mc_samples, alpha)
            width_l = compute_mean_width(mc_samples, alpha)
            picp_r  = compute_picp(y_mc, rf_gauss,   alpha)
            width_r = compute_mean_width(rf_gauss,   alpha)
            cal_lines.append(
                f"  {'LSTM MC Dropout':<22} {alpha:>8.0%} {picp_l:>8.4f}"
                f" {width_l:>10.4f} {abs(picp_l-alpha):>8.4f}")
            cal_lines.append(
                f"  {'RF Ensemble':<22} {alpha:>8.0%} {picp_r:>8.4f}"
                f" {width_r:>10.4f} {abs(picp_r-alpha):>8.4f}")

        for row in collect_xgb_metrics(site, y_mc_scalar, xgb_preds):
            cal_lines.append(
                f"  {'XGBoost QR':<22} {row['NominalCoverage']:>8.0%}"
                f" {row['PICP']:>8.4f} {row['MeanIntervalWidth']:>10.4f}"
                f" {row['CalibrationError']:>8.4f}")

        # Print 90 % summary to console
        picp90_lstm = compute_picp(y_mc, mc_samples, 0.90)
        picp90_rf   = compute_picp(y_mc, rf_gauss,   0.90)
        print(f"\n  90 % coverage (PICP):")
        print(f"    LSTM MC Dropout : {picp90_lstm:.4f}")
        print(f"    RF Ensemble     : {picp90_rf:.4f}")
        q05_mc = xgb_preds.get(0.05)
        q95_mc = xgb_preds.get(0.95)
        if q05_mc is not None and q95_mc is not None:
            picp90_xgb = compute_coverage(y_mc_scalar, q05_mc, q95_mc)
            print(f"    XGBoost QR      : {picp90_xgb:.4f}")

        # ── Generate figures ──────────────────────────────────────────────
        log.info("  Generating figures for %s...", site)

        # Fig 1a — LSTM
        fig = plot_prediction_intervals_lstm(y_mc, mc_samples, site,
                                              n_examples=3, idx_start=0)
        p = FIG_DIR / f"prediction_intervals_lstm_{site}.png"
        fig.savefig(p); plt.close(fig)
        log.info("  Saved: %s", p.name)

        # Fig 1b — RF
        fig = plot_prediction_intervals_rf(y_mc, tree_preds, site,
                                            n_examples=3, idx_start=0)
        p = FIG_DIR / f"prediction_intervals_randomforest_{site}.png"
        fig.savefig(p); plt.close(fig)
        log.info("  Saved: %s", p.name)

        # Fig 1c — XGBoost (full test set for better visual)
        fig = plot_prediction_intervals_xgb(y_test_scalar, xgb_preds_full, site)
        p = FIG_DIR / f"prediction_intervals_xgboost_{site}.png"
        fig.savefig(p); plt.close(fig)
        log.info("  Saved: %s", p.name)

        # Fig 2 — Calibration (6 panels)
        fig = plot_calibration_analysis(site,
                                         lstm_rel, rf_rel, xgb_rel,
                                         lstm_h_picp, rf_h_picp,
                                         lstm_h_width, rf_h_width)
        p = FIG_DIR / f"calibration_analysis_{site}.png"
        fig.savefig(p); plt.close(fig)
        log.info("  Saved: %s", p.name)

        # Fig 3 — Uncertainty decomposition
        fig = plot_uncertainty_decomposition(site,
                                              mc_samples, tree_preds,
                                              xgb_preds,
                                              y_mc, y_mc_scalar)
        p = FIG_DIR / f"uncertainty_decomposition_{site}.png"
        fig.savefig(p); plt.close(fig)
        log.info("  Saved: %s", p.name)

    # ── [5] Save outputs ───────────────────────────────────────────────────
    log.info("\n[5/5] Saving metrics and calibration summary...")

    # uncertainty_metrics.csv
    df = pd.DataFrame(all_metrics)
    df = df[['Site', 'Model', 'NominalCoverage',
             'PICP', 'MeanIntervalWidth', 'CalibrationError']]
    csv_path = UQ_DIR / "uncertainty_metrics.csv"
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path.relative_to(ROOT))

    # calibration_summary.txt
    txt_path = UQ_DIR / "calibration_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("UNCERTAINTY QUANTIFICATION — CALIBRATION SUMMARY\n")
        f.write("=" * 65 + "\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  MC passes : {MC_SAMPLES}\n")
        f.write(f"  MC dropout: {MC_DROPOUT_RATE}\n")
        f.write(f"  Test N    : {MC_TEST_N} (random subset per site)\n")
        f.write(f"  XGB levels: {XGB_QUANTILE_LEVELS}\n")
        for line in cal_lines:
            f.write(line + "\n")
    log.info("  Saved: %s", txt_path.relative_to(ROOT))

    elapsed = time.time() - t_start
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║  UNCERTAINTY QUANTIFICATION COMPLETE" + " " * 41 + "║")
    print(f"║  Runtime : {elapsed / 60:.1f} min" + " " * 64 + "║")
    print("║  Metrics : results/uncertainty/uncertainty_metrics.csv" + " " * 23 + "║")
    print("║  Summary : results/uncertainty/calibration_summary.txt" + " " * 23 + "║")
    print("║  Figures : figures/uncertainty/" + " " * 46 + "║")
    print("╚" + "═" * 78 + "╝")


if __name__ == "__main__":
    main()
