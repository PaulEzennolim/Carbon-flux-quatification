"""
Feature Importance Analysis for Carbon Flux Prediction Models
=============================================================
Analyses which input features and historical time-lags drive model performance.

Approaches implemented:
  1. Built-in MDI importance   (XGBoost, Random Forest)
     Aggregated per feature name (sum over 336 time lags) and per day (14-day heatmap).
  2. Grouped permutation importance  (XGBoost, Random Forest)
     All 336 time-step columns for each feature permuted simultaneously across samples.
     Reports ΔRMSE ± std over n_repeats shuffles.
  3. Occlusion-based ablation  (XGBoost, Random Forest, LSTM)
     Each feature's values are replaced by its cross-site median; ΔRMSE is recorded.
  4. Feature correlation matrix
     Spearman ρ of mean feature values across the lookback window.
  5. Feature dependence plots
     Binned RMSE vs mean feature value for the top-4 most important features.

Usage
-----
  python scripts/feature_importance.py
  python scripts/feature_importance.py --n-repeats 2     # faster
  python scripts/feature_importance.py --skip-perm       # skip permutation (occlusion only)

Outputs
-------
  figures/feature_importance/feature_importance_{model}_{site}.png / .pdf
  figures/feature_importance/temporal_importance_{model}_{site}.png / .pdf
  figures/feature_importance/ablation_comparison_{site}.png / .pdf
  figures/feature_importance/feature_correlations_{site}.png / .pdf
  figures/feature_importance/feature_dependence_{model}_{site}.png / .pdf
  results/analysis/ablation_results.csv
  results/analysis/feature_importance_summary.txt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "results" / "predictions"
BASELINE_DIR = PRED_DIR / "baselines"
MODEL_DIR = ROOT / "models" / "checkpoints" / "baselines"
FIG_DIR = ROOT / "figures" / "feature_importance"
OUT_DIR = ROOT / "results" / "analysis"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITES = ["UK-AMo", "SE-Htm"]
SITE_LABELS = {"UK-AMo": "UK-AMo (Wetland)", "SE-Htm": "SE-Htm (Forest)"}

LOOKBACK = 336   # 14 days × 24 hours
HORIZON  = 96    # 4-day forecast
N_DAYS   = LOOKBACK // 24   # 14
N_FEATURES = 19

FEATURE_NAMES: List[str] = [
    'SW_IN_F', 'LW_IN_F', 'VPD_F', 'TA_F', 'PA_F', 'P_F', 'WS_F',
    'G_F_MDS', 'LE_F_MDS', 'H_F_MDS',
    'MODIS_band_1', 'MODIS_band_2', 'MODIS_band_3', 'MODIS_band_4',
    'MODIS_band_5', 'MODIS_band_6', 'MODIS_band_7',
    'DOY', 'TOD',
]

FEATURE_LABELS: Dict[str, str] = {
    'SW_IN_F':      'Solar Radiation',
    'LW_IN_F':      'Longwave Rad.',
    'VPD_F':        'VPD',
    'TA_F':         'Temperature',
    'PA_F':         'Air Pressure',
    'P_F':          'Precipitation',
    'WS_F':         'Wind Speed',
    'G_F_MDS':      'Ground Heat Flux',
    'LE_F_MDS':     'Latent Heat',
    'H_F_MDS':      'Sensible Heat',
    'MODIS_band_1': 'MODIS Band 1',
    'MODIS_band_2': 'MODIS Band 2',
    'MODIS_band_3': 'MODIS Band 3',
    'MODIS_band_4': 'MODIS Band 4',
    'MODIS_band_5': 'MODIS Band 5',
    'MODIS_band_6': 'MODIS Band 6',
    'MODIS_band_7': 'MODIS Band 7',
    'DOY':          'Day of Year',
    'TOD':          'Time of Day',
}

FEATURE_GROUPS: Dict[str, List[str]] = {
    'Meteorological': ['SW_IN_F', 'LW_IN_F', 'VPD_F', 'TA_F', 'PA_F', 'P_F', 'WS_F'],
    'Energy Fluxes':  ['G_F_MDS', 'LE_F_MDS', 'H_F_MDS'],
    'Remote Sensing': [f'MODIS_band_{i}' for i in range(1, 8)],
    'Temporal':       ['DOY', 'TOD'],
}

GROUP_COLORS: Dict[str, str] = {
    'Meteorological': '#2196F3',
    'Energy Fluxes':  '#FF5722',
    'Remote Sensing': '#4CAF50',
    'Temporal':       '#9C27B0',
}

# feature → group lookup
FEATURE_GROUP_MAP: Dict[str, str] = {
    f: grp for grp, feats in FEATURE_GROUPS.items() for f in feats
}

TREE_MODELS = {'xgboost': 'XGBoost', 'randomforest': 'Random Forest'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def seq_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-sequence RMSE: (n_seq, horizon) → scalar."""
    return float(np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))))


def flatten_X(X: np.ndarray) -> np.ndarray:
    """(n_seq, LOOKBACK, N_FEATURES) → (n_seq, LOOKBACK*N_FEATURES)."""
    return X.reshape(X.shape[0], -1)


def feature_columns(f_idx: int) -> List[int]:
    """Indices of all lookback timestep slots for feature f_idx in a flattened row."""
    return [t * N_FEATURES + f_idx for t in range(LOOKBACK)]


def feature_group_colors(feature_list: List[str]) -> List[str]:
    return [GROUP_COLORS[FEATURE_GROUP_MAP[f]] for f in feature_list]


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cross_site_medians() -> np.ndarray:
    """Return (N_FEATURES,) array of cross-site median values."""
    path = PROCESSED_DIR / "cross_site_medians.json"
    with open(path) as fh:
        d = json.load(fh)
    return np.array([d[name] for name in FEATURE_NAMES])


def load_test_data(site: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test X and y for a site.

    Returns
    -------
    X : (n_seq, LOOKBACK, N_FEATURES)
    y : (n_seq, HORIZON)
    """
    X = np.load(PROCESSED_DIR / f"test_{site}_X.npy")
    y = np.load(PROCESSED_DIR / f"test_{site}_y.npy")

    if X.ndim == 2:
        n_seq = X.shape[0] // LOOKBACK
        X = X[:n_seq * LOOKBACK].reshape(n_seq, LOOKBACK, N_FEATURES)
    if y.ndim == 1:
        n_seq = X.shape[0]
        y = y[:n_seq * HORIZON].reshape(n_seq, HORIZON)

    return X, y


def load_tree_models() -> Dict[str, object]:
    """Load saved XGBoost and Random Forest joblib models."""
    models: Dict[str, object] = {}
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        key = next((k for k in TREE_MODELS if k in path.stem.lower()), None)
        if key and key not in models:
            models[key] = joblib.load(path)
            print(f"  Loaded {TREE_MODELS[key]}: {path.name}")
    return models


def load_lstm_model():
    """Load the LSTM model from disk. Returns None gracefully on failure."""
    try:
        import torch

        sys.path.insert(0, str(ROOT / "models"))
        from lstm_baseline import LSTMBaseline 

        pt_files = sorted(MODEL_DIR.glob("lstm_*.pt"))
        if not pt_files:
            print("  WARNING: No lstm_*.pt found — skipping LSTM.")
            return None

        ckpt = torch.load(pt_files[0], map_location='cpu')

        hp = {}
        hp_files = sorted(MODEL_DIR.glob("hyperparameters_*.json"))
        if hp_files:
            with open(hp_files[0]) as fh:
                hp = json.load(fh).get("lstm", {})

        model = LSTMBaseline(
            input_size=N_FEATURES,
            hidden_size=int(hp.get("hidden_size", 128)),
            num_layers=int(hp.get("num_layers", 2)),
            horizon=HORIZON,
        )

        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            model = ckpt  # full model was saved

        model.eval()
        print(f"  Loaded LSTM: {pt_files[0].name}")
        return model

    except Exception as exc:
        print(f"  WARNING: Could not load LSTM model ({exc}) — skipping LSTM ablation.")
        return None

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_builtin_importance(
    model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and normalise MDI importance from a fitted tree model.

    The model was trained on flattened input (n_seq, LOOKBACK*N_FEATURES),
    so feature_importances_ has shape (LOOKBACK*N_FEATURES,).

    Returns
    -------
    by_feature  : (N_FEATURES,)           sum over all time lags, normalised
    by_lag      : (LOOKBACK,)             sum over all features, normalised
    matrix      : (LOOKBACK, N_FEATURES)  normalised importance matrix [t, f]
    """
    raw = model.feature_importances_               # (LOOKBACK * N_FEATURES,)
    matrix = raw.reshape(LOOKBACK, N_FEATURES).copy()

    total = matrix.sum()
    if total > 0:
        matrix /= total

    by_feature = matrix.sum(axis=0)   # already sums to 1 since matrix does
    by_lag     = matrix.sum(axis=1)
    return by_feature, by_lag, matrix


def compute_grouped_permutation_importance(
    model,
    X_flat: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Grouped permutation importance: shuffle all LOOKBACK timestep columns
    for each feature simultaneously across samples.

    Returns
    -------
    importances : (N_FEATURES, n_repeats)  ΔRMSE per feature per repeat
                  Positive value → feature is informative.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_seq = X_flat.shape[0]
    baseline_preds = model.predict(X_flat)
    if baseline_preds.ndim == 1:
        baseline_preds = baseline_preds.reshape(-1, HORIZON)
    baseline_rmse = seq_rmse(y, baseline_preds)

    importances = np.zeros((N_FEATURES, n_repeats))
    for f_idx in range(N_FEATURES):
        cols = feature_columns(f_idx)
        for r in range(n_repeats):
            X_perm = X_flat.copy()
            perm_idx = rng.permutation(n_seq)
            X_perm[:, cols] = X_flat[perm_idx][:, cols]
            preds = model.predict(X_perm)
            if preds.ndim == 1:
                preds = preds.reshape(-1, HORIZON)
            importances[f_idx, r] = seq_rmse(y, preds) - baseline_rmse

    return importances


def compute_occlusion_importance(
    model,
    X_flat: Optional[np.ndarray],
    y: np.ndarray,
    medians: np.ndarray,
    is_lstm: bool = False,
    lstm_X_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Replace each feature with its cross-site median value and measure ΔRMSE.

    Returns
    -------
    delta_rmse : (N_FEATURES,)  positive = feature is informative
    """
    if is_lstm:
        import torch
        model.eval()
        with torch.no_grad():
            baseline_preds = model(
                torch.tensor(lstm_X_3d, dtype=torch.float32)
            ).numpy()
        baseline_rmse = seq_rmse(y, baseline_preds)

        delta_rmse = np.zeros(N_FEATURES)
        for f_idx in range(N_FEATURES):
            X_occ = lstm_X_3d.copy()
            X_occ[:, :, f_idx] = medians[f_idx]
            with torch.no_grad():
                occ_preds = model(
                    torch.tensor(X_occ, dtype=torch.float32)
                ).numpy()
            delta_rmse[f_idx] = seq_rmse(y, occ_preds) - baseline_rmse

    else:
        baseline_preds = model.predict(X_flat)
        if baseline_preds.ndim == 1:
            baseline_preds = baseline_preds.reshape(-1, HORIZON)
        baseline_rmse = seq_rmse(y, baseline_preds)

        delta_rmse = np.zeros(N_FEATURES)
        for f_idx in range(N_FEATURES):
            cols = feature_columns(f_idx)
            X_occ = X_flat.copy()
            X_occ[:, cols] = medians[f_idx]
            occ_preds = model.predict(X_occ)
            if occ_preds.ndim == 1:
                occ_preds = occ_preds.reshape(-1, HORIZON)
            delta_rmse[f_idx] = seq_rmse(y, occ_preds) - baseline_rmse

    return delta_rmse


def compute_feature_correlations(X: np.ndarray) -> np.ndarray:
    """
    Spearman correlation matrix of the 19 features.
    Uses per-sequence mean value across the lookback window.

    Returns
    -------
    corr_matrix : (N_FEATURES, N_FEATURES)
    """
    X_mean = X.mean(axis=1)                          # (n_seq, N_FEATURES)
    df = pd.DataFrame(X_mean, columns=FEATURE_NAMES)
    return df.corr(method='spearman').values


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_feature_importance(
    builtin_imp: np.ndarray,
    perm_imp: np.ndarray,
    model_name: str,
    site: str,
) -> None:
    """Side-by-side bar chart: MDI (left) vs grouped permutation importance (right)."""
    sort_idx   = np.argsort(builtin_imp)
    feat_sorted = [FEATURE_NAMES[i] for i in sort_idx]
    labels      = [FEATURE_LABELS[f] for f in feat_sorted]
    colors      = feature_group_colors(feat_sorted)

    builtin_sorted = builtin_imp[sort_idx]
    perm_mean  = perm_imp.mean(axis=1)[sort_idx]
    perm_std   = perm_imp.std(axis=1)[sort_idx]
    y_pos = np.arange(N_FEATURES)

    legend_handles = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        f"Feature Importance — {model_name} — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    # Left — MDI
    ax_l.barh(y_pos, builtin_sorted, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(labels, fontsize=8.5)
    ax_l.set_xlabel("MDI importance (normalised, summed over time lags)", fontsize=9)
    ax_l.set_title("Built-in MDI Importance", fontsize=10, fontweight='bold')
    ax_l.grid(axis='x', alpha=0.3)
    ax_l.set_xlim(left=0)
    ax_l.legend(handles=legend_handles, fontsize=7.5, loc='lower right')

    # Right — permutation
    ax_r.barh(y_pos, perm_mean, xerr=perm_std, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5, capsize=3,
              error_kw=dict(ecolor='black', capthick=1, linewidth=1))
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(labels, fontsize=8.5)
    ax_r.set_xlabel(
        "ΔRMSE when feature is permuted\n(higher = more important)", fontsize=9
    )
    ax_r.set_title("Grouped Permutation Importance", fontsize=10, fontweight='bold')
    ax_r.grid(axis='x', alpha=0.3)
    ax_r.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax_r.legend(handles=legend_handles, fontsize=7.5, loc='lower right')

    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    _savefig(fig, FIG_DIR / f"feature_importance_{slug}_{site}.png")


def fig_temporal_importance(
    importance_matrix: np.ndarray,
    model_name: str,
    site: str,
) -> None:
    """
    Heatmap of importance per day × feature.

    importance_matrix : (LOOKBACK, N_FEATURES) — [t, f]
      t=0 → oldest observation (336 h ago), t=335 → most recent (1 h ago)
    """
    # Reverse time axis so index 0 = 1 h ago (most recent)
    imp_rev = importance_matrix[::-1, :]                           # (336, 19)
    # Aggregate into 14 daily buckets (24 h each)
    imp_daily = imp_rev.reshape(N_DAYS, 24, N_FEATURES).sum(axis=1)  # (14, 19)
    lag_total = imp_daily.sum(axis=1)                              # (14,)

    feat_labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]
    day_labels  = [f"D−{d+1}" for d in range(N_DAYS)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [3, 1]},
    )
    fig.suptitle(
        f"Temporal Lag Importance — {model_name} — {SITE_LABELS[site]}\n"
        "(colour = MDI importance summed over 24 h bin; D−1 = most recent day)",
        fontsize=11, fontweight='bold', y=1.01,
    )

    # Heatmap: rows = features, columns = days
    im = ax_top.imshow(
        imp_daily.T,           # (N_FEATURES, N_DAYS)
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest',
    )
    ax_top.set_xticks(range(N_DAYS))
    ax_top.set_xticklabels(day_labels, fontsize=8.5)
    ax_top.set_yticks(range(N_FEATURES))
    ax_top.set_yticklabels(feat_labels, fontsize=8.5)
    ax_top.set_xlabel("Days before forecast start (D−1 = most recent)", fontsize=9)
    plt.colorbar(im, ax=ax_top, shrink=0.8, label="Normalised importance")

    # Colour y-axis labels by feature group
    for yi, fname in enumerate(FEATURE_NAMES):
        ax_top.get_yticklabels()[yi].set_color(
            GROUP_COLORS[FEATURE_GROUP_MAP[fname]]
        )

    # Bar chart of total importance per day
    bar_colors = plt.cm.YlOrRd(lag_total / (lag_total.max() + 1e-12))
    ax_bot.bar(range(N_DAYS), lag_total, color=bar_colors,
               edgecolor='white', linewidth=0.5)
    ax_bot.set_xticks(range(N_DAYS))
    ax_bot.set_xticklabels(day_labels, fontsize=8.5)
    ax_bot.set_ylabel("Total importance", fontsize=9)
    ax_bot.set_title("Aggregate importance per day (all features)", fontsize=9,
                     fontweight='bold')
    ax_bot.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    _savefig(fig, FIG_DIR / f"temporal_importance_{slug}_{site}.png")


def fig_ablation_comparison(
    ablation_results: Dict[str, np.ndarray],
    site: str,
) -> None:
    """
    Grouped horizontal bar chart of ΔRMSE when each feature is occluded.

    Parameters
    ----------
    ablation_results : {model_display_name: (N_FEATURES,) delta_rmse}
    """
    model_names = list(ablation_results.keys())
    n_models = len(model_names)
    if n_models == 0:
        return

    avg_imp = np.mean(list(ablation_results.values()), axis=0)
    sort_idx    = np.argsort(avg_imp)
    feat_sorted = [FEATURE_NAMES[i] for i in sort_idx]
    labels      = [FEATURE_LABELS[f] for f in feat_sorted]
    colors      = feature_group_colors(feat_sorted)
    y_pos = np.arange(N_FEATURES)

    fig, axes = plt.subplots(
        1, n_models, figsize=(6 * n_models, 9), sharey=True
    )
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"Ablation Study (Occlusion) — {SITE_LABELS[site]}\n"
        "ΔRMSE when feature replaced by cross-site median  "
        "(higher = more critical)",
        fontsize=12, fontweight='bold', y=1.02,
    )

    for ax, mname in zip(axes, model_names):
        delta = ablation_results[mname][sort_idx]
        ax.barh(y_pos, delta, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_yticks(y_pos)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("ΔRMSE (µmol m⁻² s⁻¹)", fontsize=9)
        ax.set_title(mname, fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Annotate the 3 largest contributors
        top3 = np.argsort(delta)[-3:]
        for ti in top3:
            if delta[ti] > 0:
                ax.annotate(
                    f"+{delta[ti]:.3f}",
                    xy=(delta[ti], ti),
                    xytext=(delta[ti] + 0.001, ti),
                    fontsize=7, va='center', color='#B71C1C',
                )

    legend_handles = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.04), fontsize=8, frameon=True,
    )

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"ablation_comparison_{site}.png")


def fig_feature_correlations(corr_matrix: np.ndarray, site: str) -> None:
    """Annotated Spearman correlation matrix heatmap."""
    labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N_FEATURES))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells where |ρ| > 0.3 (off-diagonal)
    for i in range(N_FEATURES):
        for j in range(N_FEATURES):
            rho = corr_matrix[i, j]
            if abs(rho) > 0.3 and i != j:
                ax.text(
                    j, i, f"{rho:.2f}", ha='center', va='center',
                    fontsize=6.5,
                    color='white' if abs(rho) > 0.7 else 'black',
                )

    # Colour-code axis labels by feature group
    for xi, fname in enumerate(FEATURE_NAMES):
        color = GROUP_COLORS[FEATURE_GROUP_MAP[fname]]
        ax.get_xticklabels()[xi].set_color(color)
        ax.get_yticklabels()[xi].set_color(color)

    ax.set_title(
        f"Feature Correlation Matrix — {SITE_LABELS[site]}\n"
        "(Spearman ρ, mean value per sequence; cells with |ρ|>0.3 annotated)",
        fontsize=11, fontweight='bold',
    )

    legend_handles = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right', fontsize=8,
        bbox_to_anchor=(1.28, 1.0), frameon=True,
    )

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"feature_correlations_{site}.png")


def fig_feature_dependence(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_features: List[str],
    model_name: str,
    site: str,
) -> None:
    """
    Binned RMSE vs feature value for each of the given features.
    Analogous to SHAP dependence plots.
    """
    n_feat = len(top_features)
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 4), squeeze=False)
    fig.suptitle(
        f"Feature Dependence Plots — {model_name} — {SITE_LABELS[site]}\n"
        "(mean RMSE per feature-value bin; binned over lookback-window mean)",
        fontsize=11, fontweight='bold', y=1.02,
    )

    # Per-sequence RMSE
    seq_errors = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))   # (n_seq,)
    X_mean = X.mean(axis=1)                                           # (n_seq, 19)

    for col_idx, feat_name in enumerate(top_features):
        ax = axes[0][col_idx]
        f_idx  = FEATURE_NAMES.index(feat_name)
        x_vals = X_mean[:, f_idx]
        color  = GROUP_COLORS[FEATURE_GROUP_MAP[feat_name]]

        # Binned means ± std
        n_bins = 20
        bin_edges = np.nanpercentile(x_vals, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_idx   = np.digitize(x_vals, bin_edges) - 1
        bin_idx   = np.clip(bin_idx, 0, len(bin_edges) - 2)

        centers, means, stds = [], [], []
        for b in range(len(bin_edges) - 1):
            m = bin_idx == b
            if m.sum() >= 3:
                centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                means.append(float(np.mean(seq_errors[m])))
                stds.append(float(np.std(seq_errors[m])))

        ax.scatter(x_vals, seq_errors, alpha=0.06, s=3, color=color,
                   rasterized=True)
        if centers:
            ax.plot(centers, means, 'o-', color=color, linewidth=2,
                    markersize=5, zorder=5)
            ax.fill_between(
                centers,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2, color=color,
            )

        rho, p_val = spearmanr(x_vals, seq_errors)
        p_str = "<0.001" if p_val < 0.001 else f"={p_val:.3f}"
        ax.set_title(
            f"{FEATURE_LABELS[feat_name]}\nρ={rho:+.3f}, p{p_str}",
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel(f"{feat_name} (normalised)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Seq. RMSE (µmol m⁻² s⁻¹)", fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    _savefig(fig, FIG_DIR / f"feature_dependence_{slug}_{site}.png")


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_ablation_csv(ablation_by_model: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Write long-form ablation ΔRMSE results."""
    rows = []
    for model_name, site_data in ablation_by_model.items():
        for site, delta_rmse in site_data.items():
            rank = N_FEATURES - np.argsort(np.argsort(-delta_rmse))
            for f_idx, feat_name in enumerate(FEATURE_NAMES):
                rows.append({
                    "site":          site,
                    "model":         model_name,
                    "feature":       feat_name,
                    "feature_group": FEATURE_GROUP_MAP[feat_name],
                    "delta_rmse":    round(float(delta_rmse[f_idx]), 6),
                    "rank":          int(rank[f_idx]),
                })
    out = OUT_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved: {out.relative_to(ROOT)}")


def save_summary_txt(
    builtin_by_model:   Dict[str, Dict[str, np.ndarray]],
    perm_by_model:      Dict[str, Dict[str, np.ndarray]],
    ablation_by_model:  Dict[str, Dict[str, np.ndarray]],
    corr_matrices:      Dict[str, np.ndarray],
) -> None:
    lines = [
        "=" * 72,
        "FEATURE IMPORTANCE ANALYSIS — CARBON FLUX MODELS",
        "=" * 72,
        f"Features        : {N_FEATURES}  ({', '.join(FEATURE_NAMES[:5])}, ...)",
        f"Lookback window : {LOOKBACK} hours  ({N_DAYS} days)",
        f"Forecast horizon: {HORIZON} hours",
        "",
        "Note: MDI importance is computed during model training and reflects the",
        "training data distribution (not test-site-specific). Permutation",
        "importance and occlusion ablation are site-specific.",
        "",
    ]

    all_models = sorted(
        set(list(builtin_by_model) + list(ablation_by_model)),
        key=lambda m: list(TREE_MODELS.values()).index(m)
        if m in TREE_MODELS.values() else 99,
    )

    for site in SITES:
        lines += [
            f"{'─' * 72}",
            f"{SITE_LABELS[site]}",
            f"{'─' * 72}",
        ]

        for mname in all_models:
            mdi  = builtin_by_model.get(mname, {}).get(site)
            perm = perm_by_model.get(mname, {}).get(site)
            occ  = ablation_by_model.get(mname, {}).get(site)

            if mdi is None and occ is None:
                continue

            lines.append(f"\n  {mname}")
            lines.append(f"  {'─' * 40}")

            if mdi is not None:
                top5 = np.argsort(-mdi)[:5]
                lines.append("  Top 5 features (MDI - global from training data):")
                for rank, fi in enumerate(top5, 1):
                    lines.append(
                        f"    {rank}. {FEATURE_NAMES[fi]:<20}  "
                        f"{FEATURE_GROUP_MAP[FEATURE_NAMES[fi]]:<16}  "
                        f"{mdi[fi]:.4f}"
                    )

            if perm is not None:
                perm_mean = perm.mean(axis=1)
                top5p = np.argsort(-perm_mean)[:5]
                lines.append("  Top 5 features (Grouped Permutation):")
                for rank, fi in enumerate(top5p, 1):
                    lines.append(
                        f"    {rank}. {FEATURE_NAMES[fi]:<20}  "
                        f"ΔRMSE={perm_mean[fi]:+.4f} ± {perm.std(axis=1)[fi]:.4f}"
                    )

            if occ is not None:
                top5o = np.argsort(-occ)[:5]
                lines.append("  Top 5 features (Occlusion Ablation):")
                for rank, fi in enumerate(top5o, 1):
                    lines.append(
                        f"    {rank}. {FEATURE_NAMES[fi]:<20}  "
                        f"ΔRMSE={occ[fi]:+.4f}"
                    )

        # Feature collinearity summary
        if site in corr_matrices:
            cm = corr_matrices[site]
            lines.append("\n  High Spearman correlations (|ρ| > 0.7):")
            found = False
            for i in range(N_FEATURES):
                for j in range(i + 1, N_FEATURES):
                    if abs(cm[i, j]) > 0.7:
                        lines.append(
                            f"    {FEATURE_NAMES[i]:<18} ↔  "
                            f"{FEATURE_NAMES[j]:<18}  ρ={cm[i, j]:+.3f}"
                        )
                        found = True
            if not found:
                lines.append("    None found (all |ρ| ≤ 0.7)")

        lines.append("")

    lines += ["=" * 72, "End of Feature Importance Analysis", "=" * 72]
    out = OUT_DIR / "feature_importance_summary.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved: {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature importance analysis for carbon flux models."
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Permutation repeats per feature (default: 5). Use 2 for a quick run.",
    )
    parser.add_argument(
        "--skip-perm", action="store_true",
        help="Skip grouped permutation importance (only run MDI + occlusion).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  FEATURE IMPORTANCE ANALYSIS — CARBON FLUX MODELS")
    print("=" * 72)
    print(f"  n_repeats = {args.n_repeats}  |  skip_perm = {args.skip_perm}")
    print(f"  Figures → {FIG_DIR.relative_to(ROOT)}/")
    print(f"  Results → {OUT_DIR.relative_to(ROOT)}/")

    # ------------------------------------------------------------------
    # [1] Load models and data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading models ...")
    tree_models = load_tree_models()
    lstm_model  = load_lstm_model()

    print("\n[2/5] Loading test data and cross-site medians ...")
    medians   = load_cross_site_medians()
    test_data = {}
    for site in SITES:
        X, y = load_test_data(site)
        test_data[site] = (X, y)
        print(f"  {site}: X={X.shape}  y={y.shape}")

    # ------------------------------------------------------------------
    # [3] Tree model analyses
    # ------------------------------------------------------------------
    print("\n[3/5] Analysing tree models ...")

    builtin_by_model:  Dict[str, Dict[str, np.ndarray]] = {}
    perm_by_model:     Dict[str, Dict[str, np.ndarray]] = {}
    ablation_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    temporal_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    corr_matrices:     Dict[str, np.ndarray]             = {}

    rng = np.random.default_rng(42)

    for model_key, model in tree_models.items():
        disp = TREE_MODELS[model_key]
        builtin_by_model[disp]  = {}
        perm_by_model[disp]     = {}
        ablation_by_model[disp] = {}
        temporal_by_model[disp] = {}

        # MDI is a property of the trained model — compute once globally
        by_feat_global, _, matrix_global = compute_builtin_importance(model)
        print(
            f"\n  {disp}  MDI top feature (global, from training data): "
            f"{FEATURE_NAMES[np.argmax(by_feat_global)]}"
        )

        for site in SITES:
            X, y = test_data[site]
            X_flat = flatten_X(X)

            print(f"\n  {disp} / {site}  (n_seq={X.shape[0]})")

            # Store the same MDI for both sites — it is training-data-intrinsic
            builtin_by_model[disp][site]  = by_feat_global
            temporal_by_model[disp][site] = matrix_global

            # Grouped permutation importance (site-specific — uses test data)
            if not args.skip_perm:
                print(
                    f"    Permutation importance "
                    f"({N_FEATURES} features × {args.n_repeats} repeats) ..."
                )
                perm = compute_grouped_permutation_importance(
                    model, X_flat, y, n_repeats=args.n_repeats, rng=rng
                )
                perm_by_model[disp][site] = perm
                print(
                    f"    Perm top feature: "
                    f"{FEATURE_NAMES[np.argmax(perm.mean(axis=1))]}"
                )

            # Occlusion ablation (site-specific — uses test data)
            print("    Occlusion ablation ...")
            occ = compute_occlusion_importance(model, X_flat, y, medians)
            ablation_by_model[disp][site] = occ
            print(
                f"    Occlusion top feature: "
                f"{FEATURE_NAMES[np.argmax(occ)]}"
            )

    # ------------------------------------------------------------------
    # LSTM occlusion
    # ------------------------------------------------------------------
    if lstm_model is not None:
        print("\n  LSTM occlusion ablation ...")
        ablation_by_model["LSTM"] = {}
        for site in SITES:
            X, y = test_data[site]
            occ = compute_occlusion_importance(
                lstm_model, None, y, medians,
                is_lstm=True, lstm_X_3d=X,
            )
            ablation_by_model["LSTM"][site] = occ
            print(
                f"    {site} top feature: "
                f"{FEATURE_NAMES[np.argmax(occ)]}"
            )
    else:
        print("\n  (LSTM not loaded — skipping LSTM ablation)")

    # ------------------------------------------------------------------
    # Feature correlations
    # ------------------------------------------------------------------
    print("\n[4/5] Computing feature correlations ...")
    for site in SITES:
        X, _ = test_data[site]
        corr_matrices[site] = compute_feature_correlations(X)
        print(f"  {site}: correlation matrix ready")

    # ------------------------------------------------------------------
    # [5] Figures
    # ------------------------------------------------------------------
    print("\n[5/5] Generating figures ...")

    for model_key, model in tree_models.items():
        disp = TREE_MODELS[model_key]
        for site in SITES:
            X, y = test_data[site]
            X_flat = flatten_X(X)

            # Feature importance (MDI + permutation)
            if disp in perm_by_model and site in perm_by_model[disp]:
                fig_feature_importance(
                    builtin_imp=builtin_by_model[disp][site],
                    perm_imp=perm_by_model[disp][site],
                    model_name=disp,
                    site=site,
                )
            else:
                # Permutation skipped: show MDI only (left panel only)
                fig_feature_importance(
                    builtin_imp=builtin_by_model[disp][site],
                    perm_imp=np.zeros((N_FEATURES, 1)),
                    model_name=disp,
                    site=site,
                )

            # Temporal lag importance heatmap
            fig_temporal_importance(
                importance_matrix=temporal_by_model[disp][site],
                model_name=disp,
                site=site,
            )

            # Feature dependence plots (top 4 features by MDI)
            top4_idx   = np.argsort(-builtin_by_model[disp][site])[:4]
            top4_feats = [FEATURE_NAMES[i] for i in top4_idx]
            y_pred = model.predict(X_flat)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, HORIZON)
            fig_feature_dependence(
                X=X, y_true=y, y_pred=y_pred,
                top_features=top4_feats,
                model_name=disp,
                site=site,
            )

    # Ablation comparison (one figure per site, all models)
    for site in SITES:
        site_ablation = {
            mname: ablation_by_model[mname][site]
            for mname in ablation_by_model
            if site in ablation_by_model[mname]
        }
        if site_ablation:
            fig_ablation_comparison(site_ablation, site)

    # Feature correlation heatmaps
    for site in SITES:
        if site in corr_matrices:
            fig_feature_correlations(corr_matrices[site], site)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_ablation_csv(ablation_by_model)
    save_summary_txt(
        builtin_by_model=builtin_by_model,
        perm_by_model=perm_by_model,
        ablation_by_model=ablation_by_model,
        corr_matrices=corr_matrices,
    )

    print("\n" + "=" * 72)
    print("  Analysis complete.")
    print(f"  Figures → {FIG_DIR.relative_to(ROOT)}/")
    print(f"  Results → {OUT_DIR.relative_to(ROOT)}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
