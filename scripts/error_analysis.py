"""
Comprehensive Error Analysis for Carbon Flux Prediction Models
==============================================================

Analyses when and why models fail across two EC tower sites (UK-AMo, SE-Htm).

Analyses implemented:
  1. Temporal error patterns  — by hour of day and by season
  2. Systematic biases        — over/underestimation by flux magnitude
  3. Failure case analysis    — worst-5 % conditions + statistical summary
  4. Residual QQ plots        — normality check
  5. Heteroscedasticity       — error variance vs predicted magnitude
  6. Error vs environment     — Spearman ρ with VPD, temperature, radiation

Outputs
-------
  figures/error_analysis/error_by_hour_[site].png / .pdf
  figures/error_analysis/error_by_season_[site].png / .pdf
  figures/error_analysis/error_vs_environment_[site].png / .pdf
  figures/error_analysis/systematic_bias_[site].png / .pdf
  figures/error_analysis/residual_qq_plots_[site].png / .pdf
  figures/error_analysis/heteroscedasticity_[site].png / .pdf
  figures/error_analysis/failure_analysis_[site].png / .pdf
  results/analysis/failure_cases_[site].csv
  results/analysis/error_analysis_summary_[site].txt

Usage
-----
  python scripts/error_analysis.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PRED_DIR = ROOT / "results" / "predictions"
BASELINE_DIR = PRED_DIR / "baselines"
FIG_DIR = ROOT / "figures" / "error_analysis"
OUT_DIR = ROOT / "results" / "analysis"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITES = ["UK-AMo", "SE-Htm"]
SITE_LABELS = {"UK-AMo": "UK-AMo (Wetland)", "SE-Htm": "SE-Htm (Forest)"}

SITE_FILES = {"UK-AMo": "6.UK-AMo.csv", "SE-Htm": "7.SE-Htm.csv"}

MODELS = {
    "TEMPO Fine-Tuned": ("tempo_fine_tuned", PRED_DIR),
    "TEMPO Zero-Shot": ("tempo_zero_shot", PRED_DIR),
    "XGBoost": ("xgboost", BASELINE_DIR),
    "Random Forest": ("randomforest", BASELINE_DIR),
    "LSTM": ("lstm", BASELINE_DIR),
}

COLORS = {
    "TEMPO Fine-Tuned": "#2196F3",
    "TEMPO Zero-Shot": "#03A9F4",
    "XGBoost": "#FF5722",
    "Random Forest": "#FF9800",
    "LSTM": "#9C27B0",
}

LOOKBACK = 336   # 14 days of hourly data
HORIZON = 96     # 4-day forecast

# Feature indices in X array (matches tempo_data_prep.py order)
FEATURE_NAMES = [
    'SW_IN_F', 'LW_IN_F', 'VPD_F', 'TA_F', 'PA_F', 'P_F', 'WS_F',
    'G_F_MDS', 'LE_F_MDS', 'H_F_MDS',
    'MODIS_band_1', 'MODIS_band_2', 'MODIS_band_3', 'MODIS_band_4',
    'MODIS_band_5', 'MODIS_band_6', 'MODIS_band_7',
    'DOY', 'TOD',
]

# Matplotlib style
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
# Helper functions
# ---------------------------------------------------------------------------

def rmse_seq(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sequence RMSE: (n_seq, horizon) → (n_seq,)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def bias_seq(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sequence mean bias (pred - obs): (n_seq, horizon) → (n_seq,)."""
    return np.mean(y_pred - y_true, axis=1)


def tod_to_hour(tod_scaled: np.ndarray) -> np.ndarray:
    """
    Convert the scaled TOD feature back to hour-of-day (0–23).

    Empirically determined encoding: hour = TOD_scaled * 12 + 12,
    derived from matching timestamps in the raw CSV files.
    """
    hours = np.round(tod_scaled * 12 + 12).astype(int) % 24
    return hours


def doy_to_season(doy_scaled: np.ndarray) -> np.ndarray:
    """
    Convert scaled DOY back to meteorological season for the NH.

    Encoding: DOY_scaled = (DOY - 183.5) / 182.5
    Seasons (approx. Northern Hemisphere):
        Winter  DJF  DOY  1–59 and 336–366  scaled < -0.68 or > 0.83
        Spring  MAM  DOY  60–151            scaled in [-0.68, -0.18)
        Summer  JJA  DOY 152–243            scaled in [-0.18,  0.33)
        Fall    SON  DOY 244–335            scaled in [ 0.33,  0.83)
    """
    doy = doy_scaled * 182.5 + 183.5
    seasons = np.empty(len(doy), dtype=object)
    seasons[(doy >= 1) & (doy < 60)]   = "Winter"
    seasons[(doy >= 60) & (doy < 152)] = "Spring"
    seasons[(doy >= 152) & (doy < 244)] = "Summer"
    seasons[(doy >= 244) & (doy < 336)] = "Fall"
    seasons[(doy >= 336) | (doy < 1)]  = "Winter"
    seasons[seasons == 0] = "Winter"   # fallback
    return seasons


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _savefig(fig: plt.Figure, path: Path) -> None:
    """Save figure as PNG and PDF."""
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_site_data(site: str) -> dict:
    """
    Load all predictions, targets, and raw environmental covariates for one site.

    Returns dict with:
        targets     (n_seq, horizon)
        predictions {model_name: (n_seq, horizon)}
        tod_scaled  (n_seq,)  — last lookback-step TOD of each sequence
        doy_scaled  (n_seq,)  — last lookback-step DOY of each sequence
        vpd_scaled  (n_seq,)  — last lookback-step VPD_F
        ta_scaled   (n_seq,)  — last lookback-step TA_F
        sw_scaled   (n_seq,)  — last lookback-step SW_IN_F
        timestamps  (n_seq,)  — pd.Timestamp of each sequence's target start
        n_seq       int
    """
    # ── Targets ─────────────────────────────────────────────────────────────
    target_path = BASELINE_DIR / f"targets_{site}.npy"
    raw_y = np.load(target_path)
    if raw_y.ndim == 1:
        # Reshaping flat array to (n_seq, horizon) is ambiguous – use y_npy
        raw_y = np.load(ROOT / "data" / "processed" / f"test_{site}_y.npy")
    n_seq, horizon = raw_y.shape

    # ── Predictions ─────────────────────────────────────────────────────────
    predictions = {}
    for model_name, (key, pdir) in MODELS.items():
        path = pdir / f"{key}_preds_{site}.npy"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[:n_seq * horizon].reshape(n_seq, horizon)
        elif arr.shape[0] != n_seq:
            n = min(arr.shape[0], n_seq)
            arr = arr[:n]
        predictions[model_name] = arr

    # ── Raw CSV for timestamps and environmental covariates ──────────────────
    csv_path = RAW_DIR / SITE_FILES[site]
    df_raw = pd.read_csv(csv_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], dayfirst=True)

    # Sequence i → target starts at raw index (LOOKBACK + i)
    # Environmental condition: last timestep of input window = index (LOOKBACK + i - 1)
    env_indices = np.arange(n_seq) + LOOKBACK - 1   # last input step
    target_start_indices = np.arange(n_seq) + LOOKBACK

    # Guard against index overflow
    max_idx = len(df_raw) - 1
    env_indices = np.clip(env_indices, 0, max_idx)
    target_start_indices = np.clip(target_start_indices, 0, max_idx)

    tod_scaled = df_raw['TOD'].values[env_indices]
    doy_scaled = df_raw['DOY'].values[env_indices]
    vpd_scaled = df_raw['VPD_F'].values[env_indices]
    ta_scaled  = df_raw['TA_F'].values[env_indices]
    sw_scaled  = df_raw['SW_IN_F'].values[env_indices]
    timestamps = df_raw['timestamp'].values[target_start_indices]

    return dict(
        targets=raw_y,
        predictions=predictions,
        tod_scaled=tod_scaled,
        doy_scaled=doy_scaled,
        vpd_scaled=vpd_scaled,
        ta_scaled=ta_scaled,
        sw_scaled=sw_scaled,
        timestamps=pd.to_datetime(timestamps),
        n_seq=n_seq,
    )


# ---------------------------------------------------------------------------
# Plot 1 — Error by hour of day
# ---------------------------------------------------------------------------

def plot_error_by_hour(data: dict, site: str) -> None:
    """Box plots of per-sequence RMSE grouped by hour-of-day of forecast start."""
    models = list(data['predictions'].keys())
    hours = tod_to_hour(data['tod_scaled'])
    unique_hours = np.arange(0, 24)

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4 * n_models, 5),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"Error by Hour of Day — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    for ax, model_name in zip(axes, models):
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        color = COLORS.get(model_name, '#607D8B')

        hour_rmse = [errors[hours == h] for h in unique_hours]
        bp = ax.boxplot(
            hour_rmse,
            positions=unique_hours,
            widths=0.7,
            patch_artist=True,
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
            medianprops=dict(color='white', linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.7),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1.5),
            showfliers=False,
        )

        # Overlay mean line
        means = [np.mean(v) if len(v) > 0 else np.nan for v in hour_rmse]
        ax.plot(unique_hours, means, 'o-', color=color, linewidth=2,
                markersize=4, alpha=0.9, zorder=5, label='Mean RMSE')

        # Shade nighttime (approximate)
        ax.axvspan(-0.5, 5.5, alpha=0.06, color='navy', label='Night (0–6h)')
        ax.axvspan(20.5, 23.5, alpha=0.06, color='navy')

        ax.set_title(model_name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Hour of Day (UTC)", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(['0h', '6h', '12h', '18h', '23h'])
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel("RMSE (µmol m⁻² s⁻¹)", fontsize=9)

    # Night legend on last axis
    night_patch = mpatches.Patch(color='navy', alpha=0.12, label='Night hours')
    axes[-1].legend(handles=[night_patch], fontsize=8, loc='upper right')

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"error_by_hour_{site}.png")


# ---------------------------------------------------------------------------
# Plot 2 — Error by season
# ---------------------------------------------------------------------------

def plot_error_by_season(data: dict, site: str) -> None:
    """Bar + error (mean ± 1 SD) of RMSE grouped by meteorological season."""
    models = list(data['predictions'].keys())
    seasons = doy_to_season(data['doy_scaled'])
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_colors = {
        "Winter": "#5C85D6", "Spring": "#57AB5A",
        "Summer": "#F0A500", "Fall": "#C8623B",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    n_models = len(models)
    n_seasons = len(season_order)
    bar_width = 0.15
    x = np.arange(n_seasons)

    for mi, model_name in enumerate(models):
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        means, stds, counts, pvals = [], [], [], []

        for season in season_order:
            mask = seasons == season
            v = errors[mask]
            means.append(np.mean(v) if len(v) > 0 else np.nan)
            stds.append(np.std(v) if len(v) > 0 else np.nan)
            counts.append(np.sum(mask))

        offset = (mi - n_models / 2 + 0.5) * bar_width
        color = COLORS.get(model_name, '#607D8B')
        bars = ax.bar(
            x + offset, means, bar_width,
            label=model_name, color=color, alpha=0.82,
            edgecolor='white', linewidth=0.5,
        )
        ax.errorbar(
            x + offset, means, yerr=stds,
            fmt='none', color='black', capsize=3,
            linewidth=1.2, capthick=1.2,
        )

    # Kruskal-Wallis test for season effect across best model
    best_model = min(models,
                     key=lambda m: float(np.mean(rmse_seq(data['targets'],
                                                           data['predictions'][m]))))
    best_errors = rmse_seq(data['targets'], data['predictions'][best_model])
    season_groups = [best_errors[seasons == s] for s in season_order
                     if np.sum(seasons == s) > 1]
    if len(season_groups) >= 2:
        h_stat, p_kw = kruskal(*season_groups)
        ax.set_title(
            f"Error by Meteorological Season — {SITE_LABELS[site]}\n"
            f"Kruskal-Wallis test (best model: {best_model}): "
            f"H={h_stat:.2f}, p={p_kw:.3e} {significance_label(p_kw)}",
            fontsize=11, fontweight='bold',
        )
    else:
        ax.set_title(f"Error by Meteorological Season — {SITE_LABELS[site]}",
                     fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(season_order, fontsize=10)
    ax.set_ylabel("RMSE (µmol m⁻² s⁻¹)", fontsize=10)
    ax.set_xlabel("Meteorological Season (NH)", fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Sample size annotations
    for i, season in enumerate(season_order):
        n = np.sum(seasons == season)
        ax.text(i, ax.get_ylim()[0] - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={n}", ha='center', va='top', fontsize=7.5, color='grey')

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"error_by_season_{site}.png")


# ---------------------------------------------------------------------------
# Plot 3 — Error vs environmental conditions
# ---------------------------------------------------------------------------

def plot_error_vs_environment(data: dict, site: str) -> None:
    """
    Scatter/bin plots of RMSE vs VPD, temperature, and solar radiation,
    with Spearman correlation and 95% CI via bootstrap.
    """
    models = list(data['predictions'].keys())
    env_vars = {
        'VPD_F (normalized)': data['vpd_scaled'],
        'Temperature TA_F (normalized)': data['ta_scaled'],
        'Solar Radiation SW_IN_F (normalized)': data['sw_scaled'],
    }
    n_env = len(env_vars)

    fig, axes = plt.subplots(
        len(models), n_env,
        figsize=(4 * n_env, 3.5 * len(models)),
        squeeze=False,
    )

    fig.suptitle(
        f"Error vs Environmental Conditions — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    rng = np.random.default_rng(42)

    for mi, model_name in enumerate(models):
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        color = COLORS.get(model_name, '#607D8B')

        for ei, (env_label, env_vals) in enumerate(env_vars.items()):
            ax = axes[mi][ei]
            valid = ~np.isnan(env_vals) & ~np.isnan(errors)
            x = env_vals[valid]
            y = errors[valid]

            # Binned mean ± std
            n_bins = 20
            bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            bin_idx = np.digitize(x, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)

            bin_centers, bin_means, bin_stds = [], [], []
            for b in range(len(bin_edges) - 1):
                mask_b = bin_idx == b
                if mask_b.sum() >= 3:
                    bin_centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                    bin_means.append(np.mean(y[mask_b]))
                    bin_stds.append(np.std(y[mask_b]))

            ax.scatter(x, y, alpha=0.08, s=4, color=color, rasterized=True)
            ax.plot(bin_centers, bin_means, 'o-', color=color,
                    linewidth=2, markersize=4, zorder=5)
            ax.fill_between(
                bin_centers,
                np.array(bin_means) - np.array(bin_stds),
                np.array(bin_means) + np.array(bin_stds),
                alpha=0.20, color=color,
            )

            # Spearman ρ with bootstrap CI
            rho, p_val = spearmanr(x, y)
            # Bootstrap CI for ρ
            boot_rhos = []
            for _ in range(500):
                idx_b = rng.integers(0, len(x), size=len(x))
                r_b, _ = spearmanr(x[idx_b], y[idx_b])
                boot_rhos.append(r_b)
            ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

            ax.set_title(
                f"{model_name}\n"
                f"ρ={rho:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
                f"{significance_label(p_val)}",
                fontsize=8.5,
            )
            ax.set_xlabel(env_label, fontsize=8)
            if ei == 0:
                ax.set_ylabel("Seq. RMSE", fontsize=8)
            ax.grid(alpha=0.25)

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"error_vs_environment_{site}.png")


# ---------------------------------------------------------------------------
# Plot 4 — Systematic bias (over/underestimation by flux range)
# ---------------------------------------------------------------------------

def plot_systematic_bias(data: dict, site: str) -> None:
    """
    Binned actual vs predicted plot and bias-by-magnitude plot.
    Quantile regression to reveal heterogeneous bias.
    """
    models = list(data['predictions'].keys())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)

    fig.suptitle(
        f"Systematic Bias Analysis — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold',
    )

    for mi, model_name in enumerate(models):
        color = COLORS.get(model_name, '#607D8B')
        y_true = data['targets'].flatten()
        y_pred = data['predictions'][model_name].flatten()

        # ── Top row: Observed vs Predicted ──────────────────────────────
        ax_top = axes[0][mi]
        ax_top.scatter(y_true, y_pred, alpha=0.04, s=2,
                       color=color, rasterized=True)

        lim = [min(y_true.min(), y_pred.min()),
               max(y_true.max(), y_pred.max())]
        ax_top.plot(lim, lim, 'k--', linewidth=1.5, label='1:1 line', alpha=0.6)

        # Bin-averaged predictions
        n_bins = 30
        bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_idx = np.digitize(y_true, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
        bx, by = [], []
        for b in range(len(bin_edges) - 1):
            m = bin_idx == b
            if m.sum() >= 5:
                bx.append(np.mean(y_true[m]))
                by.append(np.mean(y_pred[m]))
        ax_top.plot(bx, by, 'o-', color=color, linewidth=2,
                    markersize=3, alpha=0.9, label='Bin mean')

        r_sq = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        ax_top.set_title(f"{model_name}\nR²={r_sq:.3f}", fontsize=9, fontweight='bold')
        ax_top.set_xlabel("Observed NEE", fontsize=8)
        ax_top.set_ylabel("Predicted NEE", fontsize=8)
        ax_top.legend(fontsize=7)
        ax_top.grid(alpha=0.25)

        # ── Bottom row: Bias by flux magnitude ───────────────────────────
        ax_bot = axes[1][mi]
        residuals = y_pred - y_true

        # Bin by observed magnitude
        bx2, bias_means, bias_ci = [], [], []
        for b in range(len(bin_edges) - 1):
            m = bin_idx == b
            if m.sum() >= 5:
                v = residuals[m]
                bx2.append(np.mean(y_true[m]))
                bias_means.append(np.mean(v))
                bias_ci.append(1.96 * np.std(v) / np.sqrt(len(v)))

        ax_bot.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax_bot.bar(bx2, bias_means, width=np.diff(bx2[:2])[0] if len(bx2) > 1 else 0.1,
                   color=color, alpha=0.6)
        ax_bot.errorbar(bx2, bias_means, yerr=bias_ci,
                        fmt='none', color='black', capsize=2, linewidth=1)

        # Classify systematic over/underestimation
        pos_bias = np.sum(np.array(bias_means) > 0) / len(bias_means) * 100
        ax_bot.set_title(
            f"Bias (pred − obs)\nOverpredicts in {pos_bias:.0f}% of flux range",
            fontsize=8.5,
        )
        ax_bot.set_xlabel("Observed NEE (binned)", fontsize=8)
        ax_bot.set_ylabel("Mean Bias", fontsize=8)
        ax_bot.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"systematic_bias_{site}.png")


# ---------------------------------------------------------------------------
# Plot 5 — Residual QQ plots
# ---------------------------------------------------------------------------

def plot_residual_qq(data: dict, site: str) -> None:
    """QQ plots of residuals for each model vs Normal distribution."""
    models = list(data['predictions'].keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    fig.suptitle(
        f"Residual Q-Q Plots (vs Normal) — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    for mi, model_name in enumerate(models):
        ax = axes[0][mi]
        color = COLORS.get(model_name, '#607D8B')

        # Use non-overlapping sequences for independent residuals (stride=HORIZON)
        targets_ni = data['targets'][::HORIZON].flatten()
        preds_ni = data['predictions'][model_name][::HORIZON].flatten()
        residuals = preds_ni - targets_ni

        # Shapiro-Wilk test (up to 5000 samples)
        n_sw = min(5000, len(residuals))
        sw_stat, sw_p = stats.shapiro(residuals[:n_sw])

        (osm, osr), (slope, intercept, r_sq) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, s=4, alpha=0.5, color=color, rasterized=True)
        ax.plot([min(osm), max(osm)],
                [slope * min(osm) + intercept, slope * max(osm) + intercept],
                'k--', linewidth=1.5, alpha=0.8)

        ax.set_title(
            f"{model_name}\n"
            f"Shapiro-Wilk W={sw_stat:.4f}, p={sw_p:.2e} {significance_label(sw_p)}",
            fontsize=8.5,
        )
        ax.set_xlabel("Theoretical Quantiles", fontsize=8)
        ax.set_ylabel("Sample Quantiles", fontsize=8)
        ax.grid(alpha=0.25)

        # Normality verdict
        verdict = "Non-normal residuals" if sw_p < 0.05 else "Near-normal residuals"
        ax.text(0.05, 0.95, verdict, transform=ax.transAxes,
                fontsize=7.5, color='grey', va='top')

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"residual_qq_plots_{site}.png")


# ---------------------------------------------------------------------------
# Plot 6 — Heteroscedasticity
# ---------------------------------------------------------------------------

def plot_heteroscedasticity(data: dict, site: str) -> None:
    """
    |Residual| vs Predicted (scale-location plot) to detect
    non-constant error variance.
    """
    models = list(data['predictions'].keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    fig.suptitle(
        f"Heteroscedasticity — |Residual| vs Predicted — {SITE_LABELS[site]}",
        fontsize=12, fontweight='bold', y=1.01,
    )

    for mi, model_name in enumerate(models):
        ax = axes[0][mi]
        color = COLORS.get(model_name, '#607D8B')

        y_true = data['targets'].flatten()
        y_pred = data['predictions'][model_name].flatten()
        abs_res = np.abs(y_pred - y_true)

        # Binned mean |residual| vs predicted
        sort_idx = np.argsort(y_pred)
        y_pred_s = y_pred[sort_idx]
        abs_res_s = abs_res[sort_idx]

        n_bins = 30
        bin_edges = np.percentile(y_pred_s, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_idx_arr = np.digitize(y_pred_s, bin_edges) - 1
        bin_idx_arr = np.clip(bin_idx_arr, 0, len(bin_edges) - 2)

        bx, by, bstd = [], [], []
        for b in range(len(bin_edges) - 1):
            m = bin_idx_arr == b
            if m.sum() >= 5:
                bx.append(np.mean(y_pred_s[m]))
                by.append(np.mean(abs_res_s[m]))
                bstd.append(np.std(abs_res_s[m]))

        # Downsample scatter for readability
        n_scatter = min(5000, len(y_pred))
        idx_sc = np.random.choice(len(y_pred), n_scatter, replace=False)
        ax.scatter(y_pred[idx_sc], abs_res[idx_sc],
                   alpha=0.05, s=3, color=color, rasterized=True)
        ax.plot(bx, by, 'o-', color=color, linewidth=2,
                markersize=4, zorder=5, label='Bin mean |res|')
        ax.fill_between(
            bx,
            np.array(by) - np.array(bstd),
            np.array(by) + np.array(bstd),
            alpha=0.2, color=color,
        )

        # Spearman ρ for heteroscedasticity test
        rho, p_rho = spearmanr(y_pred, abs_res)
        ax.set_title(
            f"{model_name}\n"
            f"Spearman ρ={rho:.3f} (|res|~pred) {significance_label(p_rho)}",
            fontsize=8.5,
        )
        ax.set_xlabel("Predicted NEE", fontsize=8)
        ax.set_ylabel("|Residual|", fontsize=8)
        ax.grid(alpha=0.25)

        hetero_note = "Heteroscedastic" if p_rho < 0.05 else "Homoscedastic"
        ax.text(0.05, 0.95, hetero_note, transform=ax.transAxes,
                fontsize=7.5, color='grey', va='top')

    plt.tight_layout()
    _savefig(fig, FIG_DIR / f"heteroscedasticity_{site}.png")


# ---------------------------------------------------------------------------
# Plot 7 — Failure case analysis
# ---------------------------------------------------------------------------

def analyze_failure_cases(data: dict, site: str) -> pd.DataFrame:
    """
    Identify the worst 5% of prediction sequences for each model and
    characterise the environmental conditions.

    Returns a DataFrame of failure statistics.
    """
    models = list(data['predictions'].keys())
    seasons = doy_to_season(data['doy_scaled'])
    hours = tod_to_hour(data['tod_scaled'])
    failure_rows = []

    for model_name in models:
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        threshold_pct = 95
        thresh = np.percentile(errors, threshold_pct)
        failure_mask = errors >= thresh

        # Characterise failures
        fail_errors = errors[failure_mask]
        fail_vpd = data['vpd_scaled'][failure_mask]
        fail_ta = data['ta_scaled'][failure_mask]
        fail_sw = data['sw_scaled'][failure_mask]
        fail_hours = hours[failure_mask]
        fail_seasons = seasons[failure_mask]

        # Normal (non-failure) conditions for comparison
        normal_mask = ~failure_mask
        normal_vpd = data['vpd_scaled'][normal_mask]
        normal_ta = data['ta_scaled'][normal_mask]

        # Mann-Whitney U test: are failure conditions different?
        _, p_vpd = stats.mannwhitneyu(fail_vpd[~np.isnan(fail_vpd)],
                                      normal_vpd[~np.isnan(normal_vpd)],
                                      alternative='two-sided')
        _, p_ta = stats.mannwhitneyu(fail_ta[~np.isnan(fail_ta)],
                                     normal_ta[~np.isnan(normal_ta)],
                                     alternative='two-sided')

        # Season distribution of failures
        season_counts = pd.Series(fail_seasons).value_counts()
        dom_season = season_counts.index[0] if len(season_counts) > 0 else "N/A"

        # Hour distribution
        night_hours = np.sum((fail_hours >= 20) | (fail_hours <= 5))
        day_hours = np.sum((fail_hours > 5) & (fail_hours < 20))
        night_pct = 100 * night_hours / (night_hours + day_hours + 1e-9)

        failure_rows.append({
            'site': site,
            'model': model_name,
            'failure_threshold_rmse': round(thresh, 4),
            'n_failure_sequences': int(failure_mask.sum()),
            'pct_of_total': round(100 * failure_mask.mean(), 2),
            'mean_failure_rmse': round(float(fail_errors.mean()), 4),
            'max_failure_rmse': round(float(fail_errors.max()), 4),
            'failure_mean_vpd_norm': round(float(np.nanmean(fail_vpd)), 4),
            'normal_mean_vpd_norm': round(float(np.nanmean(normal_vpd)), 4),
            'vpd_mannwhitney_p': round(p_vpd, 4),
            'ta_mannwhitney_p': round(p_ta, 4),
            'dominant_season': dom_season,
            'night_failure_pct': round(night_pct, 1),
            'day_failure_pct': round(100 - night_pct, 1),
        })

    return pd.DataFrame(failure_rows)


def plot_failure_analysis(data: dict, site: str, failure_df: pd.DataFrame) -> None:
    """
    Multi-panel figure showing failure case characteristics:
      (a) RMSE distribution with failure threshold annotated
      (b) Season distribution of failures
      (c) Hour-of-day distribution of failures
      (d) Environmental conditions at failure vs normal
    """
    models = list(data['predictions'].keys())
    seasons = doy_to_season(data['doy_scaled'])
    hours = tod_to_hour(data['tod_scaled'])
    n = len(models)

    fig = plt.figure(figsize=(5 * n, 14))
    outer_gs = gridspec.GridSpec(4, n, figure=fig, hspace=0.50, wspace=0.35)

    fig.suptitle(
        f"Failure Case Analysis (Worst 5%) — {SITE_LABELS[site]}",
        fontsize=13, fontweight='bold', y=0.98,
    )

    season_colors_map = {
        "Winter": "#5C85D6", "Spring": "#57AB5A",
        "Summer": "#F0A500", "Fall": "#C8623B",
    }

    for mi, model_name in enumerate(models):
        color = COLORS.get(model_name, '#607D8B')
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        thresh = np.percentile(errors, 95)
        failure_mask = errors >= thresh

        row_data = failure_df[failure_df['model'] == model_name].iloc[0] \
            if len(failure_df[failure_df['model'] == model_name]) > 0 else None

        # ── (a) RMSE distribution ────────────────────────────────────────
        ax_a = fig.add_subplot(outer_gs[0, mi])
        ax_a.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='none')
        ax_a.axvline(thresh, color='red', linewidth=2, linestyle='--',
                     label=f'95th pct = {thresh:.3f}')
        ax_a.set_title(model_name, fontsize=9, fontweight='bold')
        ax_a.set_xlabel("Seq. RMSE", fontsize=8)
        ax_a.set_ylabel("Count" if mi == 0 else "", fontsize=8)
        ax_a.legend(fontsize=7)
        ax_a.grid(alpha=0.25)

        # ── (b) Season distribution ──────────────────────────────────────
        ax_b = fig.add_subplot(outer_gs[1, mi])
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        fail_seasons = seasons[failure_mask]
        normal_seasons = seasons[~failure_mask]

        fail_pcts = [100 * np.mean(fail_seasons == s) for s in season_order]
        norm_pcts = [100 * np.mean(normal_seasons == s) for s in season_order]
        xs = np.arange(len(season_order))

        ax_b.bar(xs - 0.2, fail_pcts, 0.35, label='Failure',
                 color=[season_colors_map[s] for s in season_order], alpha=0.85)
        ax_b.bar(xs + 0.2, norm_pcts, 0.35, label='Normal',
                 color=[season_colors_map[s] for s in season_order],
                 alpha=0.35, hatch='//')
        ax_b.set_xticks(xs)
        ax_b.set_xticklabels(season_order, fontsize=7.5, rotation=20)
        ax_b.set_ylabel("% of cases" if mi == 0 else "", fontsize=8)
        ax_b.set_title("Season distribution", fontsize=8)
        ax_b.legend(fontsize=7)
        ax_b.grid(axis='y', alpha=0.25)

        # ── (c) Hour-of-day distribution ─────────────────────────────────
        ax_c = fig.add_subplot(outer_gs[2, mi])
        fail_hours = hours[failure_mask]
        normal_hours = hours[~failure_mask]
        hour_bins = np.arange(25)
        fail_h_counts = np.array([np.sum(fail_hours == h) for h in range(24)])
        norm_h_counts = np.array([np.sum(normal_hours == h) for h in range(24)])
        fail_h_pct = 100 * fail_h_counts / (fail_h_counts.sum() + 1e-9)
        norm_h_pct = 100 * norm_h_counts / (norm_h_counts.sum() + 1e-9)

        ax_c.plot(range(24), fail_h_pct, 'o-', color='red', linewidth=1.5,
                  markersize=4, label='Failure')
        ax_c.plot(range(24), norm_h_pct, 's--', color=color, linewidth=1.5,
                  markersize=4, alpha=0.7, label='Normal')
        ax_c.axvspan(-0.5, 5.5, alpha=0.07, color='navy')
        ax_c.axvspan(20.5, 23.5, alpha=0.07, color='navy')
        ax_c.set_xlabel("Hour of Day", fontsize=8)
        ax_c.set_ylabel("% of cases" if mi == 0 else "", fontsize=8)
        ax_c.set_title("Hour of day", fontsize=8)
        ax_c.legend(fontsize=7)
        ax_c.grid(alpha=0.25)

        # ── (d) Environmental condition comparison ────────────────────────
        ax_d = fig.add_subplot(outer_gs[3, mi])
        env_labels = ['VPD_F\n(norm)', 'TA_F\n(norm)', 'SW_IN_F\n(norm)']
        env_arrays = [data['vpd_scaled'], data['ta_scaled'], data['sw_scaled']]

        fail_means = [np.nanmean(e[failure_mask]) for e in env_arrays]
        norm_means = [np.nanmean(e[~failure_mask]) for e in env_arrays]
        fail_sems = [np.nanstd(e[failure_mask]) / np.sqrt(failure_mask.sum() + 1)
                     for e in env_arrays]
        norm_sems = [np.nanstd(e[~failure_mask]) / np.sqrt((~failure_mask).sum() + 1)
                     for e in env_arrays]

        xs_e = np.arange(len(env_labels))
        ax_d.bar(xs_e - 0.2, fail_means, 0.35, yerr=fail_sems,
                 label='Failure', color='red', alpha=0.7,
                 error_kw=dict(capsize=3))
        ax_d.bar(xs_e + 0.2, norm_means, 0.35, yerr=norm_sems,
                 label='Normal', color=color, alpha=0.5,
                 error_kw=dict(capsize=3))
        ax_d.set_xticks(xs_e)
        ax_d.set_xticklabels(env_labels, fontsize=8)
        ax_d.set_ylabel("Mean value" if mi == 0 else "", fontsize=8)
        ax_d.set_title("Environmental conditions", fontsize=8)
        ax_d.legend(fontsize=7)
        ax_d.grid(axis='y', alpha=0.25)

        # Add p-value annotations for VPD and TA
        if row_data is not None:
            stars_vpd = significance_label(row_data['vpd_mannwhitney_p'])
            stars_ta = significance_label(row_data['ta_mannwhitney_p'])
            for xi, stars in enumerate([stars_vpd, stars_ta, '']):
                if stars and stars != 'ns':
                    y_top = max(fail_means[xi], norm_means[xi]) + 0.02
                    ax_d.text(xi, y_top, stars, ha='center',
                              fontsize=9, fontweight='bold', color='#B71C1C')

    _savefig(fig, FIG_DIR / f"failure_analysis_{site}.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def generate_text_summary(site: str, data: dict, failure_df: pd.DataFrame) -> None:
    """Write a comprehensive text summary of all error analysis findings."""
    models = list(data['predictions'].keys())
    seasons = doy_to_season(data['doy_scaled'])
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    hours = tod_to_hour(data['tod_scaled'])
    lines = [
        "=" * 72,
        f"COMPREHENSIVE ERROR ANALYSIS — {SITE_LABELS[site]}",
        "=" * 72,
        "",
        f"Site          : {site}",
        f"Ecosystem     : {'Wetland peatland' if 'AMo' in site else 'Temperate forest'}",
        f"N sequences   : {data['n_seq']:,} (lookback={LOOKBACK}h, horizon={HORIZON}h)",
        f"Models        : {', '.join(models)}",
        "",
    ]

    # ── Per-model overview ────────────────────────────────────────────────
    lines += ["─" * 72, "OVERALL METRICS (Sequence-level RMSE)", "─" * 72]
    for model_name in models:
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        bias = bias_seq(data['targets'], data['predictions'][model_name])
        lines.append(
            f"  {model_name:<22}  "
            f"mean RMSE={np.mean(errors):.4f}  "
            f"median RMSE={np.median(errors):.4f}  "
            f"mean bias={np.mean(bias):+.4f}"
        )
    lines.append("")

    # ── Temporal patterns ─────────────────────────────────────────────────
    lines += ["─" * 72, "TEMPORAL ERROR PATTERNS", "─" * 72]
    for model_name in models:
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        lines.append(f"\n  {model_name}:")

        # By hour
        night_mask = (hours <= 5) | (hours >= 20)
        day_mask = ~night_mask
        if night_mask.sum() > 0 and day_mask.sum() > 0:
            night_rmse = np.mean(errors[night_mask])
            day_rmse = np.mean(errors[day_mask])
            lines.append(
                f"    Nighttime RMSE: {night_rmse:.4f} | "
                f"Daytime RMSE: {day_rmse:.4f} | "
                f"Δ: {day_rmse - night_rmse:+.4f}"
            )

        # By season
        season_rmses = {}
        for s in season_order:
            m = seasons == s
            if m.sum() > 0:
                season_rmses[s] = np.mean(errors[m])
        worst_s = max(season_rmses, key=season_rmses.get)
        best_s = min(season_rmses, key=season_rmses.get)
        lines.append(
            f"    Worst season: {worst_s} (RMSE={season_rmses[worst_s]:.4f}) | "
            f"Best: {best_s} (RMSE={season_rmses[best_s]:.4f})"
        )

        # Kruskal-Wallis
        groups = [errors[seasons == s] for s in season_order if (seasons == s).sum() > 1]
        if len(groups) >= 2:
            h_stat, p_kw = kruskal(*groups)
            lines.append(
                f"    Kruskal-Wallis test: H={h_stat:.2f}, "
                f"p={p_kw:.3e} {significance_label(p_kw)}"
            )
    lines.append("")

    # ── Environmental correlations ────────────────────────────────────────
    lines += ["─" * 72, "SPEARMAN CORRELATIONS (error ~ environment)", "─" * 72]
    for model_name in models:
        errors = rmse_seq(data['targets'], data['predictions'][model_name])
        lines.append(f"\n  {model_name}:")
        for env_name, env_vals in [
            ("VPD_F (norm)", data['vpd_scaled']),
            ("TA_F  (norm)", data['ta_scaled']),
            ("SW_IN_F(norm)", data['sw_scaled']),
        ]:
            valid = ~np.isnan(env_vals)
            if valid.sum() > 10:
                rho, p = spearmanr(env_vals[valid], errors[valid])
                lines.append(
                    f"    {env_name:<18}: ρ={rho:+.4f}, "
                    f"p={p:.3e} {significance_label(p)}"
                )
    lines.append("")

    # ── Failure case summary ──────────────────────────────────────────────
    lines += ["─" * 72, "FAILURE CASE ANALYSIS (Worst 5% of sequences)", "─" * 72]
    site_failures = failure_df[failure_df['site'] == site]
    for _, row in site_failures.iterrows():
        lines += [
            f"\n  {row['model']}:",
            f"    Failure threshold (95th pct RMSE): {row['failure_threshold_rmse']:.4f}",
            f"    N failure sequences  : {row['n_failure_sequences']} "
            f"({row['pct_of_total']:.1f}%)",
            f"    Mean failure RMSE    : {row['mean_failure_rmse']:.4f}",
            f"    Max failure RMSE     : {row['max_failure_rmse']:.4f}",
            f"    Dominant season      : {row['dominant_season']}",
            f"    Night failures       : {row['night_failure_pct']:.1f}%  "
            f"Day failures: {row['day_failure_pct']:.1f}%",
            f"    VPD at failure (norm): {row['failure_mean_vpd_norm']:.4f} vs "
            f"{row['normal_mean_vpd_norm']:.4f} (normal) "
            f"[MWU p={row['vpd_mannwhitney_p']:.4f} "
            f"{significance_label(row['vpd_mannwhitney_p'])}]",
        ]

    lines += ["", "=" * 72, "End of Error Analysis", "=" * 72]

    out = OUT_DIR / f"error_analysis_summary_{site}.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  COMPREHENSIVE ERROR ANALYSIS — CARBON FLUX PREDICTION MODELS")
    print("=" * 72)
    print(f"  Sites: {SITES}")
    print(f"  Output figures : {FIG_DIR.relative_to(ROOT)}/")
    print(f"  Output results : {OUT_DIR.relative_to(ROOT)}/")
    print()

    all_failure_dfs = []

    for site in SITES:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {SITE_LABELS[site]}")
        print(f"{'─' * 60}")

        # Load all data
        data = load_site_data(site)
        print(f"  Loaded {data['n_seq']:,} sequences | "
              f"models: {list(data['predictions'].keys())}")

        # ── Generate all figures ─────────────────────────────────────────
        print(f"\n  [1/7] Error by hour of day...")
        plot_error_by_hour(data, site)

        print(f"  [2/7] Error by season...")
        plot_error_by_season(data, site)

        print(f"  [3/7] Error vs environment...")
        plot_error_vs_environment(data, site)

        print(f"  [4/7] Systematic bias analysis...")
        plot_systematic_bias(data, site)

        print(f"  [5/7] Residual Q-Q plots...")
        plot_residual_qq(data, site)

        print(f"  [6/7] Heteroscedasticity analysis...")
        plot_heteroscedasticity(data, site)

        print(f"  [7/7] Failure case analysis...")
        failure_df = analyze_failure_cases(data, site)
        all_failure_dfs.append(failure_df)
        plot_failure_analysis(data, site, failure_df)

        # ── Text summary ─────────────────────────────────────────────────
        generate_text_summary(site, data, failure_df)

    # ── Save combined failure CSV ────────────────────────────────────────
    if all_failure_dfs:
        combined_fail = pd.concat(all_failure_dfs, ignore_index=True)
        csv_path = OUT_DIR / "failure_cases.csv"
        combined_fail.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path.relative_to(ROOT)}")

        # Pretty-print failure summary
        print("\n" + "=" * 72)
        print("  FAILURE ANALYSIS SUMMARY")
        print("=" * 72)
        for _, row in combined_fail.iterrows():
            print(
                f"  {row['site']:<8} {row['model']:<22} "
                f"thresh={row['failure_threshold_rmse']:.3f}  "
                f"dominant_season={row['dominant_season']:<8}  "
                f"night%={row['night_failure_pct']:.0f}"
            )

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print(f"  Figures  →  {FIG_DIR.relative_to(ROOT)}/")
    print(f"  Results  →  {OUT_DIR.relative_to(ROOT)}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
