"""
Horizon-Specific Forecasting Analysis for Carbon Flux Prediction
=================================================================
Evaluates how prediction quality degrades with forecast horizon (1–96 hours)
across all trained models at both test sites (UK-AMo, SE-Htm).

Analysis components
-------------------
1. Per-horizon metrics  — RMSE, MAE, R², Skill Score at every step 1–96
2. Persistence baseline — naive forecast: y_pred[h] = last observed NEE
3. Threshold analysis   — horizons where R² drops below 0.6 / 0.5 / 0.3
4. Degradation rates    — linear slope of R² decline (R²/hour)
5. Effective horizon    — last horizon with Skill Score > 0
6. TEMPO advantage      — relative RMSE improvement vs best baseline per horizon

Outputs
-------
  results/horizon_analysis/horizon_metrics.csv          (full per-horizon table)
  results/horizon_analysis/degradation_summary.csv      (degradation rates)
  results/horizon_analysis/threshold_analysis.csv       (R² threshold crossings)
  results/horizon_analysis/HORIZON_ANALYSIS_SUMMARY.txt
  figures/horizon_analysis/rmse_vs_horizon_{site}.png
  figures/horizon_analysis/r2_vs_horizon_{site}.png
  figures/horizon_analysis/skill_score_vs_horizon_{site}.png
  figures/horizon_analysis/model_comparison_by_horizon.png
  figures/horizon_analysis/degradation_rate_comparison.png

Notes
-----
  All predictions are loaded from pre-saved .npy files — no model inference
  is re-run.  Windows are 1-step sliding, so the persistence baseline is
  constructed as persist[i, :] = y[i-1, 0]  (constant = last observed NEE).
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
PRED_DIR    = ROOT / "results" / "predictions"
BASELINE_DIR= PRED_DIR / "baselines"
OUT_DIR     = ROOT / "results" / "horizon_analysis"
FIG_DIR     = ROOT / "figures" / "horizon_analysis"

SITES       = ["UK-AMo", "SE-Htm"]
HORIZON     = 96
SEED        = 42

# Key horizons to report in tables and annotations
KEY_HORIZONS = [1, 3, 6, 12, 24, 48, 72, 96]

# ---------------------------------------------------------------------------
# Model registry: display name → (pred_dir, file_stem)
# ---------------------------------------------------------------------------
MODELS = {
    "Random Forest"    : (BASELINE_DIR,  "randomforest_preds"),
    "XGBoost"          : (BASELINE_DIR,  "xgboost_preds"),
    "LSTM"             : (BASELINE_DIR,  "lstm_preds"),
    "TEMPO Zero-Shot"  : (PRED_DIR,      "tempo_zero_shot_preds"),
    "TEMPO Fine-Tuned" : (PRED_DIR,      "tempo_fine_tuned_preds"),
}

COLORS = {
    "Random Forest"    : "#4CAF50",
    "XGBoost"          : "#FF5722",
    "LSTM"             : "#9C27B0",
    "TEMPO Zero-Shot"  : "#03A9F4",
    "TEMPO Fine-Tuned" : "#2196F3",
    "Persistence"      : "#9E9E9E",
}
LSTYLES = {
    "Random Forest"    : "-",
    "XGBoost"          : "--",
    "LSTM"             : "-.",
    "TEMPO Zero-Shot"  : ":",
    "TEMPO Fine-Tuned" : "-",
    "Persistence"      : ":",
}
LWIDTHS = {m: 2.0 for m in MODELS}
LWIDTHS["TEMPO Fine-Tuned"] = 2.5
LWIDTHS["Persistence"] = 1.2

SITE_NAMES = {"UK-AMo": "UK-AMo (Wetland)", "SE-Htm": "SE-Htm (Forest)"}

# Threshold levels for R² analysis
R2_THRESHOLDS = [0.6, 0.5, 0.3]

# Baseline models (for TEMPO advantage comparison)
BASELINE_MODELS = ["Random Forest", "XGBoost", "LSTM"]
TEMPO_MODELS    = ["TEMPO Zero-Shot", "TEMPO Fine-Tuned"]

NEE_UNIT = r"NEE ($\mu$mol m$^{-2}$ s$^{-1}$)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_site_data(site: str) -> dict:
    """
    Load ground truth, all model predictions, and build the persistence baseline.

    Persistence baseline: predict[i, :] = y[i-1, 0] for i > 0 (constant forecast
    equal to the last observed NEE value, which is y[i-1, 0] by the 1-step
    sliding-window property: y[i-1, 1] == y[i, 0]).
    """
    y_true = np.load(BASELINE_DIR / f"targets_{site}.npy")       # (N, 96)

    preds = {}
    for model, (d, stem) in MODELS.items():
        path = d / f"{stem}_{site}.npy"
        if path.exists():
            preds[model] = np.load(path)                          # (N, 96)

    # Persistence: constant = last observed NEE = y[i-1, 0]
    N = len(y_true)
    persist = np.empty((N, HORIZON), dtype=np.float32)
    persist[0] = y_true[0, 0]          # first window: no prior; use y[0,0]
    persist[1:] = y_true[:-1, 0:1]    # broadcast last observed to all horizons
    preds["Persistence"] = persist

    # Validate TEMPO Zero-Shot predictions
    if "TEMPO Zero-Shot" in preds:
        tz_pred = preds["TEMPO Zero-Shot"]
        # Check for NaN/inf
        if np.any(np.isnan(tz_pred)) or np.any(np.isinf(tz_pred)):
            print(f"  WARNING: TEMPO Zero-Shot {site} has {np.isnan(tz_pred).sum()} NaN values")
        # Check variance across horizon
        var_per_h = tz_pred.var(axis=0)
        if np.any(var_per_h < 1e-6):
            print(f"  WARNING: TEMPO Zero-Shot {site} has near-zero variance at some horizons")

    return {"y_true": y_true, "preds": preds}


# ---------------------------------------------------------------------------
# Per-horizon metrics
# ---------------------------------------------------------------------------
def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R² at every forecast step h = 1 … HORIZON.

    Returns dict with arrays of shape (HORIZON,).
    """
    N, H = y_true.shape
    rmse = np.zeros(H)
    mae  = np.zeros(H)
    r2   = np.zeros(H)

    for h in range(H):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        diff = yt - yp
        rmse[h] = np.sqrt(np.mean(diff ** 2))
        mae[h]  = np.mean(np.abs(diff))
        ss_res  = np.sum(diff ** 2)
        ss_tot  = np.sum((yt - yt.mean()) ** 2)
        r2[h]   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def skill_score(rmse_model: np.ndarray, rmse_persistence: np.ndarray) -> np.ndarray:
    """SS = 1 - RMSE_model / RMSE_persistence  (positive = better than persistence)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = 1.0 - np.where(rmse_persistence > 0,
                             rmse_model / rmse_persistence,
                             np.nan)
    return ss


# ---------------------------------------------------------------------------
# Degradation analysis
# ---------------------------------------------------------------------------
def degradation_rate(r2_array: np.ndarray, is_persistence: bool = False) -> tuple:
    """
    Fit a linear trend to R²(h) over all HORIZON steps.

    For persistence baseline, R² should decline as forecast horizon increases.
    If getting positive slope for persistence, likely indicates metric calculation issue.

    Returns (slope, intercept, r_value, p_value, std_err).
    Slope < 0 means degradation (R² falls with horizon).
    """
    h = np.arange(1, HORIZON + 1, dtype=float)
    # Mask NaN
    mask = ~np.isnan(r2_array)
    if mask.sum() < 4:
        return (np.nan,) * 5

    result = stats.linregress(h[mask], r2_array[mask])

    # Diagnostic: warn if persistence shows positive slope
    if is_persistence and result.slope > 0:
        print(f"    WARNING: Persistence shows positive slope ({result.slope:.6f})")
        print(f"    R² range: {r2_array[mask].min():.4f} to {r2_array[mask].max():.4f}")

    return result


def half_life(r2_array: np.ndarray) -> float:
    """
    Horizon at which R² drops to half of its initial value (h=1).
    Returns NaN if R² never reaches half, or never starts above zero.
    """
    r2_init = r2_array[0]
    if np.isnan(r2_init) or r2_init <= 0:
        return np.nan
    target = 0.5 * r2_init
    crossings = np.where(r2_array <= target)[0]
    return float(crossings[0] + 1) if len(crossings) > 0 else np.nan


def threshold_crossing(r2_array: np.ndarray, threshold: float) -> float:
    """First horizon (1-indexed) where R² drops below threshold; NaN if never."""
    crossings = np.where(r2_array < threshold)[0]
    return float(crossings[0] + 1) if len(crossings) > 0 else np.nan


def effective_horizon(ss_array: np.ndarray) -> float:
    """Last horizon (1-indexed) where Skill Score > 0; NaN if never positive."""
    positives = np.where(ss_array > 0)[0]
    return float(positives[-1] + 1) if len(positives) > 0 else np.nan


# ---------------------------------------------------------------------------
# Utility: TEMPO advantage over best baseline
# ---------------------------------------------------------------------------
def tempo_advantage_pct(rmse_tempo: np.ndarray,
                          rmse_baselines: dict) -> np.ndarray:
    """
    At each horizon: (best_baseline_RMSE - TEMPO_RMSE) / best_baseline_RMSE × 100.
    Positive = TEMPO better.
    """
    best_bl = np.stack([rmse_baselines[m] for m in BASELINE_MODELS
                         if m in rmse_baselines]).min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(best_bl > 0,
                         (best_bl - rmse_tempo) / best_bl * 100.0,
                         np.nan)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
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


def _legend_handles(models):
    return [Line2D([0], [0],
                    color=COLORS[m], lw=LWIDTHS.get(m, 1.8),
                    ls=LSTYLES.get(m, "-"), label=m)
            for m in models]


def _annotate_thresholds(ax, y_levels: list, labels: list, xmax=96):
    for lv, lb in zip(y_levels, labels):
        ax.axhline(lv, color="black", lw=0.8, ls=":", alpha=0.55)
        ax.text(xmax * 0.98, lv + 0.01, lb,
                ha="right", va="bottom", fontsize=7.5, alpha=0.7)


def _mark_key_horizons(ax, metric_arrays: dict, key_h: list, y_lim=None):
    """Light vertical lines at key horizon steps."""
    for h in key_h:
        ax.axvline(h, color="lightgray", lw=0.6, ls="--", zorder=0)


# ---------------------------------------------------------------------------
# Figure 1: RMSE vs horizon (per site)
# ---------------------------------------------------------------------------
def plot_rmse_vs_horizon(site: str, metrics_per_model: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    hours   = np.arange(1, HORIZON + 1)

    for model, m in metrics_per_model.items():
        if model == "Persistence":
            continue  # Skip persistence - RMSE is misleading for periodic data
        ax.plot(hours, m["RMSE"],
                color=COLORS[model], lw=LWIDTHS.get(model, 1.8),
                ls=LSTYLES.get(model, "-"), alpha=0.9)

    _mark_key_horizons(ax, metrics_per_model, KEY_HORIZONS)

    for h in KEY_HORIZONS:
        ax.axvline(h, color="lightgray", lw=0.6, ls="--", zorder=0)

    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel(r"RMSE ($\mu$mol m$^{-2}$ s$^{-1}$)")
    ax.set_title(f"RMSE vs Forecast Horizon — {SITE_NAMES[site]}")
    ax.set_xlim(1, HORIZON)
    ax.legend(handles=_legend_handles([m for m in metrics_per_model if m != "Persistence"]),
               fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: R² vs horizon (per site)
# ---------------------------------------------------------------------------
def plot_r2_vs_horizon(site: str, metrics_per_model: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    hours   = np.arange(1, HORIZON + 1)

    for model, m in metrics_per_model.items():
        if model == "Persistence":
            continue            # persistence R² is often negative / uninformative
        ax.plot(hours, m["R2"],
                color=COLORS[model], lw=LWIDTHS.get(model, 1.8),
                ls=LSTYLES.get(model, "-"), alpha=0.9)

    _annotate_thresholds(ax,
                          y_levels=[0.6, 0.5, 0.3],
                          labels=["R²=0.6", "R²=0.5", "R²=0.3"])
    ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.4)

    for h in KEY_HORIZONS:
        ax.axvline(h, color="lightgray", lw=0.6, ls="--", zorder=0)

    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel("R²")
    ax.set_title(f"R² vs Forecast Horizon — {SITE_NAMES[site]}")
    ax.set_xlim(1, HORIZON)
    ax.legend(handles=_legend_handles([m for m in metrics_per_model if m != "Persistence"]),
               fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Skill Score vs horizon (per site)
# ---------------------------------------------------------------------------
def plot_skill_score(site: str, ss_per_model: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    hours   = np.arange(1, HORIZON + 1)

    ax.axhline(0, color="black", lw=1.0, ls="-", alpha=0.5,
                label="Persistence baseline (SS = 0)")
    ax.axhspan(ymin=-5, ymax=0, alpha=0.04, color="red")

    for model, ss in ss_per_model.items():
        if model == "Persistence":
            continue
        ax.plot(hours, ss,
                color=COLORS[model], lw=LWIDTHS.get(model, 1.8),
                ls=LSTYLES.get(model, "-"), alpha=0.9)

    for h in KEY_HORIZONS:
        ax.axvline(h, color="lightgray", lw=0.6, ls="--", zorder=0)

    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel("Skill Score  (SS = 1 − RMSE/RMSE_persist)")
    ax.set_title(f"Skill Score vs Forecast Horizon — {SITE_NAMES[site]}")
    ax.set_xlim(1, HORIZON)
    handles = [Line2D([0], [0], color="black", lw=1.0, ls="-",
                       label="Persistence (SS = 0)")]
    handles += _legend_handles([m for m in ss_per_model if m != "Persistence"])
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Multi-panel comparison (RMSE + R² for both sites)
# ---------------------------------------------------------------------------
def plot_model_comparison(all_metrics: dict) -> plt.Figure:
    """
    2×2 grid:  (site row) × (RMSE | R²)
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)
    hours = np.arange(1, HORIZON + 1)

    # Build one shared legend
    all_models = list(next(iter(all_metrics.values())).keys())

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel_idx = 0
    for row, site in enumerate(SITES):
        for col, (metric, ylabel, ylbl) in enumerate([
            ("RMSE", r"RMSE ($\mu$mol m$^{-2}$ s$^{-1}$)", "RMSE"),
            ("R2",   "R²",                                   "R²"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            for model, m in all_metrics[site].items():
                if model == "Persistence":
                    continue
                ax.plot(hours, m[metric],
                         color=COLORS[model], lw=LWIDTHS.get(model, 1.8),
                         ls=LSTYLES.get(model, "-"), alpha=0.9)
            for h in KEY_HORIZONS:
                ax.axvline(h, color="lightgray", lw=0.6, ls="--", zorder=0)
            if metric == "R2":
                ax.axhline(0, color="gray", lw=0.8, alpha=0.4)
                _annotate_thresholds(ax, [0.6, 0.5], ["R²=0.6", "R²=0.5"])
            ax.set_xlabel("Forecast Horizon (h)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{panel_labels[panel_idx]} {SITE_NAMES[site]} — {ylbl}")
            ax.set_xlim(1, HORIZON)
            ax.grid(True, alpha=0.25)
            panel_idx += 1

    # Shared legend below the subplots
    legend_models = [m for m in all_models if m != "Persistence"]
    handles = _legend_handles(legend_models)
    fig.legend(handles=handles, loc="lower center",
                ncol=len(handles), fontsize=9,
                bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Model Comparison: RMSE and R² Across Forecast Horizons",
                  fontsize=13, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Degradation rate comparison
# ---------------------------------------------------------------------------
def plot_degradation_rates(degrad_df: pd.DataFrame) -> plt.Figure:
    """
    Two panels: slope of R² decline (left) and half-life horizon (right).
    Grouped by site.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sites   = SITES
    models  = [m for m in MODELS if m in degrad_df["Model"].unique()]
    n_m     = len(models)
    x       = np.arange(len(sites))
    bw      = 0.8 / n_m

    # Panel A: degradation slope (more negative = faster degradation)
    ax = axes[0]
    for k, model in enumerate(models):
        sub = degrad_df[degrad_df["Model"] == model]
        slopes = [sub[sub["Site"] == s]["Slope"].values[0]
                   if len(sub[sub["Site"] == s]) else np.nan
                   for s in sites]
        ax.bar(x + (k - n_m / 2 + 0.5) * bw, slopes,
                width=bw * 0.9, color=COLORS[model], alpha=0.85, label=model)
    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(SITE_NAMES.values()), fontsize=9)
    ax.set_ylabel("R² Degradation Rate (R²/hour, slope < 0 = worsening)")
    ax.set_title("(a) R² Degradation Rate")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: half-life
    ax = axes[1]
    for k, model in enumerate(models):
        sub = degrad_df[degrad_df["Model"] == model]
        hl = [sub[sub["Site"] == s]["HalfLife_h"].values[0]
               if len(sub[sub["Site"] == s]) else np.nan
               for s in sites]
        ax.bar(x + (k - n_m / 2 + 0.5) * bw, hl,
                width=bw * 0.9, color=COLORS[model], alpha=0.85, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(list(SITE_NAMES.values()), fontsize=9)
    ax.set_ylabel("R² Half-Life (h)  [horizon where R² = 0.5 × R²_h1]")
    ax.set_title("(b) R² Half-Life by Model and Site")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Forecast Quality Degradation Analysis",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Build DataFrames
# ---------------------------------------------------------------------------
def build_horizon_metrics_df(all_metrics: dict) -> pd.DataFrame:
    rows = []
    hours = np.arange(1, HORIZON + 1)
    for site, model_metrics in all_metrics.items():
        for model, m in model_metrics.items():
            for h_idx, h in enumerate(hours):
                rows.append({
                    "Site"   : site,
                    "Model"  : model,
                    "Horizon": h,
                    "RMSE"   : round(float(m["RMSE"][h_idx]), 5),
                    "MAE"    : round(float(m["MAE"][h_idx]),  5),
                    "R2"     : round(float(m["R2"][h_idx]),   5),
                })
    return pd.DataFrame(rows)


def build_degradation_df(all_metrics: dict) -> pd.DataFrame:
    rows = []
    for site, model_metrics in all_metrics.items():
        for model, m in model_metrics.items():
            if model == "Persistence":
                continue
            r2 = m["R2"]
            slope, intercept, rv, pv, se = degradation_rate(r2)
            hl  = half_life(r2)
            rows.append({
                "Site"           : site,
                "Model"          : model,
                "Slope"          : round(float(slope),     6) if not np.isnan(slope) else np.nan,
                "Intercept"      : round(float(intercept), 4) if not np.isnan(intercept) else np.nan,
                "R2_h1"          : round(float(r2[0]),     4),
                "R2_h96"         : round(float(r2[-1]),    4),
                "HalfLife_h"     : round(hl, 1) if not np.isnan(hl) else np.nan,
                "Slope_pvalue": (f"{pv:.2e}" if not np.isnan(pv) else np.nan),
            })
    return pd.DataFrame(rows)


def build_threshold_df(all_metrics: dict) -> pd.DataFrame:
    rows = []
    for site, model_metrics in all_metrics.items():
        for model, m in model_metrics.items():
            if model == "Persistence":
                continue
            r2 = m["R2"]
            row = {"Site": site, "Model": model}
            for thr in R2_THRESHOLDS:
                row[f"R2_below_{thr:.0%}_at_h"] = threshold_crossing(r2, thr)
            row["R2_at_h6"]  = round(float(m["R2"][5]),  4)
            row["R2_at_h24"] = round(float(m["R2"][23]), 4)
            row["R2_at_h48"] = round(float(m["R2"][47]), 4)
            row["R2_at_h96"] = round(float(m["R2"][95]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# summary text
# ---------------------------------------------------------------------------
def build_summary_text(hm_df: pd.DataFrame,
                        deg_df: pd.DataFrame,
                        thr_df: pd.DataFrame,
                        all_metrics: dict) -> str:
    lines = [
        "=" * 85,
        "HORIZON-SPECIFIC FORECASTING ANALYSIS — CARBON FLUX PREDICTION",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Horizon   : 1–{HORIZON} hours",
        f"Sites     : {', '.join(SITE_NAMES.values())}",
        "=" * 85,
        "",
        "── R² AT KEY HORIZONS ────────────────────────────────────────────────────────",
        "",
    ]

    # R² table at key horizons
    header = f"{'Model':<22} {'Site':<10}" + "".join(f" h={h:3d}h" for h in KEY_HORIZONS)
    lines += [header, "─" * len(header)]
    for _, row in thr_df.sort_values(["Site", "Model"]).iterrows():
        model_metrics = all_metrics[row["Site"]][row["Model"]]
        r2_vals = [model_metrics["R2"][h - 1] for h in KEY_HORIZONS]
        vals_str = "".join(f" {v:6.3f}" for v in r2_vals)
        lines.append(f"{row['Model']:<22} {row['Site']:<10}{vals_str}")
    lines.append("")

    # Threshold crossings
    lines += [
        "── THRESHOLD CROSSINGS (first horizon where R² < threshold) ─────────────────",
        "",
        f"{'Model':<22} {'Site':<10} {'R²<0.6 at h':>12} {'R²<0.5 at h':>12}"
        f" {'R²<0.3 at h':>12}",
        "─" * 70,
    ]
    for _, row in thr_df.sort_values(["Site", "Model"]).iterrows():
        def _fmt(v):
            return f"{v:>10.0f}h" if not (isinstance(v, float) and np.isnan(v)) else f"{'> 96h':>11}"
        lines.append(
            f"{row['Model']:<22} {row['Site']:<10}"
            f" {_fmt(row['R2_below_60%_at_h'])}"
            f" {_fmt(row['R2_below_50%_at_h'])}"
            f" {_fmt(row['R2_below_30%_at_h'])}"
        )
    lines.append("")

    # Degradation rates
    lines += [
        "── DEGRADATION RATES (linear fit: R²(h) = a·h + b) ─────────────────────────",
        "",
        f"{'Model':<22} {'Site':<10} {'Slope(R²/h)':>12} {'R²_h=1':>8}"
        f" {'R²_h=96':>8} {'Half-life(h)':>13}",
        "─" * 84,
    ]
    for _, row in deg_df.sort_values(["Site", "Model"]).iterrows():
        sl = f"{row['Slope']:.5f}" if not np.isnan(row["Slope"]) else "  N/A"
        hl = f"{row['HalfLife_h']:.1f}" if not np.isnan(row["HalfLife_h"]) else ">96"
        lines.append(
            f"{row['Model']:<22} {row['Site']:<10}"
            f" {sl:>12} {row['R2_h1']:>8.4f} {row['R2_h96']:>8.4f}"
            f" {hl:>13}"
        )
    lines.append("")

    # TEMPO advantage
    lines += [
        "── TEMPO FINE-TUNED ADVANTAGE vs BEST BASELINE (RMSE) ──────────────────────",
        "",
    ]
    for site in SITES:
        if "TEMPO Fine-Tuned" not in all_metrics[site]:
            continue
        rmse_bl  = {m: all_metrics[site][m]["RMSE"]
                     for m in BASELINE_MODELS if m in all_metrics[site]}
        rmse_ft  = all_metrics[site]["TEMPO Fine-Tuned"]["RMSE"]
        adv      = tempo_advantage_pct(rmse_ft, rmse_bl)
        lines.append(f"  {SITE_NAMES[site]}:")
        for h in KEY_HORIZONS:
            v = adv[h - 1]
            tag = "better" if v > 0 else "worse "
            lines.append(f"    h={h:3d}h:  {v:+.1f}%  ({tag} than best baseline)")
        lines.append("")

    # Narrative summary
    lines += ["── NARRATIVE STATEMENTS ────────────────────────────────────────", ""]

    for site in SITES:
        lines.append(f"  [{SITE_NAMES[site]}]")
        thr_site = thr_df[thr_df["Site"] == site].set_index("Model")
        deg_site = deg_df[deg_df["Site"] == site].set_index("Model")

        # Best model at h=24 and h=96
        r2_h24 = {m: all_metrics[site][m]["R2"][23]
                   for m in MODELS if m in all_metrics[site]}
        r2_h96 = {m: all_metrics[site][m]["R2"][95]
                   for m in MODELS if m in all_metrics[site]}
        best_h24 = max(r2_h24, key=r2_h24.get)
        best_h96 = max(r2_h96, key=r2_h96.get)
        lines.append(f"  - Best model at 24h: {best_h24} (R²={r2_h24[best_h24]:.3f})")
        lines.append(f"  - Best model at 96h: {best_h96} (R²={r2_h96[best_h96]:.3f})")

        # TEMPO threshold
        for tm in TEMPO_MODELS:
            if tm not in thr_site.index:
                continue
            h06 = thr_site.loc[tm, "R2_below_60%_at_h"]
            h05 = thr_site.loc[tm, "R2_below_50%_at_h"]
            h06s = f"{h06:.0f}h" if not np.isnan(h06) else ">96h"
            h05s = f"{h05:.0f}h" if not np.isnan(h05) else ">96h"
            lines.append(f"  - {tm} maintains R²>0.6 to {h06s},"
                           f" R²>0.5 to {h05s}")

        # Slowest-degrading baseline
        bl_rows = deg_site.loc[[m for m in BASELINE_MODELS
                                  if m in deg_site.index], "Slope"]
        if len(bl_rows):
            slowest = bl_rows.idxmax()   # least negative = slowest degradation
            fastest = bl_rows.idxmin()
            lines.append(
                f"  - Slowest-degrading baseline: {slowest}"
                f" ({bl_rows[slowest]:.5f} R²/h)")
            lines.append(
                f"  - Fastest-degrading baseline: {fastest}"
                f" ({bl_rows[fastest]:.5f} R²/h)")

        lines.append("")

    lines += ["=" * 85]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("HORIZON-SPECIFIC FORECASTING ANALYSIS")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Horizon   : 1–{HORIZON} hours")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ── [1] Load predictions ───────────────────────────────────────────────
    print("\n[1/4] Loading predictions and building persistence baseline...")
    all_data = {}
    for site in SITES:
        all_data[site] = load_site_data(site)
        loaded = list(all_data[site]["preds"].keys())
        print(f"  {site}: {len(loaded)} models loaded — {loaded}")

    # ── [2] Compute per-horizon metrics ────────────────────────────────────
    print("\n[2/4] Computing per-horizon metrics...")
    all_metrics: dict = {}
    all_ss: dict = {}

    for site in SITES:
        y_true = all_data[site]["y_true"]
        preds  = all_data[site]["preds"]
        all_metrics[site] = {}
        all_ss[site]      = {}

        persist_rmse = horizon_metrics(y_true, preds["Persistence"])["RMSE"]

        for model, y_pred in preds.items():
            m = horizon_metrics(y_true, y_pred)
            all_metrics[site][model] = m
            all_ss[site][model]      = skill_score(m["RMSE"], persist_rmse)

        # Print key horizon summary
        print(f"\n  {site} — R² at key horizons:")
        hdr = f"  {'Model':<22}" + "".join(f" h={h:2d}h" for h in KEY_HORIZONS)
        print(hdr)
        for model in list(MODELS.keys()) + ["Persistence"]:
            if model not in all_metrics[site]:
                continue
            r2_vals = [all_metrics[site][model]["R2"][h - 1] for h in KEY_HORIZONS]
            row = f"  {model:<22}" + "".join(f" {v:6.3f}" for v in r2_vals)
            print(row)

    # ── [DIAGNOSTIC] Non-monotonic R² check ───────────────────────────────
    print("\n[DIAGNOSTIC] Checking for non-monotonic R² patterns...")
    for site in SITES:
        for model in all_metrics[site]:
            if model == "Persistence":
                continue
            r2 = all_metrics[site][model]["R2"]
            # Find places where R² increases by more than 0.05 (shouldn't happen normally)
            increases = np.where(np.diff(r2) > 0.05)[0]
            if len(increases) > 0:
                print(f"  WARNING: {model} on {site} shows R² increases at horizons: {increases + 1}")
                for idx in increases[:3]:  # Show first 3
                    print(f"    h={idx+1}: R²={r2[idx]:.4f} → h={idx+2}: R²={r2[idx+1]:.4f}")

    # ── [3] Build output DataFrames ────────────────────────────────────────
    print("\n[3/4] Building summary tables...")
    hm_df  = build_horizon_metrics_df(all_metrics)
    deg_df = build_degradation_df(all_metrics)
    thr_df = build_threshold_df(all_metrics)

    hm_df.to_csv(OUT_DIR / "horizon_metrics.csv", index=False)
    deg_df.to_csv(OUT_DIR / "degradation_summary.csv", index=False)
    thr_df.to_csv(OUT_DIR / "threshold_analysis.csv", index=False)
    print(f"  Saved: {OUT_DIR.relative_to(ROOT)}/horizon_metrics.csv"
          f"  ({len(hm_df)} rows)")
    print(f"  Saved: {OUT_DIR.relative_to(ROOT)}/degradation_summary.csv")
    print(f"  Saved: {OUT_DIR.relative_to(ROOT)}/threshold_analysis.csv")

    summary = build_summary_text(hm_df, deg_df, thr_df, all_metrics)
    txt_path = OUT_DIR / "HORIZON_ANALYSIS_SUMMARY.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {txt_path.relative_to(ROOT)}")
    print("\n" + summary)

    # ── [4] Figures ────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")

    # Per-site RMSE, R², Skill Score
    for site in SITES:
        fig = plot_rmse_vs_horizon(site, all_metrics[site])
        p = FIG_DIR / f"rmse_vs_horizon_{site}.png"
        fig.savefig(p); plt.close(fig)
        print(f"  Saved: {p.name}")

        fig = plot_r2_vs_horizon(site, all_metrics[site])
        p = FIG_DIR / f"r2_vs_horizon_{site}.png"
        fig.savefig(p); plt.close(fig)
        print(f"  Saved: {p.name}")

        # Skill score figures removed — persistence R² is non-monotonic for
        # periodic data and misleads the linear-trend interpretation.

    # Multi-panel comparison (both sites)
    fig = plot_model_comparison(all_metrics)
    p = FIG_DIR / "model_comparison_by_horizon.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")

    # Degradation rate comparison
    fig = plot_degradation_rates(deg_df)
    p = FIG_DIR / "degradation_rate_comparison.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║  HORIZON ANALYSIS COMPLETE" + " " * 51 + "║")
    print("║  Metrics  : results/horizon_analysis/*.csv" + " " * 35 + "║")
    print("║  Summary  : results/horizon_analysis/HORIZON_ANALYSIS_SUMMARY.txt" + " " * 9 + "║")
    print("║  Figures  : figures/horizon_analysis/" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝")


if __name__ == "__main__":
    main()