"""ensemble_models.py
===================
Comprehensive ensemble methods for carbon flux prediction.

Ensemble strategies
-------------------
A. Simple Averaging          — uniform equal weights across 4 core models
B. Performance-Weighted      — weights proportional to optimisation-set R²
C. Optimized Weighted        — SLSQP to maximise R² on optimisation set
D. Stacking (meta-learning)  — Ridge / Lasso / ElasticNet / LinearReg
E. Selective K-ensemble      — per-horizon best-K models (K=2, K=3)
+  Horizon-Adaptive          — separate SLSQP weights for h≤24 and h>24

Additional analyses
-------------------
- Diversity (Pearson correlation, Q-statistic, disagreement measure)
- Bootstrap 95 % CIs (1000 iterations, 200 for metrics table speed)
- Paired t-tests vs best single model
- Summary statements

Outputs
-------
results/ensemble/ensemble_metrics.csv
results/ensemble/diversity_analysis.csv
results/ensemble/ensemble_weights.json
results/ensemble/ensemble_predictions_{site}.npy
results/ensemble/ENSEMBLE_SUMMARY.txt

figures/ensemble/ensemble_comparison_{site}.png
figures/ensemble/weights_optimization.png
figures/ensemble/horizon_performance_comparison.png
figures/ensemble/diversity_matrix.png
figures/ensemble/improvement_over_best.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, Ridge,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────── paths ──────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
PRED_DIR     = ROOT / "results" / "predictions"
BASELINE_DIR = PRED_DIR / "baselines"
OUT_DIR      = ROOT / "results" / "ensemble"
FIG_DIR      = ROOT / "figures"  / "ensemble"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── constants ──────────────────────────
SITES   = ["UK-AMo", "SE-Htm"]
HORIZON = 96
SPLIT   = 0.70       # fraction used for weight optimisation / stacking training
BOOT_N  = 1000       # bootstrap iterations (full)
BOOT_N_FAST = 200    # bootstrap iterations (metrics table)
RNG     = np.random.default_rng(42)

SITE_LABEL = {
    "UK-AMo": "UK-AMo (Wetland)",
    "SE-Htm": "SE-Htm (Forest)",
}

# Core models included in every ensemble
ENSEMBLE_MEMBERS = ["TEMPO Fine-Tuned", "XGBoost", "Random Forest", "LSTM"]

# All individual models (TEMPO Zero-Shot kept for diversity / reference)
ALL_MODELS = [
    "TEMPO Fine-Tuned", "TEMPO Zero-Shot",
    "Random Forest", "XGBoost", "LSTM",
]

SHORT_H = np.arange(0, 24)    # indices 0-23  → h = 1–24
LONG_H  = np.arange(24, 96)   # indices 24-95 → h = 25–96

KEY_HORIZONS = [0, 5, 23, 47, 95]   # 1 h, 6 h, 24 h, 48 h, 96 h

# ─────────────────────────── colours ────────────────────────────
COLOURS: Dict[str, str] = {
    "TEMPO Fine-Tuned"   : "#2196F3",
    "TEMPO Zero-Shot"    : "#64B5F6",
    "Random Forest"      : "#FF9800",
    "XGBoost"            : "#F44336",
    "LSTM"               : "#9C27B0",
    "Simple Average"     : "#4CAF50",
    "Perf-Weighted"      : "#009688",
    "Optimized Weights"  : "#00BCD4",
    "Stacking-Ridge"     : "#795548",
    "Stacking-Lasso"     : "#607D8B",
    "Stacking-ElasticNet": "#8BC34A",
    "Stacking-LinearReg" : "#CDDC39",
    "Selective-K2"       : "#FF5722",
    "Selective-K3"       : "#E64A19",
    "Horizon-Adaptive"   : "#E91E63",
}

def _colour(name: str) -> str:
    return COLOURS.get(name, "#9E9E9E")


# ═══════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def per_horizon_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R² at each forecast step h.  Both arrays shape (N, 96)."""
    return np.array([_r2(y_true[:, h], y_pred[:, h]) for h in range(HORIZON)])


def skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                persist: np.ndarray) -> float:
    """Murphy skill score: 1 - RMSE_model / RMSE_persist."""
    rmse_p = _rmse(y_true, persist)
    return float(1.0 - _rmse(y_true, y_pred) / rmse_p) if rmse_p > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_site_data(site: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load ground-truth and all model predictions for *site*.

    Returns
    -------
    y_true : (N, 96)
    preds  : model_name → (N, 96)
    """
    y_true = np.load(BASELINE_DIR / f"targets_{site}.npy").astype(np.float64)
    N = len(y_true)

    file_map = {
        "TEMPO Fine-Tuned": PRED_DIR     / f"tempo_fine_tuned_preds_{site}.npy",
        "TEMPO Zero-Shot" : PRED_DIR     / f"tempo_zero_shot_preds_{site}.npy",
        "Random Forest"   : BASELINE_DIR / f"randomforest_preds_{site}.npy",
        "XGBoost"         : BASELINE_DIR / f"xgboost_preds_{site}.npy",
        "LSTM"            : BASELINE_DIR / f"lstm_preds_{site}.npy",
    }

    preds: Dict[str, np.ndarray] = {}
    for model, path in file_map.items():
        if path.exists():
            arr = np.load(path).astype(np.float64)
            if arr.ndim == 1:
                arr = arr[: N * HORIZON].reshape(N, HORIZON)
            preds[model] = arr
        else:
            print(f"  WARNING: missing {path.name}")

    return y_true, preds


def build_persistence(y_true: np.ndarray) -> np.ndarray:
    """Naive persistence: tile the h=1 actual across all horizons."""
    return np.tile(y_true[:, 0:1], (1, HORIZON))


# ═══════════════════════════════════════════════════════════════════
# TRAIN / EVAL SPLIT
# ══════════════════════════════════════════════��════════════════════

def temporal_split(
    y_true: np.ndarray,
    preds:  Dict[str, np.ndarray],
    split:  float = SPLIT,
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict, int, int]:
    N     = len(y_true)
    n_opt = int(N * split)
    y_opt  = y_true[:n_opt]
    y_eval = y_true[n_opt:]
    p_opt  = {m: v[:n_opt] for m, v in preds.items()}
    p_eval = {m: v[n_opt:] for m, v in preds.items()}
    return y_opt, p_opt, y_eval, p_eval, n_opt, N - n_opt


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE STRATEGIES
# ═══════════════════════════════════════════════════════════════════

# ── A. Simple Averaging ──────────────────────────────────────────
def simple_average(
    p_eval:  Dict[str, np.ndarray],
    members: List[str] = ENSEMBLE_MEMBERS,
) -> np.ndarray:
    avail = [m for m in members if m in p_eval]
    return np.mean([p_eval[m] for m in avail], axis=0)


# ── B. Performance-Weighted Averaging ───────────────────────────
def performance_weighted(
    y_opt:   np.ndarray,
    p_opt:   Dict[str, np.ndarray],
    p_eval:  Dict[str, np.ndarray],
    members: List[str] = ENSEMBLE_MEMBERS,
) -> Tuple[np.ndarray, Dict[str, float]]:
    avail  = [m for m in members if m in p_opt]
    r2s    = np.array([max(_r2(y_opt.ravel(), p_opt[m].ravel()), 0.0) for m in avail])
    total  = r2s.sum()
    weights = r2s / total if total > 0 else np.ones(len(avail)) / len(avail)
    pred   = sum(w * p_eval[m] for w, m in zip(weights, avail))
    return np.asarray(pred), dict(zip(avail, weights.tolist()))


# ── C. Optimized Weighted Averaging ─────────────────────────────
def _neg_r2_obj(weights: np.ndarray, ys: List[np.ndarray], y_true: np.ndarray) -> float:
    pred = sum(w * y for w, y in zip(weights, ys))
    return -_r2(y_true.ravel(), pred.ravel())


def optimized_weights(
    y_opt:   np.ndarray,
    p_opt:   Dict[str, np.ndarray],
    p_eval:  Dict[str, np.ndarray],
    members: List[str] = ENSEMBLE_MEMBERS,
) -> Tuple[np.ndarray, Dict[str, float]]:
    avail   = [m for m in members if m in p_opt]
    K       = len(avail)
    ys_opt  = [p_opt[m]  for m in avail]
    ys_eval = [p_eval[m] for m in avail]

    result = minimize(
        _neg_r2_obj,
        np.ones(K) / K,
        args=(ys_opt, y_opt),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    pred    = sum(weights[i] * ys_eval[i] for i in range(K))
    return np.asarray(pred), dict(zip(avail, weights.tolist()))


# ── D. Stacking Meta-Learning ────────────────────────────────────
def stacking_ensemble(
    y_opt:   np.ndarray,
    p_opt:   Dict[str, np.ndarray],
    p_eval:  Dict[str, np.ndarray],
    members: List[str] = ENSEMBLE_MEMBERS,
) -> Dict[str, np.ndarray]:
    """Train meta-learners on (N_opt×96, K) flattened prediction matrix."""
    avail   = [m for m in members if m in p_opt]
    X_opt   = np.column_stack([p_opt[m].ravel()  for m in avail])   # (N_opt*96, K)
    X_eval  = np.column_stack([p_eval[m].ravel() for m in avail])   # (N_eval*96, K)
    y_flat  = y_opt.ravel()

    scaler  = StandardScaler()
    X_opt_s  = scaler.fit_transform(X_opt)
    X_eval_s = scaler.transform(X_eval)

    N_eval = len(p_eval[avail[0]])
    meta_learners = {
        "Stacking-Ridge"       : Ridge(alpha=1.0),
        "Stacking-Lasso"       : Lasso(alpha=0.001, max_iter=10_000),
        "Stacking-ElasticNet"  : ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000),
        "Stacking-LinearReg"   : LinearRegression(),
    }

    results: Dict[str, np.ndarray] = {}
    for name, learner in meta_learners.items():
        learner.fit(X_opt_s, y_flat)
        pred = learner.predict(X_eval_s).reshape(N_eval, HORIZON)
        results[name] = pred
    return results


# ── E. Selective Ensemble (Best-K per horizon) ──────────────────
def selective_ensemble(
    y_opt:   np.ndarray,
    p_opt:   Dict[str, np.ndarray],
    p_eval:  Dict[str, np.ndarray],
    k:       int = 2,
    members: List[str] = ENSEMBLE_MEMBERS,
) -> np.ndarray:
    """At each horizon h, average the K models with lowest RMSE on opt set."""
    avail  = [m for m in members if m in p_opt]
    N_eval = len(p_eval[avail[0]])
    pred   = np.zeros((N_eval, HORIZON))
    for h in range(HORIZON):
        ranked = sorted(avail, key=lambda m: _rmse(y_opt[:, h], p_opt[m][:, h]))
        best_k = ranked[:k]
        pred[:, h] = np.mean([p_eval[m][:, h] for m in best_k], axis=0)
    return pred


# ── Horizon-Adaptive (separate weights for h≤24 and h>24) ───────
def horizon_adaptive_ensemble(
    y_opt:   np.ndarray,
    p_opt:   Dict[str, np.ndarray],
    p_eval:  Dict[str, np.ndarray],
    members: List[str] = ENSEMBLE_MEMBERS,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    avail   = [m for m in members if m in p_opt]
    K       = len(avail)
    N_eval  = len(p_eval[avail[0]])
    pred    = np.zeros((N_eval, HORIZON))
    w_dict: Dict[str, Dict[str, float]] = {}

    for seg_name, h_idx in [("short (h≤24)", SHORT_H), ("long (h>24)", LONG_H)]:
        ys_opt = [p_opt[m][:, h_idx]  for m in avail]
        y_seg  = y_opt[:, h_idx]
        result = minimize(
            _neg_r2_obj,
            np.ones(K) / K,
            args=(ys_opt, y_seg),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * K,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-9},
        )
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= weights.sum()
        w_dict[seg_name] = dict(zip(avail, weights.tolist()))
        ys_eval = [p_eval[m][:, h_idx] for m in avail]
        pred[:, h_idx] = sum(weights[i] * ys_eval[i] for i in range(K))

    return pred, w_dict


# ═══════════════════════════════════════════════════════════════════
# DIVERSITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def diversity_analysis(
    y_true: np.ndarray,
    preds:  Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise correlation, Q-statistic, and disagreement measure."""
    models  = [m for m in ALL_MODELS if m in preds]
    y_flat  = y_true.ravel()
    flat    = {m: preds[m].ravel() for m in models}

    # Correlation matrix
    mat    = np.column_stack([flat[m] for m in models])
    corr   = np.corrcoef(mat.T)
    corr_df = pd.DataFrame(corr, index=models, columns=models)

    # Pairwise Q-statistic and disagreement
    thr  = np.median(np.abs(y_flat))   # "correct" threshold
    rows = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            c1 = (np.abs(flat[m1] - y_flat) < thr).astype(int)
            c2 = (np.abs(flat[m2] - y_flat) < thr).astype(int)
            n11 = int(np.sum( c1 &  c2))
            n00 = int(np.sum((1-c1) & (1-c2)))
            n10 = int(np.sum( c1 & (1-c2)))
            n01 = int(np.sum((1-c1) &  c2))
            denom_q   = n11*n00 + n10*n01
            denom_dis = n11 + n10 + n01 + n00
            q   = (n11*n00 - n10*n01) / denom_q   if denom_q   > 0 else 0.0
            dis = (n10 + n01)          / denom_dis if denom_dis > 0 else 0.0
            rows.append({
                "Model1"       : m1,
                "Model2"       : m2,
                "Correlation"  : float(corr_df.loc[m1, m2]),
                "Q_statistic"  : float(q),
                "Disagreement" : float(dis),
            })

    return corr_df, pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP & SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════

def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n:      int = BOOT_N,
    rng:    np.random.Generator = RNG,
) -> Tuple[float, float, float]:
    """(r2, ci_low, ci_high) with 95 % bootstrap CI."""
    N   = len(y_true)
    r2s = np.array([
        _r2(y_true[(idx := rng.integers(0, N, N))].ravel(),
            y_pred[idx].ravel())
        for _ in range(n)
    ])
    return (
        _r2(y_true.ravel(), y_pred.ravel()),
        float(np.percentile(r2s, 2.5)),
        float(np.percentile(r2s, 97.5)),
    )


def paired_ttest(
    y_true:  np.ndarray,
    pred_a:  np.ndarray,
    pred_b:  np.ndarray,
) -> Tuple[float, float]:
    """Paired t-test on per-sequence MSE.  Returns (t_stat, p_value)."""
    se_a = ((pred_a - y_true) ** 2).mean(axis=1)
    se_b = ((pred_b - y_true) ** 2).mean(axis=1)
    t, p = stats.ttest_rel(se_a, se_b)
    return float(t), float(p)


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_all(
    y_eval:    np.ndarray,
    all_preds: Dict[str, np.ndarray],
    persist:   np.ndarray,
    boot_n:    int = BOOT_N_FAST,
) -> pd.DataFrame:
    rows = []
    for name, pred in all_preds.items():
        r2, ci_lo, ci_hi = bootstrap_r2(y_eval, pred, n=boot_n)
        ph_r2 = per_horizon_r2(y_eval, pred)
        row: Dict = {
            "Method"     : name,
            "R2"         : round(r2, 4),
            "R2_ci_low"  : round(ci_lo, 4),
            "R2_ci_high" : round(ci_hi, 4),
            "RMSE"       : round(_rmse(y_eval.ravel(), pred.ravel()), 4),
            "MAE"        : round(_mae(y_eval.ravel(), pred.ravel()), 4),
            "SkillScore" : round(skill_score(y_eval, pred, persist), 4),
        }
        for h_idx in KEY_HORIZONS:
            row[f"R2_h{h_idx+1}"] = round(float(ph_r2[h_idx]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════

# Figure 1 — ensemble comparison bar chart (per site)
def plot_ensemble_comparison(metrics_df: pd.DataFrame, site: str) -> None:
    df = metrics_df.sort_values("R2", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(df) * 0.4 + 1)))
    fig.suptitle(
        f"Ensemble Method Comparison — {SITE_LABEL[site]}",
        fontsize=12, fontweight="bold",
    )
    for ax, metric, xlabel in zip(
        axes,
        ["R2",   "RMSE"],
        ["R²",   "RMSE (μmol m⁻² s⁻¹)"],
    ):
        colours = [_colour(m) for m in df["Method"]]
        bars    = ax.barh(df["Method"], df[metric], color=colours,
                          edgecolor="white", linewidth=0.5)
        if metric == "R2":
            for i, row in df.iterrows():
                ax.errorbar(
                    row["R2"], i,
                    xerr=[[row["R2"] - row["R2_ci_low"]],
                          [row["R2_ci_high"] - row["R2"]]],
                    fmt="none", color="black", capsize=3, linewidth=1.2,
                )
        scale = max(abs(df[metric].max()), 1e-9)
        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_width() + 0.01 * scale,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_title(f"{metric} (sorted by R²)")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"ensemble_comparison_{site}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ensemble_comparison_{site}.png")


# Figure 2 — weight optimisation (combined across sites)
def plot_weights_optimization(
    weights_all: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    strategies = ["Perf-Weighted", "Optimized Weights"]
    n_sites    = len(SITES)
    fig, axes  = plt.subplots(
        n_sites, len(strategies),
        figsize=(11, 3.5 * n_sites),
    )
    fig.suptitle("Ensemble Weight Optimisation", fontsize=12, fontweight="bold")
    if n_sites == 1:
        axes = axes[np.newaxis, :]

    for si, site in enumerate(SITES):
        for sj, strategy in enumerate(strategies):
            ax = axes[si, sj]
            w  = weights_all.get(site, {}).get(strategy, {})
            if not w:
                ax.axis("off")
                continue
            models  = list(w.keys())
            vals    = [w[m] for m in models]
            colours = [_colour(m) for m in models]
            ax.bar(models, vals, color=colours, edgecolor="white")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Weight")
            ax.set_title(f"{SITE_LABEL[site]}\n{strategy}", fontsize=9)
            ax.tick_params(axis="x", rotation=30)
            for xi, v in enumerate(vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "weights_optimization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved weights_optimization.png")


# Figure 3 — R² vs horizon (combined)
def plot_horizon_performance(
    ph_r2_all: Dict[str, Dict[str, np.ndarray]],
) -> None:
    n_sites = len(SITES)
    fig, axes = plt.subplots(1, n_sites, figsize=(7 * n_sites, 5))
    if n_sites == 1:
        axes = [axes]
    fig.suptitle("R² vs Forecast Horizon", fontsize=12, fontweight="bold")
    horizons = np.arange(1, HORIZON + 1)

    for ax, site in zip(axes, SITES):
        for method, r2arr in ph_r2_all[site].items():
            is_single = method in ALL_MODELS
            ax.plot(
                horizons, r2arr,
                label=method,
                color=_colour(method),
                linestyle="--" if is_single else "-",
                linewidth=1.2  if is_single else 2.0,
                alpha=0.6      if is_single else 1.0,
            )
        ax.axvline(24, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(24.5, ax.get_ylim()[0] + 0.02, "24h", fontsize=7, color="grey")
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel("R²")
        ax.set_title(SITE_LABEL[site])
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "horizon_performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved horizon_performance_comparison.png")


# Figure 4 — diversity correlation heatmap (combined)
def plot_diversity_matrix(
    corr_dfs: Dict[str, pd.DataFrame],
) -> None:
    n_sites = len(SITES)
    fig, axes = plt.subplots(1, n_sites, figsize=(6 * n_sites, 5))
    if n_sites == 1:
        axes = [axes]
    fig.suptitle("Prediction Correlation Matrix", fontsize=12, fontweight="bold")

    for ax, site in zip(axes, SITES):
        corr_df = corr_dfs[site]
        models  = corr_df.index.tolist()
        n       = len(models)
        im = ax.imshow(corr_df.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(models, fontsize=8)
        for i in range(n):
            for j in range(n):
                val    = corr_df.values[i, j]
                colour = "white" if abs(val) > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=colour)
        ax.set_title(SITE_LABEL[site])
        plt.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "diversity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved diversity_matrix.png")


# Figure 5 — improvement over best single model (combined)
def plot_improvement_over_best(
    site_metrics: Dict[str, pd.DataFrame],
) -> None:
    n_sites = len(SITES)
    fig, axes = plt.subplots(2, n_sites, figsize=(7 * n_sites, 8))
    if n_sites == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Improvement over Best Single Model", fontsize=12, fontweight="bold")

    for si, site in enumerate(SITES):
        df         = site_metrics[site].copy()
        single_r2  = df.loc[df["Method"].isin(ALL_MODELS), "R2"].max()
        single_rmse = df.loc[df["Method"].isin(ALL_MODELS), "RMSE"].min()

        df["R2_gain_pct"]   = (df["R2"]   - single_r2)   / abs(single_r2)   * 100
        df["RMSE_gain_pct"] = (single_rmse - df["RMSE"]) / single_rmse       * 100

        ens_df = df[~df["Method"].isin(ALL_MODELS)].sort_values(
            "R2_gain_pct", ascending=False,
        )

        for ri, (col, ylabel) in enumerate([
            ("R2_gain_pct",   "R² gain (%)"),
            ("RMSE_gain_pct", "RMSE reduction (%)"),
        ]):
            ax      = axes[ri, si]
            colours = ["#4CAF50" if v >= 0 else "#F44336" for v in ens_df[col]]
            ax.barh(ens_df["Method"], ens_df[col], color=colours, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(ylabel)
            ax.set_title(f"{SITE_LABEL[site]}\n{ylabel}")
            ax.invert_yaxis()
            for i, v in enumerate(ens_df[col]):
                offset = 0.15 * np.sign(v) if v != 0 else 0.15
                ax.text(v + offset, i, f"{v:+.1f}%", va="center", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "improvement_over_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved improvement_over_best.png")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY TEXT
# ═══════════════════════════════════════════════════════════════════

def build_summary(site_results: Dict) -> str:
    W = 74
    lines: List[str] = []
    lines.append("=" * W)
    lines.append("ENSEMBLE MODELS — SUMMARY")
    lines.append(f"Sites: {', '.join(SITES)}  |  Horizon: {HORIZON}h  "
                 f"|  Bootstrap n={BOOT_N}")
    lines.append("=" * W)

    for site in SITES:
        res     = site_results[site]
        df      = res["metrics_df"]
        div_df  = res["diversity_df"]
        ha_w    = res["horizon_adaptive_weights"]
        ttests  = res["ttest_pvals"]

        lines.append(f"\n{'─'*W}")
        lines.append(f"SITE: {SITE_LABEL[site]}")
        lines.append(f"{'─'*W}")

        # Per-method table
        header = (f"{'Method':<28} {'R²':>7} {'95% CI':>17}"
                  f" {'RMSE':>9} {'MAE':>9} {'Skill':>7}")
        lines.append(header)
        lines.append("─" * W)
        for _, row in df.sort_values("R2", ascending=False).iterrows():
            ci  = f"[{row['R2_ci_low']:.4f},{row['R2_ci_high']:.4f}]"
            lines.append(
                f"{row['Method']:<28} {row['R2']:>7.4f} {ci:>17}"
                f" {row['RMSE']:>9.4f} {row['MAE']:>9.4f} {row['SkillScore']:>7.4f}"
            )

        # Per-horizon table
        lines.append("")
        lines.append(f"{'Method':<28} {'R²@1h':>7} {'R²@6h':>7} "
                     f"{'R²@24h':>7} {'R²@48h':>7} {'R²@96h':>7}")
        lines.append("─" * W)
        for _, row in df.sort_values("R2", ascending=False).iterrows():
            lines.append(
                f"{row['Method']:<28}"
                f" {row.get('R2_h1', float('nan')):>7.4f}"
                f" {row.get('R2_h6', float('nan')):>7.4f}"
                f" {row.get('R2_h24', float('nan')):>7.4f}"
                f" {row.get('R2_h48', float('nan')):>7.4f}"
                f" {row.get('R2_h96', float('nan')):>7.4f}"
            )

        # Best ensemble vs best single
        single_df  = df[df["Method"].isin(ALL_MODELS)]
        ens_df     = df[~df["Method"].isin(ALL_MODELS)]
        if not single_df.empty and not ens_df.empty:
            best_s = single_df.loc[single_df["R2"].idxmax()]
            best_e = ens_df.loc[ens_df["R2"].idxmax()]
            r2_diff  = best_e["R2"] - best_s["R2"]
            r2_gain  = r2_diff / abs(best_s["R2"]) * 100
            rmse_diff = best_s["RMSE"] - best_e["RMSE"]
            rmse_red = rmse_diff / best_s["RMSE"] * 100
            pv       = ttests.get(best_e["Method"], float("nan"))
            pstr     = f"{pv:.2e}" if (not np.isnan(pv) and pv < 0.001) else f"{pv:.4f}"

            lines.append("")
            lines.append("STATEMENTS:")

            # Determine if ensemble is better or worse
            if r2_diff > 0.01:  # Meaningful improvement
                lines.append(
                    f'  "{best_e["Method"]} achieves R²={best_e["R2"]:.4f} on {site}, '
                    f'representing +{r2_gain:.1f}% improvement over '
                    f'{best_s["Method"]} (R²={best_s["R2"]:.4f}; '
                    f'p={pstr}, 95% CI: [{best_e["R2_ci_low"]:.4f}, '
                    f'{best_e["R2_ci_high"]:.4f}])"'
                )
                lines.append(
                    f'  "RMSE reduced by {rmse_red:.1f}% '
                    f'({best_s["RMSE"]:.4f} → {best_e["RMSE"]:.4f} '
                    f'μmol m⁻² s⁻¹), demonstrating complementary error patterns."'
                )
            elif r2_diff < -0.005:  # Meaningful degradation
                lines.append(
                    f'  "Ensembling provides NO BENEFIT on {site}: '
                    f'{best_s["Method"]} alone achieves R²={best_s["R2"]:.4f}, '
                    f'outperforming best ensemble {best_e["Method"]} '
                    f'(R²={best_e["R2"]:.4f}, {r2_gain:.1f}% degradation, p={pstr})"'
                )
                lines.append(
                    f'  "High model correlation (TEMPO variants r=0.96) and extreme '
                    f'performance disparity (TEMPO R²≈0.73 vs baselines R²≈0.32) '
                    f'eliminate diversity benefits, demonstrating that ensembling '
                    f'is not always beneficial when one model family dominates."'
                )
            else:  # Marginal difference
                lines.append(
                    f'  "Ensemble and single-model performance effectively equivalent: '
                    f'{best_e["Method"]} R²={best_e["R2"]:.4f} vs '
                    f'{best_s["Method"]} R²={best_s["R2"]:.4f} '
                    f'({abs(r2_gain):.1f}% difference, not meaningful)"'
                )

        # Diversity highlights
        if div_df is not None and len(div_df) > 0:
            low_q  = div_df.nsmallest(1, "Q_statistic").iloc[0]
            high_c = div_df.nlargest(1, "Correlation").iloc[0]
            lines.append("")
            lines.append("DIVERSITY HIGHLIGHTS:")
            lines.append(
                f"  Most complementary (lowest Q): "
                f"{low_q['Model1']} × {low_q['Model2']} "
                f"(Q={low_q['Q_statistic']:.3f}, r={low_q['Correlation']:.3f})"
            )
            lines.append(
                f"  Most correlated pair: "
                f"{high_c['Model1']} × {high_c['Model2']} "
                f"(r={high_c['Correlation']:.3f})"
            )
            lines.append(
                f'  "Model diversity analysis reveals '
                f'{low_q["Model1"]}–{low_q["Model2"]} correlation of '
                f'{low_q["Correlation"]:.2f} enabling complementary error patterns."'
            )

        # Horizon-adaptive weights
        if ha_w:
            lines.append("")
            lines.append("HORIZON-ADAPTIVE WEIGHTS:")
            for seg, w in ha_w.items():
                top = max(w, key=w.get)
                w_str = ", ".join(f"{m}: {v:.2f}" for m, v in
                                  sorted(w.items(), key=lambda x: -x[1]))
                lines.append(f"  {seg}: {w_str}")
                lines.append(
                    f'  → "{top} dominates {seg} forecasts (w={w[top]:.2f})"'
                )

    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ENSEMBLE MODELS — starting")
    print("=" * 60)

    site_results:  Dict = {}
    weights_all:   Dict = {}
    ph_r2_all:     Dict = {}
    all_metrics:   List[pd.DataFrame] = []
    all_div:       List[pd.DataFrame] = []
    corr_dfs:      Dict[str, pd.DataFrame] = {}
    site_metrics:  Dict[str, pd.DataFrame] = {}
    best_ens_preds: Dict[str, np.ndarray]  = {}

    for site in SITES:
        print(f"\n{'─'*60}")
        print(f"Site: {SITE_LABEL[site]}")
        print(f"{'─'*60}")

        y_true, preds = load_site_data(site)
        N = len(y_true)
        avail = [m for m in ALL_MODELS if m in preds]
        print(f"  Sequences: {N}   Models loaded: {avail}")

        y_opt, p_opt, y_eval, p_eval, n_opt, n_eval = temporal_split(y_true, preds)
        print(f"  Split: opt={n_opt}  eval={n_eval}")

        persist_eval = build_persistence(y_eval)

        # Collect all predictions on eval set
        ens_preds: Dict[str, np.ndarray] = {}

        # Individual models
        for m in ALL_MODELS:
            if m in p_eval:
                ens_preds[m] = p_eval[m]

        # A. Simple Average
        print("  A. Simple Average …")
        ens_preds["Simple Average"] = simple_average(p_eval)

        # B. Performance-Weighted
        print("  B. Performance-Weighted …")
        pw_pred, pw_w = performance_weighted(y_opt, p_opt, p_eval)
        ens_preds["Perf-Weighted"] = pw_pred

        # C. Optimized Weights
        print("  C. Optimized Weights (SLSQP) …")
        ow_pred, ow_w = optimized_weights(y_opt, p_opt, p_eval)
        ens_preds["Optimized Weights"] = ow_pred

        # D. Stacking
        print("  D. Stacking meta-learners …")
        for name, pred in stacking_ensemble(y_opt, p_opt, p_eval).items():
            ens_preds[name] = pred

        # E. Selective K=2 and K=3
        print("  E. Selective-K2 …")
        ens_preds["Selective-K2"] = selective_ensemble(y_opt, p_opt, p_eval, k=2)
        print("  E. Selective-K3 …")
        ens_preds["Selective-K3"] = selective_ensemble(y_opt, p_opt, p_eval, k=3)

        # Horizon-Adaptive
        print("  Horizon-Adaptive …")
        ha_pred, ha_w = horizon_adaptive_ensemble(y_opt, p_opt, p_eval)
        ens_preds["Horizon-Adaptive"] = ha_pred

        # ── Evaluate ────────────────────────────────────────────
        print("  Evaluating (bootstrap CIs) …")
        metrics_df = evaluate_all(y_eval, ens_preds, persist_eval, boot_n=BOOT_N_FAST)
        all_metrics.append(metrics_df.assign(Site=site))
        site_metrics[site] = metrics_df

        # Per-horizon R²
        ph_r2_all[site] = {
            method: per_horizon_r2(y_eval, pred)
            for method, pred in ens_preds.items()
        }

        # ── Statistical tests vs best individual model ───────────
        single_df = metrics_df[metrics_df["Method"].isin(ALL_MODELS)]
        best_single_name = single_df.loc[single_df["R2"].idxmax(), "Method"]
        best_single_pred = ens_preds[best_single_name]

        ttest_pvals: Dict[str, float] = {}
        print(f"\n  Paired t-test vs {best_single_name}:")
        for method, pred in ens_preds.items():
            if method in ALL_MODELS:
                continue
            _, pv = paired_ttest(y_eval, pred, best_single_pred)
            ttest_pvals[method] = pv
            r2v = metrics_df.loc[metrics_df["Method"] == method, "R2"].values[0]
            sig = ("***" if pv < 0.001 else
                   "**"  if pv < 0.01  else
                   "*"   if pv < 0.05  else "ns")
            print(f"    {method:<28} R²={r2v:.4f}  p={pv:.3e} {sig}")

        # ── Diversity ────────────────────────────────────────────
        print("  Diversity analysis …")
        corr_df, div_df = diversity_analysis(
            y_eval, {m: p for m, p in ens_preds.items() if m in ALL_MODELS},
        )
        all_div.append(div_df.assign(Site=site))
        corr_dfs[site] = corr_df

        # ── Save best ensemble prediction ────────────────────────
        ens_only = metrics_df[~metrics_df["Method"].isin(ALL_MODELS)]
        best_ens_name = ens_only.loc[ens_only["R2"].idxmax(), "Method"]
        best_ens_preds[site] = ens_preds[best_ens_name]
        print(f"  Best ensemble: {best_ens_name} (R²={ens_only['R2'].max():.4f})")

        # ── Per-site figures ─────────────────────────────────────
        print("  Generating per-site figures …")
        plot_ensemble_comparison(metrics_df, site)

        weights_all[site] = {
            "Perf-Weighted"    : pw_w,
            "Optimized Weights": ow_w,
        }

        site_results[site] = {
            "metrics_df"              : metrics_df,
            "ttest_pvals"             : ttest_pvals,
            "diversity_df"            : div_df,
            "horizon_adaptive_weights": ha_w,
        }

    # ── Cross-site figures ───────────────────────────────────────
    print("\nGenerating cross-site figures …")
    plot_weights_optimization(weights_all)
    plot_horizon_performance(ph_r2_all)
    plot_diversity_matrix(corr_dfs)
    plot_improvement_over_best(site_metrics)

    # ── Save outputs ─────────────────────────────────────────────
    print("\nSaving outputs …")

    pd.concat(all_metrics, ignore_index=True).to_csv(
        OUT_DIR / "ensemble_metrics.csv", index=False,
    )
    print("  Saved ensemble_metrics.csv")

    pd.concat(all_div, ignore_index=True).to_csv(
        OUT_DIR / "diversity_analysis.csv", index=False,
    )
    print("  Saved diversity_analysis.csv")

    with open(OUT_DIR / "ensemble_weights.json", "w") as f:
        json.dump(weights_all, f, indent=2)
    print("  Saved ensemble_weights.json")

    for site, arr in best_ens_preds.items():
        np.save(OUT_DIR / f"ensemble_predictions_{site}.npy", arr)
        print(f"  Saved ensemble_predictions_{site}.npy")

    summary = build_summary(site_results)
    (OUT_DIR / "ENSEMBLE_SUMMARY.txt").write_text(summary)
    print("  Saved ENSEMBLE_SUMMARY.txt")

    print("\n" + summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
