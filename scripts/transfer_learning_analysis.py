"""
Transfer Learning Analysis for Carbon Flux Quantification
======================================================
Investigates cross-ecosystem and cross-site training effects on NEE prediction.

Data reality
------------
Training data (combined in train_X/y): 5 WETLAND sites
  FI-Lom  Subarctic minerotrophic fen (Finland)
  GL-ZaF  High Arctic tundra fen (Greenland)
  IE-Cra  Ombrotrophic raised bog (Ireland)
  DE-Akm  Rewetted coastal peatland (Germany)
  FR-LGt  Acidic fen (France)

Test data (separate held-out arrays): 2 sites
  UK-AMo  Blanket bog, UK           — SAME ecosystem [Wetland → Wetland]
  SE-Htm  Norway spruce, Sweden     — DIFFERENT ecosystem [Wetland → Forest]

Experiment groups
-----------------
  A. Data Coverage (5 configs)  — train on 20/40/60/80/100% of combined data
  B. Leave-One-Block-Out (5 configs) — exclude each approximate 1/5 block
     (5 equal blocks approximate the 5 training sites)

Models per experiment
  Random Forest — fast tree ensemble
  XGBoost       — gradient boosting
  LSTM          — sequential deep learning (capped at MAX_TRAIN_N for speed)

Reference benchmarks (pre-saved predictions, no re-training)
  TEMPO Zero-Shot    — foundation model, no adaptation
  TEMPO Fine-Tuned   — foundation model fine-tuned on all 5 wetland sites

Key hypotheses
--------------
  H1: More wetland training data improves UK-AMo (same ecosystem) more than SE-Htm
  H2: Individual wetland sites differ in how much they help forest transfer
  H3: TEMPO shows better cross-ecosystem generalisation than tree-based baselines
  H4: Wetland→Forest penalty is large and consistent across models

Outputs
-------
  results/transfer_learning/transfer_matrix.csv
  results/transfer_learning/negative_transfer_summary.csv
  results/transfer_learning/statistical_tests.csv
  results/transfer_learning/TRANSFER_LEARNING_SUMMARY.txt
  figures/transfer_learning/coverage_vs_performance.png
  figures/transfer_learning/transfer_matrix_heatmap.png
  figures/transfer_learning/loo_site_contribution.png
  figures/transfer_learning/model_comparison_transfer.png
  figures/transfer_learning/within_vs_cross_ecosystem.png
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / "data" / "processed"
PRED_DIR     = ROOT / "results" / "predictions"
BASELINE_DIR = PRED_DIR / "baselines"
OUT_DIR      = ROOT / "results" / "transfer_learning"
FIG_DIR      = ROOT / "figures" / "transfer_learning"

sys.path.insert(0, str(ROOT))
from models.lstm_baseline import LSTMForecaster

# ---------------------------------------------------------------------------
# Constants & metadata
# ---------------------------------------------------------------------------
TRAIN_SITES = ["FI-Lom", "GL-ZaF", "IE-Cra", "DE-Akm", "FR-LGt"]
TEST_SITES  = ["UK-AMo", "SE-Htm"]
N_TRAIN_SITES = len(TRAIN_SITES)

ECOSYSTEM_TYPE = {
    "FI-Lom": "Wetland (Subarctic fen)",
    "GL-ZaF": "Wetland (Tundra fen)",
    "IE-Cra": "Wetland (Raised bog)",
    "DE-Akm": "Wetland (Coastal peatland)",
    "FR-LGt": "Wetland (Acidic fen)",
    "UK-AMo": "Wetland (Blanket bog)",
    "SE-Htm": "Forest (Norway spruce)",
}

NAN_FEATURE_INDICES = [1, 4, 5, 6, 7]
HORIZON    = 96
INPUT_SIZE = 19
SEED       = 42

# Experiment settings
COVERAGE_FRACS = [0.20, 0.40, 0.60, 0.80, 1.00]
MAX_TRAIN_N    = 5000      # cap training samples for speed
BOOTSTRAP_N    = 100       # bootstrap iterations for 95% CI
LSTM_EPOCHS    = 15
LSTM_BATCH     = 128
LSTM_HIDDEN    = 64
LSTM_LR        = 5e-4

MODEL_COLORS = {
    "XGBoost"          : "#FF5722",
    "Random Forest"    : "#4CAF50",
    "LSTM"             : "#9C27B0",
    "TEMPO Fine-Tuned" : "#2196F3",
    "TEMPO Zero-Shot"  : "#03A9F4",
}
SITE_COLORS = {"UK-AMo": "#2196F3", "SE-Htm": "#E53935"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_medians() -> dict:
    """Load cross-site imputation medians from JSON (keys are feature indices)."""
    path = DATA_DIR / "cross_site_medians.json"
    with open(path) as f:
        raw = json.load(f)
    medians = {}
    for k, v in raw.items():
        try:
            medians[int(k)] = float(v)
        except (ValueError, TypeError):
            pass
    return medians


def impute(X: np.ndarray, medians: dict) -> np.ndarray:
    """Median imputation for known NaN-prone features."""
    X = X.copy().astype(np.float32)
    for idx in NAN_FEATURE_INDICES:
        col = X[:, :, idx]
        mask = np.isnan(col)
        col[mask] = medians.get(idx, 0.0)
        X[:, :, idx] = col
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def load_train() -> tuple:
    med = _load_medians()
    X = np.load(DATA_DIR / "train_X.npy").astype(np.float32)
    y = np.load(DATA_DIR / "train_y.npy").astype(np.float32)
    return impute(X, med), y


def load_test(site: str) -> tuple:
    med = _load_medians()
    X = np.load(DATA_DIR / f"test_{site}_X.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"test_{site}_y.npy").astype(np.float32)
    return impute(X, med), y


def load_tempo_preds(site: str) -> dict:
    """Load pre-saved TEMPO predictions for a test site."""
    return {
        "TEMPO Fine-Tuned": np.load(PRED_DIR / f"tempo_fine_tuned_preds_{site}.npy"),
        "TEMPO Zero-Shot" : np.load(PRED_DIR / f"tempo_zero_shot_preds_{site}.npy"),
    }


def subsample(X: np.ndarray, y: np.ndarray, n: int = MAX_TRAIN_N) -> tuple:
    """Random subsample to at most n rows (reproducible)."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def coverage_slice(X: np.ndarray, y: np.ndarray, frac: float) -> tuple:
    """Temporal first-fraction slice, preserving time order."""
    n = max(200, int(len(X) * frac))
    return X[:n], y[:n]


def loo_slice(X: np.ndarray, y: np.ndarray, exclude_block: int) -> tuple:
    """Remove the k-th equal block of rows (approximates leaving out one site)."""
    blocks = np.array_split(np.arange(len(X)), N_TRAIN_SITES)
    keep = np.concatenate([b for i, b in enumerate(blocks)
                            if i != exclude_block])
    return X[keep], y[keep]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = y_true.ravel(), y_pred.ravel()
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def horizon_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    H = y_true.shape[1]
    out = np.zeros(H)
    for h in range(H):
        yt, yp = y_true[:, h], y_pred[:, h]
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        out[h] = (1.0 - np.sum((yt - yp) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return out


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2h = horizon_r2(y_true, y_pred)
    return {
        "R2"    : r2_score(y_true, y_pred),
        "RMSE"  : rmse_score(y_true, y_pred),
        "R2_h1" : float(r2h[0]),
        "R2_h24": float(r2h[23]),
        "R2_h96": float(r2h[95]),
        "R2_mean": float(np.nanmean(r2h)),
    }


def bootstrap_r2_ci(y_true: np.ndarray, y_pred: np.ndarray,
                    n_boot: int = BOOTSTRAP_N) -> tuple:
    rng  = np.random.default_rng(SEED)
    vals = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(r2_score(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def flat(X: np.ndarray) -> np.ndarray:
    """(N, T, F) → (N, T·F) for sklearn models."""
    return X.reshape(len(X), -1)


def _get_device() -> torch.device:
    """Force CPU — MPS backend has LSTM hidden-state shape bugs on macOS."""
    return torch.device("cpu")


def train_rf(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    Xc, yc = subsample(flat(X), y)
    model = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_split=10,
        n_jobs=-1, random_state=SEED)
    model.fit(Xc, yc)
    return model


def predict_rf(model, X: np.ndarray) -> np.ndarray:
    return model.predict(flat(X))


def train_xgb(X: np.ndarray, y: np.ndarray):
    from xgboost import XGBRegressor
    Xc, yc = subsample(flat(X), y)
    model = XGBRegressor(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=SEED, verbosity=0)
    model.fit(Xc, yc)
    return model


def predict_xgb(model, X: np.ndarray) -> np.ndarray:
    return model.predict(flat(X))


def train_lstm_local(X: np.ndarray, y: np.ndarray) -> nn.Module:
    """Train a lightweight LSTM (capped at MAX_TRAIN_N samples, LSTM_EPOCHS epochs)."""
    device = _get_device()
    Xc, yc = subsample(X, y, MAX_TRAIN_N)

    val_n = max(1, int(0.1 * len(Xc)))
    Xtr, ytr = Xc[:-val_n], yc[:-val_n]
    Xvl, yvl = Xc[-val_n:], yc[-val_n:]

    tr_ld = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=LSTM_BATCH, shuffle=True)
    vl_ld = DataLoader(
        TensorDataset(torch.from_numpy(Xvl), torch.from_numpy(yvl)),
        batch_size=LSTM_BATCH)

    model = LSTMForecaster(
        input_size=INPUT_SIZE, hidden_size=LSTM_HIDDEN,
        num_layers=1, horizon=HORIZON).to(device)
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    crit = nn.MSELoss()

    best_val, best_state, no_imp, patience = np.inf, None, 0, 4

    for _ in range(LSTM_EPOCHS):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for Xb, yb in vl_ld:
                vl += crit(model(Xb.to(device)), yb.to(device)).item()
        vl /= max(1, len(vl_ld))

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval().cpu()


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = _get_device()
    model = model.to(device).eval()
    ld = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=512)
    preds = []
    with torch.no_grad():
        for (Xb,) in ld:
            preds.append(model(Xb.to(device)).cpu().numpy())
    model.cpu()
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
def build_configs() -> list:
    cfgs = []
    # Group A: data coverage
    for frac in COVERAGE_FRACS:
        pct = int(frac * 100)
        cfgs.append({
            "name" : f"A_cov_{pct:03d}pct",
            "group": "Coverage",
            "label": f"{pct}% data",
            "frac" : frac,
            "loo"  : None,
        })
    # Group B: leave-one-block-out
    for k, site in enumerate(TRAIN_SITES):
        cfgs.append({
            "name" : f"B_loo_{k}_{site.replace('-', '')}",
            "group": "LOO",
            "label": f"LOO: −{site}",
            "frac" : None,
            "loo"  : k,
        })
    return cfgs


def _get_train_slice(X, y, cfg):
    if cfg["loo"] is not None:
        return loo_slice(X, y, cfg["loo"])
    return coverage_slice(X, y, cfg["frac"])


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiments(X_train, y_train, test_data: dict,
                    tempo_preds: dict) -> pd.DataFrame:
    """
    Train RF, XGB, LSTM for each experiment config and evaluate on both test sites.
    Returns a long-format DataFrame.
    """
    configs = build_configs()
    rows    = []
    n_cfg   = len(configs)

    for ci, cfg in enumerate(configs):
        label = cfg["label"]
        print(f"  [{ci+1:02d}/{n_cfg}] {label:30s}", end="", flush=True)
        t0 = time.time()

        Xtr, ytr = _get_train_slice(X_train, y_train, cfg)
        n_eff = min(len(Xtr), MAX_TRAIN_N)

        # ── RF ──────────────────────────────────────────────────────────────
        rf = train_rf(Xtr, ytr)
        for site, (Xte, yte) in test_data.items():
            yp = predict_rf(rf, Xte)
            m  = full_metrics(yte, yp)
            lo, hi = bootstrap_r2_ci(yte, yp)
            rows.append({"Config": cfg["name"], "Group": cfg["group"],
                         "Label": label, "Model": "Random Forest",
                         "Site": site, "N_train": n_eff,
                         "CI_lo": lo, "CI_hi": hi, **m})

        # ── XGB ─────────────────────────────────────────────────────────────
        xgb = train_xgb(Xtr, ytr)
        for site, (Xte, yte) in test_data.items():
            yp = predict_xgb(xgb, Xte)
            m  = full_metrics(yte, yp)
            lo, hi = bootstrap_r2_ci(yte, yp)
            rows.append({"Config": cfg["name"], "Group": cfg["group"],
                         "Label": label, "Model": "XGBoost",
                         "Site": site, "N_train": n_eff,
                         "CI_lo": lo, "CI_hi": hi, **m})

        # ── LSTM ─────────────────────────────────────────────────────────────
        lstm = train_lstm_local(Xtr, ytr)
        for site, (Xte, yte) in test_data.items():
            yp = predict_lstm(lstm, Xte)
            m  = full_metrics(yte, yp)
            lo, hi = bootstrap_r2_ci(yte, yp)
            rows.append({"Config": cfg["name"], "Group": cfg["group"],
                         "Label": label, "Model": "LSTM",
                         "Site": site, "N_train": n_eff,
                         "CI_lo": lo, "CI_hi": hi, **m})

        print(f"  {time.time()-t0:.0f}s")

    # ── TEMPO reference (pre-saved, no training loop) ────────────────────────
    print("  TEMPO reference benchmarks (pre-saved)...", flush=True)
    for site, (_, yte) in test_data.items():
        for model_name, yp in tempo_preds[site].items():
            m  = full_metrics(yte, yp)
            lo, hi = bootstrap_r2_ci(yte, yp)
            rows.append({"Config": "TEMPO_ref", "Group": "TEMPO",
                         "Label": model_name, "Model": model_name,
                         "Site": site, "N_train": 23899,
                         "CI_lo": lo, "CI_hi": hi, **m})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transfer gain & LOO sensitivity
# ---------------------------------------------------------------------------
def compute_transfer_gain(df: pd.DataFrame) -> pd.DataFrame:
    """R² gain relative to 100% coverage baseline (A_cov_100pct)."""
    bl = (df[df["Config"] == "A_cov_100pct"]
          .set_index(["Model", "Site"])["R2"].to_dict())
    rows = []
    for _, r in df.iterrows():
        base = bl.get((r["Model"], r["Site"]), np.nan)
        gain = r["R2"] - base
        rows.append({
            "Config"         : r["Config"],
            "Group"          : r["Group"],
            "Label"          : r["Label"],
            "Model"          : r["Model"],
            "Site"           : r["Site"],
            "R2"             : r["R2"],
            "R2_baseline"    : base,
            "Transfer_Gain"  : gain,
            "Negative_Transfer": gain < -0.01,
        })
    return pd.DataFrame(rows)


def loo_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Delta-R² vs 100% baseline for each LOO experiment."""
    bl = (df[df["Config"] == "A_cov_100pct"]
          .set_index(["Model", "Site"])["R2"].to_dict())
    loo = df[df["Group"] == "LOO"].copy()
    loo["R2_baseline"] = loo.apply(
        lambda r: bl.get((r["Model"], r["Site"]), np.nan), axis=1)
    loo["R2_delta"] = loo["R2"] - loo["R2_baseline"]
    # Positive delta: removing that block IMPROVED performance
    # → that block was causing negative transfer
    return loo


def coverage_ttest(df: pd.DataFrame) -> pd.DataFrame:
    """95% CI overlap test: 20% vs 100% coverage, per model × site."""
    rows = []
    for model in df["Model"].unique():
        if model in ("TEMPO Fine-Tuned", "TEMPO Zero-Shot"):
            continue
        for site in TEST_SITES:
            sub = df[(df["Model"] == model) & (df["Site"] == site)]
            r20 = sub[sub["Config"] == "A_cov_020pct"]
            r100 = sub[sub["Config"] == "A_cov_100pct"]
            if len(r20) == 0 or len(r100) == 0:
                continue
            lo20, hi20 = r20[["CI_lo", "CI_hi"]].values[0]
            lo100, hi100 = r100[["CI_lo", "CI_hi"]].values[0]
            overlap = max(0.0, min(hi20, hi100) - max(lo20, lo100))
            rows.append({
                "Model"      : model,
                "Site"       : site,
                "R2_20pct"   : round(float(r20["R2"].values[0]), 4),
                "R2_100pct"  : round(float(r100["R2"].values[0]), 4),
                "Delta_R2"   : round(float(r100["R2"].values[0]) -
                                     float(r20["R2"].values[0]), 4),
                "CI_20"      : f"[{lo20:.3f}, {hi20:.3f}]",
                "CI_100"     : f"[{lo100:.3f}, {hi100:.3f}]",
                "CI_overlap" : round(overlap, 4),
                "Significant": overlap < 0.01,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.family"    : "serif",
        "font.size"      : 10,
        "axes.labelsize" : 11,
        "axes.titlesize" : 12,
        "figure.dpi"     : 150,
        "savefig.dpi"    : 300,
        "savefig.bbox"   : "tight",
    })


def _r2_for(df, model, site, config):
    sub = df[(df["Model"] == model) & (df["Site"] == site) &
             (df["Config"] == config)]
    return float(sub["R2"].values[0]) if len(sub) else np.nan


def _ci_for(df, model, site, config):
    sub = df[(df["Model"] == model) & (df["Site"] == site) &
             (df["Config"] == config)]
    if len(sub) == 0:
        return np.nan, np.nan
    return float(sub["CI_lo"].values[0]), float(sub["CI_hi"].values[0])


# ---------------------------------------------------------------------------
# Figure 1: Coverage vs Performance
# ---------------------------------------------------------------------------
def plot_coverage_performance(df: pd.DataFrame) -> plt.Figure:
    cov = df[df["Group"] == "Coverage"]
    configs = [f"A_cov_{int(f*100):03d}pct" for f in COVERAGE_FRACS]
    xlabels = [f"{int(f*100)}%" for f in COVERAGE_FRACS]
    x = np.arange(len(configs))

    models = ["Random Forest", "XGBoost", "LSTM"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ai, site in enumerate(TEST_SITES):
        ax = axes[ai]
        sdf = cov[cov["Site"] == site]

        for model in models:
            mdf = sdf[sdf["Model"] == model]
            r2v = [float(mdf[mdf["Config"] == c]["R2"].values[0])
                   if len(mdf[mdf["Config"] == c]) else np.nan for c in configs]
            lo  = [float(mdf[mdf["Config"] == c]["CI_lo"].values[0])
                   if len(mdf[mdf["Config"] == c]) else np.nan for c in configs]
            hi  = [float(mdf[mdf["Config"] == c]["CI_hi"].values[0])
                   if len(mdf[mdf["Config"] == c]) else np.nan for c in configs]

            ax.plot(x, r2v, "-o", color=MODEL_COLORS[model], lw=2, ms=6, label=model)
            ax.fill_between(x, lo, hi, color=MODEL_COLORS[model], alpha=0.12)

        # TEMPO reference lines
        tempo = df[(df["Group"] == "TEMPO") & (df["Site"] == site)]
        for tm in ["TEMPO Fine-Tuned", "TEMPO Zero-Shot"]:
            trow = tempo[tempo["Model"] == tm]
            if len(trow):
                ax.axhline(float(trow["R2"].values[0]),
                           color=MODEL_COLORS[tm], lw=1.5, ls="--",
                           label=f"{tm} (ref.)")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Training Data Coverage (temporal fraction)")
        ax.set_ylabel("Overall R²")
        eco = ECOSYSTEM_TYPE.get(site, site)
        ax.set_title(f"({'ab'[ai]}) {site}  [{eco}]")
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Effect of Training Data Coverage on Transfer Performance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Transfer matrix heatmap
# ---------------------------------------------------------------------------
def plot_transfer_matrix(df: pd.DataFrame) -> plt.Figure:
    models = ["Random Forest", "XGBoost", "LSTM"]
    cfgs   = build_configs()
    cfg_labels = [c["label"] for c in cfgs]
    cfg_names  = [c["name"]  for c in cfgs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for ai, model in enumerate(models):
        ax = axes[ai]
        mdf = df[df["Model"] == model]

        mat = np.full((len(cfg_names), len(TEST_SITES)), np.nan)
        for ri, cfg_name in enumerate(cfg_names):
            for ci, site in enumerate(TEST_SITES):
                sub = mdf[(mdf["Config"] == cfg_name) & (mdf["Site"] == site)]
                if len(sub):
                    mat[ri, ci] = float(sub["R2"].values[0])

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(TEST_SITES, fontsize=9)
        ax.set_yticks(range(len(cfg_labels)))
        ax.set_yticklabels(cfg_labels, fontsize=7.5)
        ax.set_title(f"({'abc'[ai]}) {model}", fontsize=10, fontweight="bold")

        # Annotate cells
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                v = mat[ri, ci]
                if not np.isnan(v):
                    colour = "white" if (v < 0.2 or v > 0.85) else "black"
                    ax.text(ci, ri, f"{v:.2f}",
                            ha="center", va="center", fontsize=7.5,
                            color=colour, fontweight="bold")

        # Separator between Coverage and LOO groups
        ax.axhline(len(COVERAGE_FRACS) - 0.5, color="white", lw=2)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.05, label="R²")

    fig.suptitle(
        "Transfer Matrix — R² by Training Configuration × Test Ecosystem\n"
        "Upper: Coverage (A); Lower: Leave-One-Out (B)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: LOO site contribution
# ---------------------------------------------------------------------------
def plot_loo_effects(loo_df: pd.DataFrame) -> plt.Figure:
    models = ["Random Forest", "XGBoost", "LSTM"]
    x      = np.arange(N_TRAIN_SITES)
    bw     = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ai, test_site in enumerate(TEST_SITES):
        ax   = axes[ai]
        sdf  = loo_df[loo_df["Site"] == test_site]

        for mi, model in enumerate(models):
            mdf    = sdf[sdf["Model"] == model]
            deltas = []
            for k, site in enumerate(TRAIN_SITES):
                tag = f"B_loo_{k}_{site.replace('-', '')}"
                sub = mdf[mdf["Config"] == tag]
                deltas.append(float(sub["R2_delta"].values[0]) if len(sub) else np.nan)

            offset = (mi - len(models) / 2 + 0.5) * bw
            ax.bar(x + offset, deltas, width=bw * 0.9,
                   color=MODEL_COLORS[model], alpha=0.85, label=model)

            # Label non-trivial changes
            for bi, d in enumerate(deltas):
                if not np.isnan(d) and abs(d) > 0.003:
                    va  = "bottom" if d >= 0 else "top"
                    off = 0.002 if d >= 0 else -0.004
                    ax.text(x[bi] + offset, d + off, f"{d:+.3f}",
                            ha="center", va=va, fontsize=6)

        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.axhspan(-2, 0, color="salmon",     alpha=0.06)
        ax.axhspan(0,  2, color="lightgreen", alpha=0.06)
        ax.set_xticks(x)
        ax.set_xticklabels([f"−{s}" for s in TRAIN_SITES],
                           rotation=25, ha="right", fontsize=8)
        ax.set_xlabel("Excluded Training Site (approx. block)")
        ax.set_ylabel("ΔR² vs 100% baseline\n(+) removing it helped  |  (−) removing it hurt")
        eco = ECOSYSTEM_TYPE.get(test_site, test_site)
        ax.set_title(f"({'ab'[ai]}) LOO Effect on {test_site} [{eco[:7]}]")
        ax.legend(fontsize=8, ncol=3, loc="best")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Leave-One-Site-Out Analysis: Which Training Sites Drive Transfer Performance?",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Model comparison at full training (100%)
# ---------------------------------------------------------------------------
def plot_model_comparison(df: pd.DataFrame) -> plt.Figure:
    ordered = ["Random Forest", "XGBoost", "LSTM",
               "TEMPO Zero-Shot", "TEMPO Fine-Tuned"]
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ai, site in enumerate(TEST_SITES):
        ax = axes[ai]
        r2v, lo, hi = [], [], []

        for model in ordered:
            cfg = "A_cov_100pct" if "TEMPO" not in model else "TEMPO_ref"
            sub = df[(df["Model"] == model) & (df["Site"] == site) &
                     (df["Config"] == cfg)]
            if len(sub):
                r2v.append(float(sub["R2"].values[0]))
                lo.append(float(sub["CI_lo"].values[0]))
                hi.append(float(sub["CI_hi"].values[0]))
            else:
                r2v.append(np.nan); lo.append(np.nan); hi.append(np.nan)

        r2a = np.array(r2v, dtype=float)
        elo = r2a - np.array(lo, dtype=float)
        ehi = np.array(hi, dtype=float) - r2a

        colors = [MODEL_COLORS.get(m, "#888") for m in ordered]
        ax.bar(x, r2a, color=colors, alpha=0.85)
        ax.errorbar(x, r2a, yerr=[np.where(np.isnan(elo), 0, elo),
                                   np.where(np.isnan(ehi), 0, ehi)],
                    fmt="none", color="black", capsize=4, lw=1.2)

        for xi, rv in enumerate(r2a):
            if not np.isnan(rv):
                ax.text(xi, rv + 0.015, f"{rv:.3f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in ordered], fontsize=8)
        ax.set_ylabel("Overall R²")
        ax.set_ylim(0, 1.08)
        eco = ECOSYSTEM_TYPE.get(site, site)
        ax.set_title(f"({'ab'[ai]}) {site} [{eco}]")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Model Comparison for Transfer Performance (Full Training Data)\n"
        "Error bars = 95% bootstrap CI",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Within vs cross-ecosystem scatter
# ---------------------------------------------------------------------------
def plot_within_vs_cross(df: pd.DataFrame) -> plt.Figure:
    models = ["Random Forest", "XGBoost", "LSTM"]
    fig, ax = plt.subplots(figsize=(8, 7))

    cov_loo = df[df["Group"].isin(["Coverage", "LOO"])]

    for model in models:
        mdf = cov_loo[cov_loo["Model"] == model]
        cfgs = mdf["Config"].unique()
        uk_pts, se_pts, labs = [], [], []
        for cfg in cfgs:
            cdf = mdf[mdf["Config"] == cfg]
            uk  = cdf[cdf["Site"] == "UK-AMo"]["R2"].values
            se  = cdf[cdf["Site"] == "SE-Htm"]["R2"].values
            if len(uk) and len(se):
                uk_pts.append(float(uk[0]))
                se_pts.append(float(se[0]))
                labs.append(cdf["Label"].values[0])

        ax.scatter(uk_pts, se_pts, color=MODEL_COLORS[model],
                   alpha=0.70, s=55, label=model, zorder=3)

        # Annotate the 100% coverage point
        for i, lb in enumerate(labs):
            if "100%" in lb:
                ax.annotate(f"{model[:2]} 100%",
                            (uk_pts[i], se_pts[i]),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=7, color=MODEL_COLORS[model])

    # TEMPO reference points
    tempo = df[df["Group"] == "TEMPO"]
    for tm in ["TEMPO Fine-Tuned", "TEMPO Zero-Shot"]:
        uk = tempo[(tempo["Model"] == tm) & (tempo["Site"] == "UK-AMo")]["R2"].values
        se = tempo[(tempo["Model"] == tm) & (tempo["Site"] == "SE-Htm")]["R2"].values
        if len(uk) and len(se):
            ax.scatter(uk, se, color=MODEL_COLORS[tm], marker="*",
                       s=220, zorder=4, label=tm)
            ax.annotate(tm.replace("TEMPO ", "TEMPO\n"),
                        (float(uk[0]), float(se[0])),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color=MODEL_COLORS[tm])

    # Equal-performance diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.35, label="Equal performance")

    ax.set_xlabel("Within-Ecosystem R²  (UK-AMo — Wetland)", fontsize=11)
    ax.set_ylabel("Cross-Ecosystem R²   (SE-Htm — Forest)", fontsize=11)
    ax.set_title(
        "Within- vs Cross-Ecosystem Transfer Performance\n"
        "Points below diagonal → cross-ecosystem penalty",
        fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary text (thesis-ready)
# ---------------------------------------------------------------------------
def build_summary(df: pd.DataFrame, tg_df: pd.DataFrame,
                  loo_df: pd.DataFrame) -> str:

    def _r2(model, site, config):
        sub = df[(df["Model"] == model) & (df["Site"] == site) &
                 (df["Config"] == config)]
        return float(sub["R2"].values[0]) if len(sub) else np.nan

    lines = [
        "=" * 85,
        "TRANSFER LEARNING ANALYSIS — CARBON FLUX QUANTIFICATION",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 85,
        "",
        "── STUDY DESIGN ─────────────────────────────────────────────────────────────",
        "",
        "  Training: 5 WETLAND sites (combined in train_X/y)",
        "    FI-Lom  Subarctic minerotrophic fen (Finland)",
        "    GL-ZaF  High Arctic tundra fen (Greenland)",
        "    IE-Cra  Ombrotrophic raised bog (Ireland)",
        "    DE-Akm  Rewetted coastal peatland (Germany)",
        "    FR-LGt  Acidic fen (France)",
        "",
        "  Test sites (held-out):",
        "    UK-AMo  Blanket bog, UK       → Wetland-to-Wetland transfer",
        "    SE-Htm  Norway spruce, Sweden → Wetland-to-Forest transfer",
        "",
    ]

    # H1: Coverage effects
    lines += [
        "── H1: TRAINING DATA COVERAGE EFFECTS ───────────────────────────────────────",
        "",
    ]
    for site in TEST_SITES:
        lines.append(f"  {site} ({ECOSYSTEM_TYPE.get(site, '')}):")
        for model in ["Random Forest", "XGBoost", "LSTM"]:
            r20  = _r2(model, site, "A_cov_020pct")
            r100 = _r2(model, site, "A_cov_100pct")
            if not (np.isnan(r20) or np.isnan(r100)):
                diff = r100 - r20
                pct  = diff / abs(r20) * 100 if r20 != 0 else 0
                direction = "improves" if diff > 0 else "degrades"
                lines.append(
                    f"    {model:<16} R²@20%={r20:.3f} → R²@100%={r100:.3f}"
                    f"  ({pct:+.1f}%, {direction} with more data)")
    lines.append("")

    # H2: LOO sensitivity
    lines += [
        "── H2: LEAVE-ONE-SITE-OUT SENSITIVITY ───────────────────────────────────────",
        "",
    ]
    for site in TEST_SITES:
        lines.append(f"  Impact on {site}:")
        sdf = loo_df[loo_df["Site"] == site]
        for model in ["Random Forest", "XGBoost", "LSTM"]:
            mdf = sdf[sdf["Model"] == model]
            if len(mdf) == 0:
                continue
            # most_harmful: removing this block HURT performance most (most neg delta)
            worst = mdf.loc[mdf["R2_delta"].idxmin()]
            # most_helpful: removing this block HELPED most (most pos delta)
            best  = mdf.loc[mdf["R2_delta"].idxmax()]
            lines.append(f"    {model}:")
            lines.append(f"      Excl. {worst['Label']:20s}  "
                         f"ΔR²={worst['R2_delta']:+.4f}  (most harmful removal)")
            lines.append(f"      Excl. {best['Label']:20s}  "
                         f"ΔR²={best['R2_delta']:+.4f}  (most beneficial removal)")
    lines.append("")

    # H3: TEMPO vs baselines
    lines += [
        "── H3: TEMPO vs BASELINES FOR CROSS-ECOSYSTEM TRANSFER ─────────────────────",
        "",
    ]
    for site in TEST_SITES:
        lines.append(f"  {site}:")
        for model in ["TEMPO Fine-Tuned", "TEMPO Zero-Shot",
                      "Random Forest", "XGBoost", "LSTM"]:
            cfg = "TEMPO_ref" if "TEMPO" in model else "A_cov_100pct"
            rv  = _r2(model, site, cfg)
            if not np.isnan(rv):
                lines.append(f"    {model:<20} R²={rv:.3f}")
        # TEMPO advantage over best baseline
        tempo_ft = _r2("TEMPO Fine-Tuned", site, "TEMPO_ref")
        baselines = [_r2(m, site, "A_cov_100pct")
                     for m in ["Random Forest", "XGBoost", "LSTM"]]
        baselines = [v for v in baselines if not np.isnan(v)]
        if baselines and not np.isnan(tempo_ft):
            best_bl = max(baselines)
            adv = (tempo_ft - best_bl) / abs(best_bl) * 100
            lines.append(f"    → TEMPO Fine-Tuned advantage vs best baseline: {adv:+.1f}%")
    lines.append("")

    # H4: Cross-ecosystem penalty
    lines += [
        "── H4: CROSS-ECOSYSTEM TRANSFER PENALTY (Wetland R² − Forest R²) ───────────",
        "",
    ]
    for model in ["Random Forest", "XGBoost", "LSTM", "TEMPO Fine-Tuned"]:
        cfg  = "TEMPO_ref" if "TEMPO" in model else "A_cov_100pct"
        uk   = _r2(model, "UK-AMo", cfg)
        se   = _r2(model, "SE-Htm", cfg)
        if not (np.isnan(uk) or np.isnan(se)):
            penalty = uk - se
            pct     = penalty / abs(uk) * 100 if uk != 0 else 0
            lines.append(
                f"    {model:<20} UK-AMo R²={uk:.3f}, SE-Htm R²={se:.3f},"
                f" Penalty={penalty:+.3f} ({pct:+.1f}%)")
    lines.append("")

    # Negative transfer summary
    neg = tg_df[tg_df["Negative_Transfer"] &
                tg_df["Group"].isin(["Coverage", "LOO"])]
    lines += [
        "── NEGATIVE TRANSFER CASES (R² < 100%-baseline − 0.01) ─────────────────────",
        "",
    ]
    if len(neg):
        lines.append(f"  {len(neg)} cases detected:")
        for _, row in neg.iterrows():
            lines.append(f"    {row['Label']:25s} | {row['Model']:16s} | {row['Site']}: "
                         f"R²={row['R2']:.3f} (Δ={row['Transfer_Gain']:+.3f})")
    else:
        lines.append("  No significant negative transfer detected.")
    lines.append("")

    lines += ["=" * 85]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("TRANSFER LEARNING ANALYSIS — CARBON FLUX QUANTIFICATION")
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Train data : 5 Wetland sites (combined)")
    print(f"  Test sites : UK-AMo (Wetland), SE-Htm (Forest)")
    print(f"  Models     : Random Forest, XGBoost, LSTM + TEMPO references")
    print(f"  Configs    : {len(COVERAGE_FRACS)} coverage × "
          f"{N_TRAIN_SITES} LOO = {len(COVERAGE_FRACS)+N_TRAIN_SITES} experiments")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ── [1] Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    X_train, y_train = load_train()
    test_data   = {site: load_test(site) for site in TEST_SITES}
    tempo_preds = {site: load_tempo_preds(site) for site in TEST_SITES}
    print(f"  Train shape   : {X_train.shape}  →  capped at {MAX_TRAIN_N}/experiment")
    for site, (Xte, yte) in test_data.items():
        print(f"  Test {site:8s}: X={Xte.shape}, y={yte.shape}")

    # ── [2] Run experiments ────────────────────────────────────────────────
    print(f"\n[2/5] Running {len(COVERAGE_FRACS)+N_TRAIN_SITES} experiments "
          f"× 3 models (RF, XGB, LSTM) ...")
    results_df = run_experiments(X_train, y_train, test_data, tempo_preds)

    # ── [3] Transfer metrics ───────────────────────────────────────────────
    print("\n[3/5] Computing transfer metrics & statistics...")
    tg_df    = compute_transfer_gain(results_df)
    loo_df   = loo_sensitivity(results_df)
    stats_df = coverage_ttest(results_df)
    neg_df   = (tg_df[tg_df["Negative_Transfer"] &
                       tg_df["Group"].isin(["Coverage", "LOO"])]
                [["Config","Group","Label","Model","Site",
                  "R2","R2_baseline","Transfer_Gain"]].copy())

    results_df.to_csv(OUT_DIR / "transfer_matrix.csv", index=False)
    neg_df.to_csv(OUT_DIR / "negative_transfer_summary.csv", index=False)
    stats_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    print(f"  Saved: results/transfer_learning/transfer_matrix.csv"
          f"  ({len(results_df)} rows)")
    print(f"  Saved: results/transfer_learning/negative_transfer_summary.csv"
          f"  ({len(neg_df)} negative transfer cases)")
    print(f"  Saved: results/transfer_learning/statistical_tests.csv")

    # ── [4] Summary text ───────────────────────────────────────────────────
    print("\n[4/5] Writing summary...")
    summary  = build_summary(results_df, tg_df, loo_df)
    txt_path = OUT_DIR / "TRANSFER_LEARNING_SUMMARY.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    print(summary)

    # ── [5] Figures ────────────────────────────────────────────────────────
    print("\n[5/5] Generating figures...")

    fig = plot_coverage_performance(results_df)
    p = FIG_DIR / "coverage_vs_performance.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    fig = plot_transfer_matrix(results_df)
    p = FIG_DIR / "transfer_matrix_heatmap.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    fig = plot_loo_effects(loo_df)
    p = FIG_DIR / "loo_site_contribution.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    fig = plot_model_comparison(results_df)
    p = FIG_DIR / "model_comparison_transfer.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    fig = plot_within_vs_cross(results_df)
    p = FIG_DIR / "within_vs_cross_ecosystem.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved: {p.name}")

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║  TRANSFER LEARNING ANALYSIS COMPLETE" + " " * 41 + "║")
    print("║  Results : results/transfer_learning/*.csv" + " " * 35 + "║")
    print("║  Summary : results/transfer_learning/TRANSFER_LEARNING_SUMMARY.txt"
          + " " * 8 + "║")
    print("║  Figures : figures/transfer_learning/" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝")


if __name__ == "__main__":
    main()
