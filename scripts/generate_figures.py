"""
Publication Figure Generation
==============================

Loads evaluation results from results/metrics/ and generates
publication-quality figures for the thesis.

Figures produced:
  1. Cross-site performance comparison (grouped bar plot)
  2. Forecast examples (observed vs predicted line plots)
  3. Error distribution histograms

Usage:
    python scripts/generate_figures.py

Requires: results from scripts/evaluate_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PREDS_DIR = METRICS_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "figures"
THESIS_FIGURES_DIR = PROJECT_ROOT / "thesis" / "figures"

TEST_SITES = ["UK-AMo", "SE-Htm"]
HORIZON = 96

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "RandomForest": "#4C72B0",
    "XGBoost": "#55A868",
    "LSTM": "#C44E52",
    "TEMPO-ZeroShot": "#8172B2",
    "TEMPO-FineTuned": "#CCB974",
}

MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
    "TEMPO-ZeroShot": "TEMPO (Zero-shot)",
    "TEMPO-FineTuned": "TEMPO (Fine-tuned)",
}

SITE_LABELS = {
    "UK-AMo": "UK-AMo (Auchencorth Moss)",
    "SE-Htm": "SE-Htm (Hyltemossa)",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_metrics():
    """Load the combined metrics JSON."""
    path = METRICS_DIR / "all_metrics.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/evaluate_models.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_predictions(model_key, site):
    """Load saved predictions for a model/site pair. Returns (preds, targets) or None."""
    name_map = {
        "RandomForest": "randomforest",
        "XGBoost": "xgboost",
        "LSTM": "lstm",
        "TEMPO-ZeroShot": "tempo_zs",
        "TEMPO-FineTuned": "tempo_ft",
    }
    prefix = name_map.get(model_key)
    if prefix is None:
        return None

    pred_path = PREDS_DIR / f"{prefix}_preds_{site}.npy"
    if not pred_path.exists():
        return None

    preds = np.load(pred_path)

    # Targets: TEMPO stores its own; baselines share a common file
    if "tempo" in prefix:
        tgt_path = PREDS_DIR / f"{prefix}_targets_{site}.npy"
    else:
        tgt_path = PREDS_DIR / f"targets_{site}.npy"

    if not tgt_path.exists():
        return None

    targets = np.load(tgt_path)
    return preds, targets


# ---------------------------------------------------------------------------
# Figure 1: Cross-site performance comparison (grouped bar plot)
# ---------------------------------------------------------------------------
def fig_performance_comparison(metrics, save_dirs):
    """Grouped bar chart: models × metrics, one subplot per test site."""
    models = [m for m in metrics if any(s in metrics[m] for s in TEST_SITES)]
    metric_names = ["RMSE", "MAE", "R2"]
    metric_labels = ["RMSE", "MAE", "R²"]

    fig, axes = plt.subplots(1, len(TEST_SITES), figsize=(7 * len(TEST_SITES), 5))
    if len(TEST_SITES) == 1:
        axes = [axes]

    for ax, site in zip(axes, TEST_SITES):
        present_models = [m for m in models if site in metrics[m]]
        n_models = len(present_models)
        x = np.arange(len(metric_names))
        width = 0.8 / max(n_models, 1)

        for i, model in enumerate(present_models):
            vals = [metrics[model][site].get(mn, 0) for mn in metric_names]
            offset = (i - (n_models - 1) / 2) * width
            color = MODEL_COLORS.get(model, f"C{i}")
            label = MODEL_LABELS.get(model, model)
            bars = ax.bar(x + offset, vals, width * 0.9, label=label, color=color)
            # Value labels on bars
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_title(SITE_LABELS.get(site, site))
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))

    fig.suptitle("Cross-Site Model Performance Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    for d in save_dirs:
        fig.savefig(d / "performance_comparison.pdf")
        fig.savefig(d / "performance_comparison.png")
    plt.close(fig)
    print("  Saved performance_comparison.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2: Forecast examples (line plots)
# ---------------------------------------------------------------------------
def fig_forecast_examples(metrics, save_dirs, n_samples=3):
    """Line plots showing observed vs predicted for sample windows."""
    models_with_preds = []
    for m in metrics:
        if any(load_predictions(m, s) is not None for s in TEST_SITES):
            models_with_preds.append(m)

    if not models_with_preds:
        print("  Skipping forecast examples — no prediction files found.")
        return

    hours = np.arange(HORIZON)

    for site in TEST_SITES:
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3.2 * n_samples))
        if n_samples == 1:
            axes = [axes]

        # Load one set of targets for sample indices
        ref = load_predictions(models_with_preds[0], site)
        if ref is None:
            continue
        _, targets_ref = ref
        n_total = len(targets_ref)
        sample_idx = np.linspace(0, n_total - 1, n_samples, dtype=int)

        for ax, si in zip(axes, sample_idx):
            ax.plot(hours, targets_ref[si], "k-", linewidth=1.8,
                    label="Observed", zorder=10)

            for model in models_with_preds:
                result = load_predictions(model, site)
                if result is None:
                    continue
                preds, targets = result
                if si >= len(preds):
                    continue
                color = MODEL_COLORS.get(model, "gray")
                label = MODEL_LABELS.get(model, model)
                ax.plot(hours, preds[si], "--", color=color,
                        linewidth=1.2, alpha=0.85, label=label)

            ax.set_ylabel("NEE (normalised)")
            ax.set_title(f"{SITE_LABELS.get(site, site)} — Window {si}")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ax is axes[0]:
                ax.legend(loc="upper right", fontsize=8, ncol=2)

        axes[-1].set_xlabel("Forecast Hour")
        fig.suptitle(f"Forecast Examples — {site}", fontsize=13, y=1.01)
        plt.tight_layout()

        for d in save_dirs:
            fig.savefig(d / f"forecast_examples_{site}.pdf")
            fig.savefig(d / f"forecast_examples_{site}.png")
        plt.close(fig)
        print(f"  Saved forecast_examples_{site}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 3: Error distribution histograms
# ---------------------------------------------------------------------------
def fig_error_distributions(metrics, save_dirs, max_samples=5000):
    """Histograms of per-timestep forecast errors for each model/site."""
    models_with_preds = []
    for m in metrics:
        if any(load_predictions(m, s) is not None for s in TEST_SITES):
            models_with_preds.append(m)

    if not models_with_preds:
        print("  Skipping error distributions — no prediction files found.")
        return

    fig, axes = plt.subplots(1, len(TEST_SITES),
                             figsize=(6.5 * len(TEST_SITES), 4.5))
    if len(TEST_SITES) == 1:
        axes = [axes]

    for ax, site in zip(axes, TEST_SITES):
        for model in models_with_preds:
            result = load_predictions(model, site)
            if result is None:
                continue
            preds, targets = result
            # Subsample to keep histogram manageable
            n = len(preds)
            if n > max_samples:
                idx = np.random.choice(n, max_samples, replace=False)
                preds, targets = preds[idx], targets[idx]
            errors = (preds - targets).flatten()

            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_LABELS.get(model, model)
            ax.hist(errors, bins=80, alpha=0.45, color=color, label=label,
                    density=True, histtype="stepfilled", edgecolor="white",
                    linewidth=0.3)

        ax.set_xlabel("Prediction Error (pred − obs)")
        ax.set_ylabel("Density")
        ax.set_title(SITE_LABELS.get(site, site))
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)

    fig.suptitle("Error Distributions Across Models", fontsize=14, y=1.02)
    plt.tight_layout()

    for d in save_dirs:
        fig.savefig(d / "error_distributions.pdf")
        fig.savefig(d / "error_distributions.png")
    plt.close(fig)
    print("  Saved error_distributions.{pdf,png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Ensure output directories exist
    for d in [FIGURES_DIR, THESIS_FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    save_dirs = [FIGURES_DIR, THESIS_FIGURES_DIR]

    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    metrics = load_metrics()
    models_found = list(metrics.keys())
    print(f"\nModels found: {', '.join(models_found)}")
    print(f"Test sites:   {', '.join(TEST_SITES)}")
    print(f"Output dirs:  {FIGURES_DIR}/")
    print(f"              {THESIS_FIGURES_DIR}/\n")

    fig_performance_comparison(metrics, save_dirs)
    fig_forecast_examples(metrics, save_dirs)
    fig_error_distributions(metrics, save_dirs)

    print(f"\nAll figures saved.")


if __name__ == "__main__":
    main()
