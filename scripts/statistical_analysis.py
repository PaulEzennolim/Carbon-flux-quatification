"""
Statistical Significance Testing for Carbon Flux Models
========================================================
Performs rigorous statistical tests to validate model performance claims
against UK-AMo (wetland) and SE-Htm (forest) eddy covariance sites.

Tests implemented:
1. Paired t-tests (parametric) — scipy.stats.ttest_rel
2. Diebold-Mariano test (forecast-specific, implemented from scratch)
3. Bootstrapped confidence intervals (non-parametric)

References:
- Diebold, F.X., & Mariano, R.S. (1995). Comparing predictive accuracy.
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
  prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.
- Efron, B., & Tibshirani, R.J. (1994). An introduction to the bootstrap. CRC press.

Usage:
    python scripts/statistical_analysis.py
    python scripts/statistical_analysis.py --bootstrap-samples 10000
    python scripts/statistical_analysis.py --alpha 0.01
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
BASELINE_DIR = PRED_DIR / "baselines"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "analysis"
FIG_DIR = ROOT / "figures" / "statistical_analysis"

SITES = ["UK-AMo", "SE-Htm"]
MODEL_DISPLAY = {
    "tempo_fine_tuned": "TEMPO Fine-Tuned",
    "tempo_zero_shot": "TEMPO Zero-Shot",
    "xgboost": "XGBoost",
    "randomforest": "Random Forest",
    "lstm": "LSTM",
}
COLORS = {
    "TEMPO Fine-Tuned": "#2196F3",
    "TEMPO Zero-Shot": "#03A9F4",
    "XGBoost": "#FF5722",
    "Random Forest": "#FF9800",
    "LSTM": "#9C27B0",
}


# ---------------------------------------------------------------------------
# Helper — metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# StatisticalAnalyzer
# ---------------------------------------------------------------------------

class StatisticalAnalyzer:
    """Rigorous statistical testing for carbon flux forecast models."""

    def __init__(self, alpha: float = 0.05, bootstrap_samples: int = 1000):
        self.alpha = alpha
        self.n_boot = bootstrap_samples
        self.rng = np.random.default_rng(seed=42)

        # Populated by load_predictions()
        self.targets: Dict[str, np.ndarray] = {}
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}

        # Results containers
        self.ttest_results: Dict = {}
        self.dm_results: Dict = {}
        self.bootstrap_results: Dict = {}

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _try_load(self, path: Path) -> Optional[np.ndarray]:
        if path.exists():
            arr = np.load(path)
            return arr.flatten()
        return None

    def load_predictions(self) -> None:
        """Load all model predictions and targets from disk."""
        for site in SITES:
            # --- targets ---
            # Prefer the targets stored alongside baseline predictions;
            # fall back to data/processed/
            t = self._try_load(BASELINE_DIR / f"targets_{site}.npy")
            if t is None:
                t = self._try_load(PROCESSED_DIR / f"test_{site}_y.npy")
            if t is None:
                raise FileNotFoundError(
                    f"Cannot find targets for {site}. Checked:\n"
                    f"  {BASELINE_DIR / f'targets_{site}.npy'}\n"
                    f"  {PROCESSED_DIR / f'test_{site}_y.npy'}"
                )
            self.targets[site] = t

            # --- predictions ---
            self.predictions[site] = {}
            model_files = {
                "tempo_fine_tuned": PRED_DIR / f"tempo_fine_tuned_preds_{site}.npy",
                "tempo_zero_shot": PRED_DIR / f"tempo_zero_shot_preds_{site}.npy",
                "xgboost": BASELINE_DIR / f"xgboost_preds_{site}.npy",
                "randomforest": BASELINE_DIR / f"randomforest_preds_{site}.npy",
                "lstm": BASELINE_DIR / f"lstm_preds_{site}.npy",
            }
            for key, path in model_files.items():
                arr = self._try_load(path)
                if arr is not None:
                    # Align length with targets
                    n = len(self.targets[site])
                    if len(arr) != n:
                        print(
                            f"  WARNING: {key} predictions for {site} have length "
                            f"{len(arr)} vs targets {n}. Truncating to min."
                        )
                        n = min(n, len(arr))
                        self.targets[site] = self.targets[site][:n]
                        arr = arr[:n]
                    self.predictions[site][key] = arr
                else:
                    print(f"  WARNING: Missing predictions for {key} / {site}")

        # Report what was loaded
        for site in SITES:
            models = list(self.predictions[site].keys())
            print(f"  {site}: {len(self.targets[site])} samples | models: {models}")

    # ------------------------------------------------------------------
    # 1. Paired t-test
    # ------------------------------------------------------------------

    def paired_ttest(
        self,
        y_true: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        name1: str,
        name2: str,
    ) -> Dict:
        """
        Paired t-test on absolute prediction errors.

        H0: mean(|e1|) == mean(|e2|)   (two-tailed)

        Returns a dict with t-statistic, p-value, 95% CI, Cohen's d,
        and a plain-English interpretation.
        """
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)
        d = e1 - e2  # positive → pred1 is worse

        t_stat, p_val = stats.ttest_rel(e1, e2)
        n = len(d)

        # 95% CI for mean difference
        se = stats.sem(d)
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
        ci_lower = float(np.mean(d) - t_crit * se)
        ci_upper = float(np.mean(d) + t_crit * se)

        # Cohen's d for paired differences
        cd = self.cohens_d(e1, e2)

        significant = p_val < self.alpha
        mean_diff = float(np.mean(d))

        # Interpretation — d = e1 - e2, so:
        #   mean_diff < 0  →  model1 has LOWER error  →  model1 outperforms model2
        #   mean_diff > 0  →  model1 has HIGHER error →  model1 is outperformed by model2
        if mean_diff < 0:
            direction = "outperforms"
        else:
            direction = "is outperformed by"
        sig_str = "significantly " if significant else "does NOT significantly "
        stars = significance_stars(p_val)
        interp = (
            f"{name1} {sig_str}{direction} {name2} "
            f"(p={p_val:.3e} {stars})"
        )

        return {
            "metric": "MAE",
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "cohens_d": round(cd, 4),
            "significant": significant,
            "stars": stars,
            "interpretation": interp,
        }

    # ------------------------------------------------------------------
    # 2. Diebold-Mariano test (from scratch)
    # ------------------------------------------------------------------

    @staticmethod
    def _newey_west_variance(d: np.ndarray, max_lag: int) -> float:
        """
        HAC (Newey-West) variance estimate for the mean of d.

        VAR(d_bar) = (1/T^2) * [gamma_0 + 2 * sum_{k=1}^{L} w_k * gamma_k]
        where gamma_k = autocovariance at lag k, w_k = 1 - k/(L+1) (Bartlett weights).
        """
        T = len(d)
        d_dm = d - d.mean()
        gamma0 = np.dot(d_dm, d_dm) / T

        var = gamma0
        for k in range(1, max_lag + 1):
            weight = 1.0 - k / (max_lag + 1)
            gamma_k = np.dot(d_dm[k:], d_dm[:-k]) / T
            var += 2 * weight * gamma_k

        return max(var / T, 1e-12)  # guard against non-positive estimates

    def diebold_mariano_test(
        self,
        y_true: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        name1: str,
        name2: str,
        h: int = 1,
    ) -> Dict:
        """
        Diebold-Mariano (1995) test for equal forecast accuracy.

        Loss function: squared errors (L = e²).
        H0: E[d_t] = 0  where d_t = L(e1_t) - L(e2_t)
        H1 (one-tailed): d_bar < 0  →  model 1 is more accurate

        Uses the Harvey, Leybourne & Newbold (1997) small-sample correction
        (modified DM statistic), which follows a t(T-1) distribution.

        Parameters
        ----------
        h : forecast horizon (used for lag selection and HLN correction)
        """
        e1 = y_true - pred1
        e2 = y_true - pred2

        L1 = e1 ** 2  # squared loss
        L2 = e2 ** 2
        d = L1 - L2   # loss differential

        T = len(d)
        max_lag = max(1, h - 1)  # per Diebold & Mariano (1995)

        hac_var = self._newey_west_variance(d, max_lag)
        dm_stat = float(np.mean(d) / np.sqrt(hac_var))

        # Harvey-Leybourne-Newbold small-sample correction
        hln_correction = np.sqrt(
            (T + 1 - 2 * h + h * (h - 1) / T) / T
        )
        mdm_stat = float(dm_stat * hln_correction)

        # Two-tailed p-value under t(T-1)
        p_two = float(2 * stats.t.sf(abs(mdm_stat), df=T - 1))
        # One-tailed: is pred1 better than pred2? (d_bar < 0)
        p_one = float(stats.t.cdf(mdm_stat, df=T - 1))

        h0_rejected = p_two < self.alpha
        stars = significance_stars(p_two)

        if h0_rejected:
            winner = name1 if np.mean(d) < 0 else name2
            conclusion = (
                f"{winner} has significantly better forecast accuracy "
                f"(MDM={mdm_stat:.3f}, p={p_two:.3e} {stars})"
            )
        else:
            conclusion = (
                f"No significant difference in forecast accuracy "
                f"(MDM={mdm_stat:.3f}, p={p_two:.3e})"
            )

        return {
            "dm_statistic": round(dm_stat, 4),
            "mdm_statistic": round(mdm_stat, 4),
            "p_value_two_tailed": p_two,
            "p_value_one_tailed": p_one,
            "h0_rejected": h0_rejected,
            "stars": stars,
            "loss_differential_mean": round(float(np.mean(d)), 6),
            "hac_std_error": round(float(np.sqrt(hac_var)), 6),
            "conclusion": conclusion,
        }

    # ------------------------------------------------------------------
    # 3. Bootstrapped confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_boot: Optional[int] = None,
    ) -> Dict:
        """
        Non-parametric bootstrap for RMSE, MAE, and R² confidence intervals.

        Algorithm
        ---------
        1. Resample (y_true, y_pred) pairs with replacement n_boot times.
        2. Compute metrics on each resample.
        3. Return 2.5th / 97.5th percentiles as the 95% CI.
        """
        n_boot = n_boot or self.n_boot
        n = len(y_true)

        boot_rmse = np.empty(n_boot)
        boot_mae = np.empty(n_boot)
        boot_r2 = np.empty(n_boot)

        for b in range(n_boot):
            idx = self.rng.integers(0, n, size=n)
            yt, yp = y_true[idx], y_pred[idx]
            boot_rmse[b] = rmse(yt, yp)
            boot_mae[b] = mae(yt, yp)
            boot_r2[b] = r2(yt, yp)

        def ci_dict(point, boot_arr):
            lo, hi = np.percentile(boot_arr, [2.5, 97.5])
            return {
                "point_estimate": round(point, 4),
                "ci_lower": round(float(lo), 4),
                "ci_upper": round(float(hi), 4),
                "std_error": round(float(boot_arr.std()), 4),
            }

        return {
            "RMSE": ci_dict(rmse(y_true, y_pred), boot_rmse),
            "MAE": ci_dict(mae(y_true, y_pred), boot_mae),
            "R2": ci_dict(r2(y_true, y_pred), boot_r2),
        }

    # ------------------------------------------------------------------
    # Cohen's d
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_d(errors1: np.ndarray, errors2: np.ndarray) -> float:
        """Cohen's d for paired differences (absolute errors)."""
        d = errors1 - errors2
        pooled_std = np.sqrt((errors1.var(ddof=1) + errors2.var(ddof=1)) / 2)
        return float(abs(d.mean()) / pooled_std) if pooled_std > 0 else 0.0

    # ------------------------------------------------------------------
    # Best baseline helper
    # ------------------------------------------------------------------

    def _best_baseline(self, site: str) -> Tuple[str, str]:
        """Return (model_key, display_name) for the baseline with lowest RMSE."""
        preds = self.predictions[site]
        yt = self.targets[site]
        baseline_keys = [k for k in preds if k not in ("tempo_fine_tuned", "tempo_zero_shot")]
        if not baseline_keys:
            raise RuntimeError(f"No baseline predictions available for {site}")
        best_key = min(baseline_keys, key=lambda k: rmse(yt, preds[k]))
        return best_key, MODEL_DISPLAY[best_key]

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------

    def run_all_tests(self) -> None:
        """Execute all three statistical tests for every site and comparison pair."""
        self.load_predictions()

        for site in SITES:
            print(f"\n{'=' * 50}")
            print(f"  Site: {site}")
            print(f"{'=' * 50}")
            yt = self.targets[site]
            preds = self.predictions[site]

            if "tempo_fine_tuned" not in preds:
                print(f"  Skipping {site}: TEMPO Fine-Tuned predictions missing.")
                continue

            # ----------------------------------------------------------
            # Define comparison pairs
            # ----------------------------------------------------------
            best_bl_key, best_bl_name = self._best_baseline(site)
            pairs: List[Tuple[str, str, str, str]] = []

            # Always include the three primary comparisons
            if "tempo_fine_tuned" in preds and best_bl_key in preds:
                pairs.append(("tempo_fine_tuned", best_bl_key,
                               "TEMPO Fine-Tuned", best_bl_name))
            if "tempo_fine_tuned" in preds and "tempo_zero_shot" in preds:
                pairs.append(("tempo_fine_tuned", "tempo_zero_shot",
                               "TEMPO Fine-Tuned", "TEMPO Zero-Shot"))
            if "tempo_zero_shot" in preds and best_bl_key in preds:
                pairs.append(("tempo_zero_shot", best_bl_key,
                               "TEMPO Zero-Shot", best_bl_name))

            # All remaining pairwise
            all_keys = list(preds.keys())
            existing_pairs = {(a, b) for a, b, _, _ in pairs} | \
                             {(b, a) for a, b, _, _ in pairs}
            for i, k1 in enumerate(all_keys):
                for k2 in all_keys[i + 1:]:
                    if (k1, k2) not in existing_pairs:
                        pairs.append((k1, k2, MODEL_DISPLAY[k1], MODEL_DISPLAY[k2]))

            self.ttest_results[site] = {}
            self.dm_results[site] = {}

            for k1, k2, n1, n2 in pairs:
                # CRITICAL: Skip self-comparisons entirely — don't test, don't save
                if k1 == k2 or n1 == n2:
                    continue

                label = f"{n1} vs {n2}"
                print(f"\n  >>> {label}")

                # Paired t-test
                tt = self.paired_ttest(yt, preds[k1], preds[k2], n1, n2)
                self.ttest_results[site][label] = tt
                print(
                    f"    t-test : t={tt['t_statistic']:.3f}, "
                    f"p={tt['p_value']:.3e} {tt['stars']}, "
                    f"d={tt['cohens_d']:.3f}"
                )

                # Diebold-Mariano
                dm = self.diebold_mariano_test(yt, preds[k1], preds[k2], n1, n2, h=96)
                self.dm_results[site][label] = dm
                print(
                    f"    DM test: MDM={dm['mdm_statistic']:.3f}, "
                    f"p={dm['p_value_two_tailed']:.3e} {dm['stars']}"
                )

            # ----------------------------------------------------------
            # Bootstrap CI for every model
            # ----------------------------------------------------------
            print(f"\n  Bootstrap CI ({self.n_boot} samples):")
            self.bootstrap_results[site] = {}
            for key, pred in preds.items():
                name = MODEL_DISPLAY[key]
                ci = self.bootstrap_ci(yt, pred)
                self.bootstrap_results[site][name] = ci
                print(
                    f"    {name:20s}  RMSE={ci['RMSE']['point_estimate']:.4f} "
                    f"[{ci['RMSE']['ci_lower']:.4f}, {ci['RMSE']['ci_upper']:.4f}]  "
                    f"R²={ci['R2']['point_estimate']:.4f} "
                    f"[{ci['R2']['ci_lower']:.4f}, {ci['R2']['ci_upper']:.4f}]"
                )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def generate_figures(self) -> None:
        """Create publication-quality figures."""
        self._fig_confidence_intervals()
        self._fig_effect_sizes()

    def _fig_confidence_intervals(self) -> None:
        """Bar chart with 95% bootstrap CI error bars, one panel per site."""
        n_sites = len([s for s in SITES if s in self.bootstrap_results])
        if n_sites == 0:
            return

        fig, axes = plt.subplots(1, n_sites, figsize=(7 * n_sites, 5), sharey=False)
        if n_sites == 1:
            axes = [axes]

        fig.suptitle(
            "Model RMSE with 95% Bootstrap Confidence Intervals",
            fontsize=14, fontweight="bold", y=1.02,
        )

        for ax, site in zip(axes, [s for s in SITES if s in self.bootstrap_results]):
            boot = self.bootstrap_results[site]
            models = list(boot.keys())
            rmses = [boot[m]["RMSE"]["point_estimate"] for m in models]
            ci_lo = [boot[m]["RMSE"]["ci_lower"] for m in models]
            ci_hi = [boot[m]["RMSE"]["ci_upper"] for m in models]
            errs_lo = [r - lo for r, lo in zip(rmses, ci_lo)]
            errs_hi = [hi - r for r, hi in zip(rmses, ci_hi)]
            colors = [COLORS.get(m, "#607D8B") for m in models]
            x = np.arange(len(models))

            ax.bar(x, rmses, color=colors, alpha=0.85, edgecolor="white",
                   linewidth=0.8, zorder=3)
            ax.errorbar(
                x, rmses,
                yerr=[errs_lo, errs_hi],
                fmt="none", color="black", capsize=5, capthick=1.5,
                linewidth=1.5, zorder=4,
            )

            # Annotate significance vs TEMPO Fine-Tuned
            if site in self.ttest_results and "TEMPO Fine-Tuned" in boot:
                for i, m in enumerate(models):
                    if m == "TEMPO Fine-Tuned":
                        continue
                    label = f"TEMPO Fine-Tuned vs {m}"
                    rev_label = f"{m} vs TEMPO Fine-Tuned"
                    tt = self.ttest_results[site].get(label) or \
                         self.ttest_results[site].get(rev_label)
                    if tt and tt["significant"]:
                        ax.annotate(
                            tt["stars"],
                            xy=(i, rmses[i] + errs_hi[i]),
                            xytext=(i, rmses[i] + errs_hi[i] + 0.04),
                            ha="center", fontsize=12, fontweight="bold", color="#B71C1C",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("RMSE (µmol m⁻² s⁻¹)", fontsize=10)
            ax.set_title(
                f"{site} — {'Wetland' if 'AMo' in site else 'Forest'}",
                fontsize=11, fontweight="bold",
            )
            ax.yaxis.grid(True, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

        legend_handles = [
            mpatches.Patch(color=c, label=m) for m, c in COLORS.items()
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=len(COLORS), bbox_to_anchor=(0.5, -0.08),
            frameon=True, fontsize=9,
        )
        note = "Error bars = 95% bootstrap CI. Significance vs TEMPO Fine-Tuned: * p<0.05  ** p<0.01  *** p<0.001"
        fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="grey")

        plt.tight_layout()
        out = FIG_DIR / "confidence_intervals.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {out}")

    def _fig_effect_sizes(self) -> None:
        """Forest plot of Cohen's d for comparisons involving TEMPO Fine-Tuned."""
        n_sites = len([s for s in SITES if s in self.ttest_results])
        if n_sites == 0:
            return

        fig, axes = plt.subplots(1, n_sites, figsize=(7 * n_sites, 5), sharey=False)
        if n_sites == 1:
            axes = [axes]

        fig.suptitle("Effect Sizes (Cohen's d) — TEMPO Fine-Tuned vs Competitors",
                     fontsize=13, fontweight="bold", y=1.02)

        for ax, site in zip(axes, [s for s in SITES if s in self.ttest_results]):
            comparisons, ds, colors_es = [], [], []
            for label, tt in self.ttest_results[site].items():
                if "TEMPO Fine-Tuned" not in label:
                    continue
                comparisons.append(label.replace("TEMPO Fine-Tuned vs ", "vs "))
                d_val = tt["cohens_d"]
                # Sign: positive = Fine-Tuned better (lower error)
                if tt["mean_diff"] > 0:  # pred1 worse → Fine-Tuned is pred2
                    d_val = -d_val
                ds.append(d_val)
                colors_es.append("#1976D2" if d_val < 0 else "#E53935")

            if not comparisons:
                ax.set_visible(False)
                continue

            y_pos = np.arange(len(comparisons))
            ax.barh(y_pos, ds, color=colors_es, alpha=0.8, edgecolor="white",
                    linewidth=0.8, height=0.5)
            ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

            # Reference lines for effect size thresholds (Cohen, 1988)
            for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                for sign in [-1, 1]:
                    ax.axvline(sign * thresh, color="grey", linewidth=0.8,
                               linestyle=":", alpha=0.6)
                ax.text(thresh + 0.01, len(comparisons) - 0.2, lbl,
                        fontsize=7, color="grey", va="top")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(comparisons, fontsize=9)
            ax.set_xlabel("Cohen's d (negative = TEMPO Fine-Tuned better)", fontsize=9)
            ax.set_title(
                f"{site} — {'Wetland' if 'AMo' in site else 'Forest'}",
                fontsize=11, fontweight="bold",
            )
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

        plt.tight_layout()
        out = FIG_DIR / "effect_sizes.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        """Write all results to disk in JSON, CSV, TXT, and LaTeX formats."""
        self._save_json()
        self._save_bootstrap_json()
        self._save_pairwise_csv()
        self._save_summary_txt()
        self._save_latex_table()

    def _save_json(self) -> None:
        """Consolidated JSON of t-test and DM results."""
        combined = {
            site: {
                "paired_ttests": self.ttest_results.get(site, {}),
                "diebold_mariano": self.dm_results.get(site, {}),
            }
            for site in SITES
        }
        out = OUTPUT_DIR / "statistical_tests.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"  Saved: {out}")

    def _save_bootstrap_json(self) -> None:
        out = OUTPUT_DIR / "bootstrap_results.json"
        with open(out, "w") as f:
            json.dump(self.bootstrap_results, f, indent=2, default=str)
        print(f"  Saved: {out}")

    def _save_pairwise_csv(self) -> None:
        """CSV with all pairwise p-values and effect sizes."""
        rows = []
        for site in SITES:
            for label, tt in self.ttest_results.get(site, {}).items():
                dm = self.dm_results.get(site, {}).get(label, {})
                rows.append({
                    "site": site,
                    "comparison": label,
                    "mean_diff_mae": tt["mean_diff"],
                    "t_statistic": tt["t_statistic"],
                    "p_value_ttest": tt["p_value"],
                    "ttest_significant": tt["significant"],
                    "cohens_d": tt["cohens_d"],
                    "ci_lower": tt["ci_lower"],
                    "ci_upper": tt["ci_upper"],
                    "dm_mdm_statistic": dm.get("mdm_statistic", "N/A"),
                    "p_value_dm": dm.get("p_value_two_tailed", "N/A"),
                    "dm_h0_rejected": dm.get("h0_rejected", "N/A"),
                })
        df = pd.DataFrame(rows)
        out = OUTPUT_DIR / "pairwise_comparisons.csv"
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")

    def _format_p(self, p: float) -> str:
        """Format p-value for display."""
        stars = significance_stars(p)
        if p < 0.001:
            return f"p<0.001 {stars}"
        return f"p={p:.3f} {stars}"

    def _effect_magnitude(self, d: float) -> str:
        if abs(d) >= 0.8:
            return "large"
        if abs(d) >= 0.5:
            return "medium"
        if abs(d) >= 0.2:
            return "small-medium"
        return "small"

    def _save_summary_txt(self) -> None:
        """Human-readable thesis-ready summary."""
        lines = [
            "=" * 70,
            "STATISTICAL SIGNIFICANCE ANALYSIS — CARBON FLUX MODELS",
            "=" * 70,
            f"Bootstrap samples    : {self.n_boot}",
            f"Significance level α : {self.alpha}",
            f"DM test horizon h    : 96 (96-hour forecasts)",
            f"Loss function        : squared errors",
            "",
        ]

        site_labels = {"UK-AMo": "Wetland", "SE-Htm": "Forest"}

        for site in SITES:
            if site not in self.ttest_results:
                continue
            lines += [
                f"{'─' * 70}",
                f"{site} ({site_labels.get(site, '')})",
                f"{'─' * 70}",
            ]

            # Bootstrap CI table
            if site in self.bootstrap_results:
                lines.append("\n  Bootstrap CI (95%) — Point Estimates:\n")
                lines.append(f"  {'Model':<22} {'RMSE':>10} {'95% CI':>22} "
                             f"{'MAE':>10} {'R²':>10}")
                lines.append("  " + "-" * 78)
                for m, ci in self.bootstrap_results[site].items():
                    r = ci["RMSE"]
                    r2v = ci["R2"]
                    ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                    lines.append(
                        f"  {m:<22} {r['point_estimate']:>10.4f} {ci_str:>22} "
                        f"{ci['MAE']['point_estimate']:>10.4f} "
                        f"{r2v['point_estimate']:>10.4f}"
                    )

            lines.append("")

            # Per-comparison block
            for label, tt in self.ttest_results[site].items():
                dm = self.dm_results[site].get(label, {})
                lines += [
                    f"\n  {label}",
                    f"  {'─' * 60}",
                    f"  MAE difference        : {tt['mean_diff']:+.4f}  "
                    f"(95% CI: [{tt['ci_lower']:.4f}, {tt['ci_upper']:.4f}])",
                    f"  Paired t-test         : t={tt['t_statistic']:.3f}, "
                    f"{self._format_p(tt['p_value'])}",
                ]
                if dm:
                    lines.append(
                        f"  Diebold-Mariano (MDM) : MDM={dm['mdm_statistic']:.3f}, "
                        f"{self._format_p(dm['p_value_two_tailed'])}"
                    )
                lines.append(
                    f"  Cohen's d             : {tt['cohens_d']:.3f} "
                    f"({self._effect_magnitude(tt['cohens_d'])} effect)"
                )
                lines.append(f"  Conclusion            : {tt['interpretation']}")

            lines.append("")

        lines += ["=" * 70, "End of Statistical Analysis", "=" * 70]
        out = OUTPUT_DIR / "statistical_summary.txt"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {out}")

    def _save_latex_table(self) -> None:
        """LaTeX table for thesis."""
        def fmt_rmse(site, label):
            """Return ΔRMSE and significance for t-test and DM."""
            tt = self.ttest_results.get(site, {}).get(label, {})
            dm = self.dm_results.get(site, {}).get(label, {})
            if not tt:
                return "—", "—", "—"
            # Use bootstrap RMSE difference for the table
            boot = self.bootstrap_results.get(site, {})
            m1, m2 = label.split(" vs ", 1)
            r1 = boot.get(m1, {}).get("RMSE", {}).get("point_estimate")
            r2v = boot.get(m2, {}).get("RMSE", {}).get("point_estimate")
            if r1 is not None and r2v is not None:
                delta = r1 - r2v
                stars = significance_stars(tt["p_value"])
                delta_str = f"{delta:+.2f}{stars if stars != 'ns' else ''}"
            else:
                delta_str = "—"
            p_tt = self._format_p(tt["p_value"]).split()[0]
            p_dm = self._format_p(dm.get("p_value_two_tailed", 1.0)).split()[0] if dm else "—"
            return delta_str, p_tt, p_dm

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Statistical Significance of Carbon Flux Model Comparisons}",
            r"\begin{tabular}{llccc}",
            r"\hline",
            r"Site & Comparison & $\Delta$ RMSE & t-test & DM test \\",
            r"\hline",
        ]

        site_labels = {"UK-AMo": "UK-AMo", "SE-Htm": "SE-Htm"}
        primary_pairs = [
            ("TEMPO Fine-Tuned", "XGBoost"),
            ("TEMPO Fine-Tuned", "Random Forest"),
            ("TEMPO Fine-Tuned", "LSTM"),
            ("TEMPO Fine-Tuned", "TEMPO Zero-Shot"),
            ("TEMPO Zero-Shot", "XGBoost"),
        ]

        for si, site in enumerate(SITES):
            if site not in self.ttest_results:
                continue
            first = True
            site_label = site_labels[site]
            for m1, m2 in primary_pairs:
                label = f"{m1} vs {m2}"
                if label not in self.ttest_results.get(site, {}):
                    # try reverse
                    label = f"{m2} vs {m1}"
                if label not in self.ttest_results.get(site, {}):
                    continue
                delta_str, p_tt, p_dm = fmt_rmse(site, label)
                site_col = site_label if first else ""
                cmp_short = label.replace("TEMPO Fine-Tuned", "TEMPO FT")
                lines.append(
                    f"{site_col} & {cmp_short} & {delta_str} & {p_tt} & {p_dm} \\\\"
                )
                first = False
            if si < len(SITES) - 1:
                lines.append(r"\hline")

        lines += [
            r"\hline",
            r"\multicolumn{5}{l}{\footnotesize *p<0.05, **p<0.01, ***p<0.001. "
            r"$\Delta$RMSE = RMSE(model 1) $-$ RMSE(model 2).} \\",
            r"\end{tabular}",
            r"\label{tab:significance}",
            r"\end{table}",
        ]

        out = OUTPUT_DIR / "significance_table.tex"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global OUTPUT_DIR  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Statistical significance testing for carbon flux forecast models."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Root directory for output files",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Statistical Significance Testing — Carbon Flux Models")
    print("=" * 60)
    print(f"  alpha={args.alpha}  |  Bootstrap samples={args.bootstrap_samples}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")

    analyzer = StatisticalAnalyzer(
        alpha=args.alpha,
        bootstrap_samples=args.bootstrap_samples,
    )

    print("\n[1/3] Loading predictions & running statistical tests ...")
    analyzer.run_all_tests()

    print("\n[2/3] Generating figures ...")
    analyzer.generate_figures()

    print("\n[3/3] Saving results ...")
    analyzer.save_results()

    print("\n" + "=" * 60)
    print("  Done. All outputs written.")
    print("=" * 60)


if __name__ == "__main__":
    main()
