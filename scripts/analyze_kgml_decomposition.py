"""
Knowledge-Guided Decomposition Analysis for Carbon Flux Thesis
================================================================

Applies STL decomposition to NEE time series from held-out test sites,
validates ecological patterns, and compares with TEMPO's implicit
decomposition of the forecast signal.

Outputs:
- 4-panel decomposition plots (original, trend, seasonal, residual)
- Ecological validation reports
- TEMPO decomposition comparison figures (if predictions available)
- JSON with decomposition statistics

Usage:
    python scripts/analyze_kgml_decomposition.py --site UK-AMo
    python scripts/analyze_kgml_decomposition.py --site SE-Htm
    python scripts/analyze_kgml_decomposition.py --all
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DECOMP_DIR = PROJECT_ROOT / "results" / "analysis" / "decomposition"

SITE_FILES = {
    "UK-AMo": "6.UK-AMo.csv",
    "SE-Htm": "7.SE-Htm.csv",
}

# Ecological metadata per site
SITE_INFO = {
    "UK-AMo": {
        "name": "Auchencorth Moss",
        "country": "Scotland, UK",
        "ecosystem": "Peatland / Bog",
        "expected_sink": "Weak sink (growing season) / source (winter)",
        "latitude": 55.79,
    },
    "SE-Htm": {
        "name": "Hyltemossa",
        "country": "Sweden",
        "ecosystem": "Boreal coniferous forest",
        "expected_sink": "Strong sink (summer) / near-neutral (winter)",
        "latitude": 56.10,
    },
}

HORIZON = 96
LOOKBACK = 336


# ===========================================================================
# KGMLAnalysis class
# ===========================================================================
class KGMLAnalysis:
    """
    Knowledge-Guided Machine Learning decomposition analysis for
    carbon flux (NEE) time series.

    Applies STL decomposition, validates results against known
    ecological patterns, and compares with TEMPO model predictions.
    """

    def __init__(self):
        DECOMP_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load raw NEE
    # -----------------------------------------------------------------------
    def load_raw_nee(self, site_name):
        """Load the full raw NEE time series with timestamps."""
        filepath = RAW_DIR / SITE_FILES[site_name]
        df = pd.read_csv(filepath)
        ts_col = "timestamp" if "timestamp" in df.columns else "TIMESTAMP"
        df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True)
        df = df.set_index(ts_col)
        nee = df["NEE_VUT_REF"].ffill().bfill().astype(np.float32)
        doy = df["DOY"].values if "DOY" in df.columns else None
        tod = df["TOD"].values if "TOD" in df.columns else None
        return nee, doy, tod

    # -----------------------------------------------------------------------
    # STL decomposition
    # -----------------------------------------------------------------------
    def analyze_decomposition(self, nee_series, site_name, period=24):
        """
        Apply STL decomposition to a NEE time series.

        Args:
            nee_series: pd.Series with datetime index, or 1-D array
            site_name: site identifier for labelling
            period: seasonal period in timesteps (24 = diurnal for hourly)

        Returns:
            dict with trend, seasonal, residual arrays and statistics
        """
        if isinstance(nee_series, np.ndarray):
            nee_series = pd.Series(
                nee_series, index=pd.RangeIndex(len(nee_series))
            )

        print(f"  Running STL decomposition (period={period})...")
        stl = STL(nee_series, period=period, robust=True)
        result = stl.fit()

        trend = result.trend.values
        seasonal = result.seasonal.values
        residual = result.resid.values

        # Component statistics
        total_var = np.var(nee_series.values)
        stats = {
            "site": site_name,
            "n_timesteps": len(nee_series),
            "period": period,
            "total_variance": float(total_var),
            "trend_variance": float(np.var(trend)),
            "seasonal_variance": float(np.var(seasonal)),
            "residual_variance": float(np.var(residual)),
            "trend_pct": float(np.var(trend) / total_var * 100) if total_var > 0 else 0,
            "seasonal_pct": float(np.var(seasonal) / total_var * 100) if total_var > 0 else 0,
            "residual_pct": float(np.var(residual) / total_var * 100) if total_var > 0 else 0,
            "trend_range": float(np.ptp(trend)),
            "seasonal_amplitude": float(np.ptp(seasonal)),
            "residual_std": float(np.std(residual)),
        }

        print(f"    Variance explained — Trend: {stats['trend_pct']:.1f}%, "
              f"Seasonal: {stats['seasonal_pct']:.1f}%, "
              f"Residual: {stats['residual_pct']:.1f}%")

        # Generate 4-panel plot
        self._plot_decomposition(nee_series, trend, seasonal, residual,
                                 site_name, stats)

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": nee_series.values,
            "stats": stats,
            "stl_result": result,
        }

    def _plot_decomposition(self, original, trend, seasonal, residual,
                            site_name, stats):
        """Generate 4-panel decomposition figure."""
        info = SITE_INFO.get(site_name, {})
        title = f"{site_name} ({info.get('name', '')}) — NEE STL Decomposition"

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        n = len(original)
        x = np.arange(n)

        # Subsample for plotting if very long
        step = max(1, n // 5000)
        xs = x[::step]

        panels = [
            ("Original NEE", original.values[::step] if hasattr(original, "values") else original[::step], "k"),
            ("Trend", trend[::step], "steelblue"),
            (f"Seasonal (period={stats['period']}h)", seasonal[::step], "coral"),
            ("Residual", residual[::step], "gray"),
        ]

        for ax, (label, data, color) in zip(axes, panels):
            ax.plot(xs, data, color=color, linewidth=0.5, alpha=0.8)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.2)

            # Add variance annotation
            var_pct = stats.get(f"{label.split()[0].lower()}_pct", "")
            if var_pct:
                ax.text(
                    0.98, 0.92, f"{var_pct:.1f}% of variance",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="white", alpha=0.8)
                )

        axes[0].set_title(title, fontsize=13)
        axes[-1].set_xlabel("Timestep (hours)", fontsize=10)

        plt.tight_layout()
        path = DECOMP_DIR / f"stl_decomposition_{site_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Decomposition plot saved: {path.name}")

    # -----------------------------------------------------------------------
    # Ecological validation
    # -----------------------------------------------------------------------
    def validate_ecological_patterns(self, decomp_result, site_name,
                                     tod=None, doy=None):
        """
        Validate decomposition against known ecological patterns.

        Checks:
        1. Diurnal cycle: seasonal component should peak during daytime
           (NEE negative = carbon uptake during photosynthesis hours)
        2. Seasonal trend: carbon sink in growing season, source in winter
        3. Residual normality and autocorrelation

        Returns:
            dict with validation results and pass/fail flags
        """
        seasonal = decomp_result["seasonal"]
        trend = decomp_result["trend"]
        residual = decomp_result["residual"]
        info = SITE_INFO.get(site_name, {})

        report = {
            "site": site_name,
            "ecosystem": info.get("ecosystem", "Unknown"),
            "checks": {},
        }

        # --- Check 1: Diurnal pattern ---
        # Extract one average diurnal cycle from the seasonal component
        period = decomp_result["stats"]["period"]
        n_full_cycles = len(seasonal) // period
        if n_full_cycles > 0:
            # Reshape into complete cycles and average
            trimmed = seasonal[: n_full_cycles * period]
            diurnal_avg = trimmed.reshape(n_full_cycles, period).mean(axis=0)

            # For hourly data: hours 6-18 are daytime
            day_hours = slice(6, 18)
            night_hours = list(range(0, 6)) + list(range(18, min(period, 24)))

            day_mean = np.mean(diurnal_avg[day_hours])
            night_mean = np.mean(diurnal_avg[night_hours])

            # NEE convention: negative = uptake (photosynthesis)
            # During day, seasonal component should be more negative
            daytime_lower = day_mean < night_mean
            amplitude = float(np.ptp(diurnal_avg))

            report["checks"]["diurnal_pattern"] = {
                "pass": bool(daytime_lower),
                "day_mean_nee": float(day_mean),
                "night_mean_nee": float(night_mean),
                "diurnal_amplitude": amplitude,
                "description": (
                    "PASS: Daytime NEE lower (more uptake) than nighttime"
                    if daytime_lower else
                    "FAIL: Expected daytime NEE < nighttime (photosynthesis)"
                ),
            }
            print(f"    Diurnal check: {'PASS' if daytime_lower else 'FAIL'} "
                  f"(day={day_mean:.4f}, night={night_mean:.4f})")
        else:
            report["checks"]["diurnal_pattern"] = {
                "pass": False,
                "description": "SKIP: Insufficient data for diurnal analysis",
            }

        # --- Check 2: Seasonal trend (if DOY available) ---
        if doy is not None and len(doy) == len(trend):
            doy_arr = np.array(doy)
            # Growing season: DOY 120-270 (~May-Sep for Northern Hemisphere)
            growing_mask = (doy_arr >= 120 / 366) & (doy_arr <= 270 / 366)
            dormant_mask = ~growing_mask

            if growing_mask.sum() > 0 and dormant_mask.sum() > 0:
                growing_trend = np.mean(trend[growing_mask])
                dormant_trend = np.mean(trend[dormant_mask])

                # Growing season should show more negative trend (uptake)
                growing_more_negative = growing_trend < dormant_trend

                report["checks"]["seasonal_trend"] = {
                    "pass": bool(growing_more_negative),
                    "growing_season_mean": float(growing_trend),
                    "dormant_season_mean": float(dormant_trend),
                    "description": (
                        "PASS: Growing season shows stronger carbon uptake"
                        if growing_more_negative else
                        "FAIL: Expected growing season NEE < dormant season"
                    ),
                }
                print(f"    Seasonal trend: "
                      f"{'PASS' if growing_more_negative else 'FAIL'} "
                      f"(growing={growing_trend:.4f}, "
                      f"dormant={dormant_trend:.4f})")
            else:
                report["checks"]["seasonal_trend"] = {
                    "pass": False,
                    "description": "SKIP: DOY range insufficient",
                }
        else:
            report["checks"]["seasonal_trend"] = {
                "pass": False,
                "description": "SKIP: DOY not available",
            }

        # --- Check 3: Residual properties ---
        resid_mean = float(np.mean(residual))
        resid_std = float(np.std(residual))
        resid_skew = float(pd.Series(residual).skew())
        resid_kurt = float(pd.Series(residual).kurtosis())

        # Residuals should be approximately zero-mean
        near_zero = abs(resid_mean) < 0.1 * resid_std if resid_std > 0 else True

        # Compute lag-1 autocorrelation
        if len(residual) > 1:
            lag1_corr = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        else:
            lag1_corr = 0.0

        report["checks"]["residual_properties"] = {
            "pass": bool(near_zero),
            "mean": resid_mean,
            "std": resid_std,
            "skewness": resid_skew,
            "kurtosis": resid_kurt,
            "lag1_autocorrelation": lag1_corr,
            "description": (
                "PASS: Residuals approximately zero-mean"
                if near_zero else
                f"WARNING: Residual mean={resid_mean:.4f} may indicate bias"
            ),
        }
        print(f"    Residual check: {'PASS' if near_zero else 'WARN'} "
              f"(mean={resid_mean:.4f}, std={resid_std:.4f}, "
              f"lag1_acf={lag1_corr:.3f})")

        # Overall pass count
        checks_passed = sum(
            1 for c in report["checks"].values() if c.get("pass", False)
        )
        report["checks_passed"] = checks_passed
        report["checks_total"] = len(report["checks"])

        return report

    # -----------------------------------------------------------------------
    # TEMPO decomposition comparison
    # -----------------------------------------------------------------------
    def compare_tempo_decomposition(self, tempo_predictions, actual_nee,
                                    site_name, model_label="TEMPO"):
        """
        Compare TEMPO's forecast signal decomposition with actual NEE
        STL decomposition.

        Both inputs should be 1-D arrays of the same length (flattened
        forecast windows or a continuous segment).

        Args:
            tempo_predictions: 1-D array of TEMPO forecast values
            actual_nee: 1-D array of actual NEE values
            site_name: site identifier
            model_label: label for the model (e.g. "TEMPO Zero-Shot")

        Returns:
            dict with per-component comparison metrics
        """
        n = min(len(tempo_predictions), len(actual_nee))
        tempo_seg = tempo_predictions[:n]
        actual_seg = actual_nee[:n]

        # Need enough data for STL (at least 2 full cycles)
        period = 24
        if n < 2 * period:
            print(f"    WARNING: Segment too short for STL ({n} < {2*period})")
            return None

        print(f"  Decomposing actual vs {model_label} "
              f"({n} timesteps)...")

        # Decompose actual
        stl_actual = STL(
            pd.Series(actual_seg), period=period, robust=True
        ).fit()

        # Decompose TEMPO predictions
        stl_pred = STL(
            pd.Series(tempo_seg), period=period, robust=True
        ).fit()

        # Per-component error metrics
        components = {}
        for comp_name in ["trend", "seasonal", "resid"]:
            actual_comp = getattr(stl_actual, comp_name).values
            pred_comp = getattr(stl_pred, comp_name).values

            rmse = float(np.sqrt(np.mean((actual_comp - pred_comp) ** 2)))
            corr = float(np.corrcoef(actual_comp, pred_comp)[0, 1])

            components[comp_name] = {"rmse": rmse, "correlation": corr}
            print(f"    {comp_name:>10}: RMSE={rmse:.4f}, corr={corr:.4f}")

        # Generate comparison figure
        self._plot_tempo_comparison(
            stl_actual, stl_pred, site_name, model_label
        )

        return components

    def _plot_tempo_comparison(self, stl_actual, stl_pred, site_name,
                               model_label):
        """Side-by-side decomposition comparison plot."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        n = len(stl_actual.trend)
        x = np.arange(n)
        step = max(1, n // 3000)
        xs = x[::step]

        comp_pairs = [
            ("Trend", stl_actual.trend.values, stl_pred.trend.values),
            ("Seasonal", stl_actual.seasonal.values, stl_pred.seasonal.values),
            ("Residual", stl_actual.resid.values, stl_pred.resid.values),
        ]

        for ax, (label, actual, pred) in zip(axes, comp_pairs):
            ax.plot(xs, actual[::step], "k-", linewidth=0.7,
                    label="Actual", alpha=0.8)
            ax.plot(xs, pred[::step], "r-", linewidth=0.7,
                    label=model_label, alpha=0.7)
            ax.set_ylabel(label, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.2)

            corr = np.corrcoef(actual, pred)[0, 1]
            ax.text(
                0.02, 0.92, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.8)
            )

        axes[0].set_title(
            f"{site_name} — STL Component Comparison: Actual vs {model_label}",
            fontsize=13,
        )
        axes[-1].set_xlabel("Timestep (hours)", fontsize=10)

        plt.tight_layout()
        safe_label = model_label.lower().replace(" ", "_")
        path = DECOMP_DIR / f"tempo_comparison_{safe_label}_{site_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Comparison plot saved: {path.name}")

    # -----------------------------------------------------------------------
    # Diurnal cycle plot
    # -----------------------------------------------------------------------
    def plot_diurnal_cycle(self, seasonal, site_name, period=24):
        """Plot the average diurnal NEE cycle from the seasonal component."""
        n_cycles = len(seasonal) // period
        if n_cycles == 0:
            return

        trimmed = seasonal[: n_cycles * period]
        cycles = trimmed.reshape(n_cycles, period)
        mean_cycle = cycles.mean(axis=0)
        std_cycle = cycles.std(axis=0)
        hours = np.arange(period)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hours, mean_cycle, "steelblue", linewidth=2, label="Mean")
        ax.fill_between(
            hours, mean_cycle - std_cycle, mean_cycle + std_cycle,
            alpha=0.2, color="steelblue", label="+/- 1 SD"
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvspan(6, 18, alpha=0.08, color="gold", label="Daytime (6-18h)")

        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Seasonal NEE Component", fontsize=11)
        ax.set_title(
            f"{site_name} — Average Diurnal NEE Cycle "
            f"({SITE_INFO.get(site_name, {}).get('ecosystem', '')})",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = DECOMP_DIR / f"diurnal_cycle_{site_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Diurnal cycle plot saved: {path.name}")

    # -----------------------------------------------------------------------
    # PDF validation report
    # -----------------------------------------------------------------------
    def generate_validation_report(self, decomp_result, validation_report,
                                   site_name):
        """Generate a PDF report with decomposition statistics."""
        path = DECOMP_DIR / f"validation_report_{site_name}.pdf"
        info = SITE_INFO.get(site_name, {})
        stats = decomp_result["stats"]
        checks = validation_report["checks"]

        with PdfPages(path) as pdf:
            # Page 1: Summary statistics
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis("off")

            lines = [
                f"KGML Decomposition Validation Report",
                f"{'='*50}",
                f"",
                f"Site: {site_name} ({info.get('name', 'N/A')})",
                f"Ecosystem: {info.get('ecosystem', 'N/A')}",
                f"Location: {info.get('country', 'N/A')} "
                f"(lat {info.get('latitude', 'N/A')})",
                f"Timesteps: {stats['n_timesteps']:,}",
                f"STL Period: {stats['period']} hours",
                f"",
                f"Variance Decomposition:",
                f"  Trend:    {stats['trend_pct']:6.1f}%  "
                f"(range: {stats['trend_range']:.4f})",
                f"  Seasonal: {stats['seasonal_pct']:6.1f}%  "
                f"(amplitude: {stats['seasonal_amplitude']:.4f})",
                f"  Residual: {stats['residual_pct']:6.1f}%  "
                f"(std: {stats['residual_std']:.4f})",
                f"",
                f"Ecological Validation "
                f"({validation_report['checks_passed']}/"
                f"{validation_report['checks_total']} passed):",
            ]

            for check_name, check in checks.items():
                status = "PASS" if check.get("pass") else "FAIL"
                lines.append(f"  [{status}] {check_name}: "
                             f"{check.get('description', '')}")

            text = "\n".join(lines)
            ax.text(
                0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                fontfamily="monospace",
            )
            pdf.savefig(fig)
            plt.close()

            # Page 2: Variance pie chart
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sizes = [stats["trend_pct"], stats["seasonal_pct"],
                     stats["residual_pct"]]
            labels = ["Trend", "Seasonal", "Residual"]
            colors = ["steelblue", "coral", "gray"]

            axes[0].pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90
            )
            axes[0].set_title(f"{site_name} — Variance Decomposition")

            # Component magnitudes bar chart
            names = ["Total", "Trend", "Seasonal", "Residual"]
            variances = [
                stats["total_variance"], stats["trend_variance"],
                stats["seasonal_variance"], stats["residual_variance"],
            ]
            bars = axes[1].bar(names, variances,
                               color=["black"] + colors, alpha=0.8)
            axes[1].set_ylabel("Variance")
            axes[1].set_title(f"{site_name} — Component Variances")
            for bar, v in zip(bars, variances):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9,
                )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        print(f"    Validation report saved: {path.name}")


# ===========================================================================
# Main analysis pipeline
# ===========================================================================
def load_predictions(site_name):
    """Load available model predictions for a site."""
    preds = {}
    pred_files = {
        "TEMPO Zero-Shot": f"tempo_zero_shot_preds_{site_name}.npy",
        "TEMPO Fine-Tuned": f"tempo_fine_tuned_preds_{site_name}.npy",
        "Random Forest": f"randomforest_preds_{site_name}.npy",
        "XGBoost": f"xgboost_preds_{site_name}.npy",
        "LSTM": f"lstm_preds_{site_name}.npy",
    }
    for label, fname in pred_files.items():
        path = PREDICTIONS_DIR / fname
        if path.exists():
            preds[label] = np.load(path)
            print(f"    Loaded {label}: {preds[label].shape}")
    return preds


def analyze_site(analyzer, site_name):
    """Run full decomposition analysis for a single site."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {site_name} "
          f"({SITE_INFO.get(site_name, {}).get('name', '')})")
    print(f"{'='*60}")

    # Load raw NEE
    print("\n  Loading raw NEE time series...")
    nee_series, doy, tod = analyzer.load_raw_nee(site_name)
    print(f"    {len(nee_series)} timesteps loaded")

    # STL decomposition on full series
    print("\n  STL decomposition on full time series...")
    decomp = analyzer.analyze_decomposition(nee_series, site_name, period=24)

    # Diurnal cycle plot
    print("\n  Extracting diurnal cycle...")
    analyzer.plot_diurnal_cycle(decomp["seasonal"], site_name, period=24)

    # Ecological validation
    print("\n  Validating ecological patterns...")
    validation = analyzer.validate_ecological_patterns(
        decomp, site_name, tod=tod, doy=doy
    )

    # PDF report
    print("\n  Generating validation report...")
    analyzer.generate_validation_report(decomp, validation, site_name)

    # Load targets for forecast-window analysis
    targets_path = PREDICTIONS_DIR / f"targets_{site_name}.npy"
    if not targets_path.exists():
        targets_path = (PREDICTIONS_DIR / "baselines"
                        / f"targets_{site_name}.npy")

    if targets_path.exists():
        targets = np.load(targets_path)  # (N, 96)
        print(f"\n  Loaded forecast targets: {targets.shape}")

        # Load available predictions
        preds = load_predictions(site_name)

        # Flatten first 500 windows into a continuous segment for comparison
        n_windows = min(500, len(targets))
        actual_flat = targets[:n_windows].flatten()

        comparison_metrics = {}
        for label, pred_arr in preds.items():
            if "TEMPO" not in label:
                continue  # Only compare TEMPO variants
            pred_flat = pred_arr[:n_windows].flatten()
            print(f"\n  Comparing {label} decomposition...")
            comp = analyzer.compare_tempo_decomposition(
                pred_flat, actual_flat, site_name, model_label=label
            )
            if comp is not None:
                comparison_metrics[label] = comp
    else:
        print(f"\n  No target predictions found for {site_name}, "
              "skipping TEMPO comparison")
        comparison_metrics = {}

    # Collect all results
    site_results = {
        "decomposition_stats": decomp["stats"],
        "validation": validation,
    }
    if comparison_metrics:
        site_results["tempo_comparison"] = comparison_metrics

    return site_results


def main():
    parser = argparse.ArgumentParser(
        description="KGML decomposition analysis for carbon flux thesis"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--site", choices=["UK-AMo", "SE-Htm"],
                       help="Analyze a single test site")
    group.add_argument("--all", action="store_true",
                       help="Analyze both test sites")
    args = parser.parse_args()

    sites = ["UK-AMo", "SE-Htm"] if args.all else [args.site]

    print("=" * 60)
    print("KGML DECOMPOSITION ANALYSIS")
    print("=" * 60)

    analyzer = KGMLAnalysis()
    all_results = {}

    for site in sites:
        all_results[site] = analyze_site(analyzer, site)

    # Save combined metrics
    metrics_path = METRICS_DIR / "decomposition_metrics.json"

    # Convert any non-serializable values
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        return obj

    with open(metrics_path, "w") as f:
        json.dump(sanitize(all_results), f, indent=2)
    print(f"\nAll metrics saved: {metrics_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    for site, res in all_results.items():
        stats = res["decomposition_stats"]
        val = res["validation"]
        print(f"\n  {site} ({SITE_INFO[site]['ecosystem']}):")
        print(f"    Variance — Trend: {stats['trend_pct']:.1f}%, "
              f"Seasonal: {stats['seasonal_pct']:.1f}%, "
              f"Residual: {stats['residual_pct']:.1f}%")
        print(f"    Ecological checks: {val['checks_passed']}/"
              f"{val['checks_total']} passed")

        if "tempo_comparison" in res:
            for label, comp in res["tempo_comparison"].items():
                print(f"    {label}:")
                print(f"      Trend corr:    {comp['trend']['correlation']:.3f}")
                print(f"      Seasonal corr: "
                      f"{comp['seasonal']['correlation']:.3f}")

    print(f"\nOutputs saved to: {DECOMP_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
