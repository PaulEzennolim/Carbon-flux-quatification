# Temporal-spatial data fusion for carbon flux quatification in agroecosystems: a multimodal learning approach
## Deep Learning for Carbon Flux Forecasting: Foundation Models vs Traditional Approaches

**BSc Environmental Science Thesis** | 2025–2026

> Evaluating the TEMPO-80M time-series foundation model against LSTM, XGBoost, and Random Forest
> baselines for net ecosystem exchange (NEE) prediction across European FLUXNET eddy-covariance sites,
> with novel findings on transfer learning, ensemble methods, and active learning strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Key Findings](#key-findings)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Reproducing Results](#reproducing-results)
9. [Usage Examples](#usage-examples)
10. [Results Summary](#results-summary)
11. [Figures](#figures)
12. [Statistical Rigor](#statistical-rigor)
13. [Computational Efficiency](#computational-efficiency)
14. [Future Work](#future-work)
15. [Citation](#citation)
16. [License](#license)
17. [Acknowledgments](#acknowledgments)
18. [Contact](#contact)
19. [References](#references)

---

## Overview

### Problem Statement

Accurate forecasting of net ecosystem exchange (NEE) — the net carbon dioxide flux between terrestrial
ecosystems and the atmosphere — is critical for understanding the global carbon cycle and projecting
climate change trajectories. Traditional machine learning approaches (Random Forest, XGBoost, LSTM)
require extensive site-specific training data and struggle to generalise across ecosystem types.
Foundation models pre-trained on large, diverse time-series corpora offer a potential paradigm shift,
but their transfer characteristics across ecosystem boundaries remain poorly understood.

### Research Motivation

This thesis addresses three open questions in the computational ecology literature:

1. Can foundation models pretrained on general time-series data achieve competitive or superior
   performance on domain-specific carbon flux forecasting without fine-tuning?
2. Does the conventional wisdom that same-domain transfer outperforms cross-domain transfer hold
   for ecosystem carbon flux prediction?
3. Can uncertainty-aware active learning meaningfully inform future eddy-covariance measurement
   campaigns?

### Novel Contributions

| Contribution | Description |
|---|---|
| **Inverted Transfer Penalty** | TEMPO achieves 21.6% *better* performance on a cross-ecosystem test site (forest) than on the same-ecosystem site (wetland), contradicting established transfer learning theory |
| **Negative Transfer Quantification** | Systematic quantification: 33/60 ensemble configurations (55%) exhibit negative transfer, with effect sizes up to −1.6% R² |
| **Honest Ensemble Failure Analysis** | First rigorous documentation that model-diversity-insufficient ensembles actively degrade performance on challenging sites |
| **Active Learning Protocol** | Data-driven recommendations for field campaign prioritisation based on ensemble uncertainty spatial analysis |
| **Foundation Model Benchmarking** | Comprehensive head-to-head comparison of zero-shot vs fine-tuned TEMPO against three baselines across two contrasting ecosystems |

---

## Research Questions

### Q1 — Foundation Model vs Traditional Baselines

> Does TEMPO-80M, in zero-shot or fine-tuned configuration, achieve statistically superior NEE
> prediction performance compared to XGBoost, Random Forest, and LSTM on held-out FLUXNET sites?

**Hypothesis:** Fine-tuned TEMPO will outperform all baselines on both test sites, with zero-shot
TEMPO competitive but inferior to XGBoost on the same-ecosystem (wetland) site.

---

### Q2 — Transfer Learning Across Ecosystems

> Does transfer learning effectiveness degrade predictably when the source and target ecosystem
> types differ, and can this degradation be quantified as a transfer penalty?

**Hypothesis:** Cross-ecosystem transfer will exhibit a systematic performance penalty, with the
magnitude inversely proportional to ecosystem functional similarity.

---

### Q3 — Ensemble Methods and Diversity

> Do heterogeneous model ensembles (combining TEMPO, XGBoost, RF, LSTM) consistently improve
> NEE forecast accuracy, and what conditions govern ensemble success vs failure?

**Hypothesis:** Ensembles will improve accuracy when constituent model errors are uncorrelated,
but degrade performance when a dominant model is diluted by lower-quality predictions.

---

### Q4 — Ecosystem-Specific Conditioning

> Does providing ecosystem-type metadata as a conditioning prompt to TEMPO improve zero-shot
> prediction accuracy, and does this benefit vary by ecosystem?

**Hypothesis:** Ecosystem conditioning will provide a consistent but modest accuracy gain,
with greater benefit on ecosystem types well-represented in TEMPO's pretraining corpus.

---

### Q5 — Active Learning for Data Collection

> Can ensemble-based uncertainty quantification identify meteorological regimes and temporal
> windows that would yield the greatest reduction in forecast uncertainty if additionally sampled?

**Hypothesis:** Uncertainty hotspots will cluster in summer high-VPD and high-temperature
regimes, providing actionable guidance for measurement campaign design.

---

## Key Findings

### Finding 1 — Inverted Transfer Penalty *(novel)*

TEMPO-80M fine-tuned on five wetland training sites achieves **R² = 0.728** on the held-out
SE-Htm *forest* site, which is **21.6% higher** than its performance on the same-ecosystem UK-AMo
*wetland* test site (R² = 0.599). This directly inverts the conventional transfer learning
prediction that cross-ecosystem transfer should underperform same-ecosystem transfer.

**Proposed mechanism:** The forest site (SE-Htm) exhibits stronger and more regular diurnal NEE
cycles driven by photosynthetic activity, creating time-series patterns more closely aligned with
TEMPO's broad pretraining distribution. The wetland site's suppressed, irregular fluxes represent
an out-of-distribution regime even for same-ecosystem fine-tuning.

### Finding 2 — Negative Transfer Quantification

Across 60 model × site × configuration combinations evaluated in the transfer learning analysis:

- **33 configurations (55%)** show negative transfer (degraded performance vs. single-site baseline)
- Random Forest exhibits the largest penalty: **+40.6%** RMSE increase on cross-ecosystem transfer
- LSTM shows **+60.8%** RMSE increase, the most vulnerable to ecosystem mismatch
- TEMPO uniquely exhibits **inverted** transfer: −21.6% RMSE (improvement) on cross-ecosystem

### Finding 3 — Ensemble Failure Conditions

On UK-AMo (wetland), weighted ensemble averaging (R² = 0.433) underperforms the best constituent
model TEMPO fine-tuned (R² = 0.599) by **−27.7%**. On SE-Htm (forest), ensemble provides no
improvement over TEMPO zero-shot (R² = 0.741 vs 0.741). Ensemble benefit only materialises when:

1. Model errors are negatively correlated across the ensemble
2. No single model has substantially higher accuracy than all others
3. Training domain diversity is sufficient to produce complementary error structures

### Finding 4 — Learning Curve Saturation

Random Forest and XGBoost performance saturates at approximately **75–80% of available training
data** (R² within 5% of asymptote), implying that additional data collection yields diminishing
returns. Cross-site diversity contributes more to baseline model performance than within-site
sample density.

### Finding 5 — Seasonal Uncertainty Dominance

Active learning analysis reveals that summer uncertainty exceeds winter uncertainty by
**≈4.1×** across both sites, while diurnal uncertainty is nearly uniform (nocturnal/midday ratio
≈1.0×). The dominant uncertainty driver is seasonal and meteorological regime, not time of day.

---

## Dataset

### FLUXNET2015 Sites

| Role | Site Code | Ecosystem | Country | MAT (°C) | MAP (mm) |
|---|---|---|---|---|---|
| Training | FI-Lom | Wetland (fen) | Finland | 0.1 | 554 |
| Training | GL-ZaF | Wetland (fen) | Greenland | −5.0 | 240 |
| Training | IE-Cra | Wetland (fen) | Ireland | 9.9 | 854 |
| Training | DE-Akm | Wetland (fen) | Germany | 9.3 | 593 |
| Training | FR-LGt | Wetland (fen) | France | 11.2 | 900 |
| **Test** | **UK-AMo** | **Wetland (blanket bog)** | **UK** | **5.7** | **1395** |
| **Test** | **SE-Htm** | **Forest (hemiboreal)** | **Sweden** | **6.8** | **707** |

All sites provide half-hourly observations aggregated to hourly resolution. Data spans 2–10 years
per site depending on FLUXNET2015 Tier 1 availability.

### Input Features (19 variables)

| Category | Variables |
|---|---|
| Energy balance | SW_IN, LW_IN, NETRAD, G_F_MDS |
| Meteorology | TA_F, PA_F, P_F, WS_F, WD |
| Humidity | RH, VPD_F |
| Carbon | CO2_F_MDS, NEE_VUT_REF (target) |
| Water | LE_F_MDS, ET |
| Soil | TS_F_MDS_1, SWC_F_MDS_1 |
| Temporal | DOY, TOD |

### Preprocessing

1. **Resampling**: Half-hourly → hourly mean aggregation
2. **Gap filling**: Forward-fill up to 3 hours; longer gaps removed
3. **Normalisation**: Z-score standardisation per site using training-period statistics only
   (no data leakage to test period)
4. **Sequence construction**: Sliding window — 336-hour lookback → 96-hour forecast horizon
5. **Train/test split**: Chronological; final year of each site used as site-level validation;
   UK-AMo and SE-Htm held out entirely until final evaluation

### Missing Data Policy

- Gaps ≤ 3 hours: linear interpolation
- Gaps 4–24 hours: forward-fill from last valid observation
- Gaps > 24 hours: sequence excluded from training/evaluation
- Night-time NEE quality flags: USTAR-filtered values used (NEE_VUT_REF)

---

## Methodology

### Models Evaluated

| Model | Type | Parameters | Training strategy |
|---|---|---|---|
| **TEMPO-80M** | Foundation (transformer) | 80M | Zero-shot & fine-tuned (20 epochs) |
| **XGBoost** | Gradient boosting | ~50K | Trained per-site with optuna HPO |
| **Random Forest** | Ensemble trees | ~10M | Trained per-site, 200 estimators |
| **LSTM** | Recurrent neural net | ~2M | Trained per-site, 2 layers × 128 units |

### Experimental Configuration

```
Lookback window  : 336 time steps (14 days at 1-hour resolution)
Forecast horizon : 96 time steps  (4 days)
Batch size       : 32
Learning rate    : 1e-4 (TEMPO fine-tuning, cosine decay)
Optimiser        : AdamW (TEMPO), default (XGBoost/RF)
Random seed      : 42 (all experiments)
```

### Evaluation Metrics

| Metric | Symbol | Notes |
|---|---|---|
| Coefficient of determination | R² | Primary accuracy metric |
| Root mean squared error | RMSE | μmol m⁻² s⁻¹ |
| Mean absolute error | MAE | μmol m⁻² s⁻¹ |
| Computational time | t (s) | Wall-clock inference time |

### Statistical Testing

- **Bootstrap confidence intervals**: n = 1000 resamples; 95% CI reported for all R² values
- **Paired t-tests**: Model comparison within site; p < 0.05 threshold
- **Bonferroni correction**: Applied for multiple comparisons across 7 model × metric combinations
- **Effect size**: Cohen's d reported for all significant pairwise comparisons

---

## Repository Structure

```
Carbon-flux-quatification/
│
├── data/
│   ├── raw/                      # Original FLUXNET2015 CSV files (not tracked by git)
│   └── processed/                # Preprocessed .npy sequence arrays (not tracked by git)
│
├── models/
│   ├── baseline_models.py        # XGBoost and Random Forest wrappers
│   ├── lstm_baseline.py          # LSTM architecture and training loop
│   ├── tempo_carbon_flux.py      # TEMPO-80M zero-shot and fine-tuning pipeline
│   └── checkpoints/              # Trained model weights (not tracked by git)
│
├── scripts/
│   ├── tempo_data_prep.py        # FLUXNET → TEMPO-format sequence builder
│   ├── run_zero_shot_tempo.py    # Zero-shot inference across all test sites
│   ├── fine_tune_tempo.py        # Fine-tuning pipeline with early stopping
│   ├── train_baselines.py        # XGBoost / RF / LSTM training and evaluation
│   ├── uncertainty_quantification.py   # Ensemble uncertainty decomposition
│   ├── transfer_learning_analysis.py   # Cross-site / cross-ecosystem transfer
│   ├── ensemble_models.py        # Weighted ensemble construction and evaluation
│   ├── ecosystem_prompting.py    # TEMPO ecosystem-conditioning experiments
│   ├── active_learning_analysis.py     # Uncertainty-based sampling strategy
│   ├── error_analysis.py         # Residual and error pattern analysis
│   ├── feature_importance.py     # SHAP-based feature importance
│   ├── statistical_analysis.py   # Bootstrap CI and hypothesis testing
│   ├── computational_efficiency.py     # Runtime and memory benchmarking
│   ├── horizon_analysis.py       # Forecast degradation over horizon length
│   ├── analyze_kgml_decomposition.py   # STL decomposition of NEE time series
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: flux distributions, seasonality
│   ├── 02_baseline_experiments.ipynb  # Interactive baseline training and tuning
│   └── 03_tempo_analysis.ipynb        # TEMPO zero-shot vs fine-tuned comparison
│
├── results/
│   ├── active_learning/          # Priority conditions, learning curves, summaries
│   ├── ensemble/                 # Ensemble weights and accuracy tables
│   ├── transfer_learning/        # Cross-site transfer matrices
│   └── uncertainty/              # Uncertainty decomposition CSVs
│
├── figures/                      # Plots (300 DPI, PDF/PNG)
│
├── .gitignore                    # Excludes data, checkpoints, large binaries
├── .gitattributes                # LFS-ready binary attributes; Unix line endings
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- `conda` (recommended) or `venv`
- GPU optional but recommended for TEMPO fine-tuning (CPU inference supported)

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/PaulEzennolim/Carbon-flux-quatification.git
cd Carbon-flux-quatification

# 2. Create and activate conda environment
conda create -n tempo python=3.10 -y
conda activate tempo

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify core imports
python -c "import torch, numpy, pandas, xgboost, momentfm; print('All OK')"
```

### FLUXNET2015 Data Access

FLUXNET2015 data requires a free account at [fluxnet.org](https://fluxnet.org).

1. Register at [fluxnet.org/data/fluxnet2015-dataset](https://fluxnet.org/data/fluxnet2015-dataset)
2. Download Tier 1 FULLSET (hourly) for sites: FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt, UK-AMo, SE-Htm
3. Place raw CSV files in `data/raw/` (format: `FLX_<SITE>_FLUXNET2015_FULLSET_HH_*.csv`)

### TEMPO Model Weights

TEMPO-80M weights are downloaded automatically by `momentfm` on first run:

```bash
python -c "from momentfm import MOMENTPipeline; MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large')"
```

Cached in `models/checkpoints/` (excluded from git via `.gitignore`).

---

## Reproducing Results

Run scripts in the following order. All scripts are self-contained and write outputs to `results/`.

### Step 1 — Data Preparation (~5 min)

```bash
python scripts/tempo_data_prep.py
```

Reads raw FLUXNET CSVs, applies normalisation and sliding-window extraction,
writes processed `.npy` sequence arrays to `data/processed/`.

### Step 2 — Baseline Training (~10 min, CPU)

```bash
python scripts/train_baselines.py
```

Trains XGBoost, Random Forest, and LSTM on the five training sites.
Evaluates on UK-AMo and SE-Htm. Writes metrics to `results/`.

### Step 3 — TEMPO Zero-Shot Inference (~3 min, CPU / <1 min GPU)

```bash
python scripts/run_zero_shot_tempo.py
```

Runs TEMPO-80M in zero-shot mode on all seven sites. No training required.

### Step 4 — TEMPO Fine-Tuning (~30 min CPU / ~5 min GPU)

```bash
python scripts/fine_tune_tempo.py --epochs 20 --lr 1e-4
```

Fine-tunes TEMPO on the five training sites; evaluates on held-out test sites.

### Step 5 — Analysis Scripts (any order after Steps 1–4)

```bash
# Uncertainty quantification
python scripts/uncertainty_quantification.py

# Transfer learning analysis
python scripts/transfer_learning_analysis.py

# Ensemble models
python scripts/ensemble_models.py

# Ecosystem prompting / conditioning
python scripts/ecosystem_prompting.py

# Active learning recommendations
python scripts/active_learning_analysis.py

# Computational efficiency benchmarking
python scripts/computational_efficiency.py

# Statistical analysis (CI, p-values)
python scripts/statistical_analysis.py
```

### Expected Runtimes (Apple M4, 24 GB RAM)

| Script | CPU runtime | GPU runtime |
|---|---|---|
| `tempo_data_prep.py` | ~5 min | — |
| `train_baselines.py` | ~10 min | — |
| `run_zero_shot_tempo.py` | ~3 min | <1 min |
| `fine_tune_tempo.py` | ~30 min | ~5 min |
| `uncertainty_quantification.py` | ~5 min | — |
| `transfer_learning_analysis.py` | ~8 min | — |
| `ensemble_models.py` | ~4 min | — |
| `active_learning_analysis.py` | ~6 min | — |

### Reproducibility

All stochastic operations use `random.seed(42)`, `np.random.seed(42)`,
and `torch.manual_seed(42)`. Results should be deterministic on the same hardware.
Minor floating-point differences may appear across architectures.

---

## Usage Examples

```python
# ── Zero-shot TEMPO inference on a new site ─────────────────────────────────
from models.tempo_carbon_flux import TEMPOCarbonFlux

model = TEMPOCarbonFlux.from_pretrained()
predictions = model.predict(X_sequence)   # X: (N, 336, 19)

# ── Fine-tune TEMPO on training sites ────────────────────────────────────────
python scripts/fine_tune_tempo.py --site UK-AMo --epochs 20 --seed 42

# ── Run transfer learning analysis ───────────────────────────────────────────
python scripts/transfer_learning_analysis.py

# ── Generate ensemble predictions ────────────────────────────────────────────
python scripts/ensemble_models.py

# ── Active learning data collection recommendations ──────────────────────────
python scripts/active_learning_analysis.py
```

---

## Results Summary

### Primary Performance Table (R²)

| Model | UK-AMo (Wetland) | SE-Htm (Forest) | Δ (Forest − Wetland) |
|---|---|---|---|
| XGBoost | 0.412 | 0.376 | −0.036 |
| Random Forest | 0.398 | 0.351 | −0.047 |
| LSTM | 0.387 | 0.329 | −0.058 |
| TEMPO Zero-Shot | 0.521 | **0.741** | +0.220 |
| TEMPO Fine-Tuned | **0.599** | 0.728 | +0.129 |
| Weighted Ensemble | 0.433 | 0.741 | +0.308 |

> Ensemble underperforms best constituent model on both sites — see Finding 3.

### Transfer Learning Transfer Penalty

| Model | Same-ecosystem penalty | Cross-ecosystem penalty | Inverted? |
|---|---|---|---|
| XGBoost | — | +40.6% RMSE | No |
| Random Forest | — | +40.6% RMSE | No |
| LSTM | — | +60.8% RMSE | No |
| **TEMPO** | — | **−21.6% RMSE** | **Yes** |

### Active Learning Priority Conditions

Top-3 high-uncertainty regimes (pooled across sites):

1. **Summer growing season** — 4.1× higher ensemble std than winter baseline
2. **High-VPD conditions** (TA_F Q4, VPD_F Q4) — 3.4–3.6× uncertainty elevation
3. **High soil moisture** — 2.8× uncertainty elevation at SWC_F Q4

---

## Figures

All figures are located in `figures/` and generated at **300 DPI** in both PNG and PDF formats.

| Figure | File | Description |
|---|---|---|
| Model comparison | `all_models_comparison.png` | R² and RMSE across all models and sites |
| Transfer matrix | `transfer_learning/transfer_matrix_heatmap.png` | Cross-site transfer penalty heatmap |
| Ensemble weights | `ensemble/weights_optimization.png` | Optimal ensemble weight distributions |
| Uncertainty maps | `uncertainty/uncertainty_decomposition_SE-Htm uncertainty/uncertainty_decomposition_UK-AMO.png` | Ensemble std vs meteorological bins |
| Learning curves | `active_learning/data_efficiency.png` | R² vs training data fraction |
| Horizon decay | `horizon_analysis/model_comparison_by_horizon.png` | Forecast accuracy vs lead time |
| Temporal uncertainty | `active_learning/temporal_uncertainty.png` | Uncertainty by hour, month, season |

---

## Statistical Rigor

### Confidence Intervals

All reported R² values include 95% bootstrap confidence intervals computed from
n = 1000 stratified resamples of the test set. Example:

```
TEMPO Fine-Tuned on UK-AMo: R² = 0.599 [0.571, 0.628]
```

### Hypothesis Testing

Model comparisons use paired t-tests on per-sequence R² values:

```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(tempo_r2_per_seq, xgboost_r2_per_seq)
```

### Multiple Comparisons

Bonferroni correction applied to all pairwise model comparisons within each site
(k = 6 comparisons → α_corrected = 0.05 / 6 = 0.0083).

### Effect Size

Cohen's d reported for all statistically significant comparisons:

```
TEMPO vs XGBoost on SE-Htm: d = 1.87 (large effect)
```

---

## Computational Efficiency

Benchmarked on Apple M4 (10-core CPU, 24 GB unified memory). GPU benchmarks
use MPS acceleration via PyTorch.

| Model | Inference time (s) | Relative speed | Memory (GB) |
|---|---|---|---|
| XGBoost | 18.4 | 1.0× (baseline) | 0.8 |
| Random Forest | 24.1 | 1.3× | 1.2 |
| LSTM | 38.7 | 2.1× | 1.8 |
| TEMPO (zero-shot) | 156.0 | 8.5× | 6.4 |
| TEMPO (fine-tuned) | 156.0 | 8.5× | 6.4 |

TEMPO's 8.5× inference overhead is justified by its R² advantage of +0.315 over the
best baseline on SE-Htm and +0.187 on UK-AMo, making it the preferred model when
forecast accuracy is the primary objective and compute is not the bottleneck.

---

## Future Work

1. **Ecosystem type expansion** — Extend evaluation to grassland (DE-Tha), cropland (DE-Kli),
   and tundra (RU-Sam) sites to test whether the inverted transfer penalty generalises
2. **Real-time deployment** — Integrate with ICOS near-real-time data streams for operational
   carbon monitoring support
3. **Active learning field campaigns** — Design and execute measurement campaigns informed by
   the uncertainty hotspot analysis from Chapter 5
4. **Satellite integration** — Fuse eddy-covariance tower data with MODIS/Sentinel-2 spectral
   indices to improve spatial generalisation
5. **Longer horizons** — Investigate whether TEMPO's cross-ecosystem advantage persists at
   7-day and 14-day forecast horizons
6. **Mechanistic hybrid models** — Couple TEMPO predictions with process-based models (JSBACH,
   CLM) for physically constrained carbon flux estimation

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@bachelorsthesis{ezennolim2026carbon,
  title   = {Temporal-spatial data fusion for carbon flux quatification in agroecosystems:
          a multimodal learning approach},
  author  = {Ezennolim, Paul},
  year    = {2026},
  school  = {Univeristy of Sheffield},
  type    = {{BSc} Thesis},
  note    = {Code available at https://github.com/PaulEzennolim/Carbon-flux-quatification.git}
}
```

---

## License

| Component | License |
|---|---|
| Source code (`models/`, `scripts/`) | [MIT License](LICENSE) |
| FLUXNET2015 data | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — cite original data providers |
| TEMPO-80M model weights | Apache 2.0 — cite [Goswami et al. 2024] |
| Figures and text | CC BY 4.0 — cite this thesis |

---

## Acknowledgments

- **Thesis supervisor: Professor Po Yang (https://sheffield.ac.uk/cs/people/academic/po-yang)** — for guidance on experimental design and statistical methodology
- **FLUXNET community** — for maintaining the open FLUXNET2015 database and all site
  principal investigators who contributed data
- **AutonLab (CMU)** — for open-sourcing the TEMPO-80M foundation model
- **ICOS RI** — for supporting eddy-covariance infrastructure across European sites
- **Computational resources** — analyses performed on personal hardware (Apple M1 Max);
  no HPC allocation required

---

## Contact

| | |
|---|---|
| **Author** | Paul Ezennolim |
| **Email** | [paulezennolim@icloud.com] |
| **GitHub** | [@PaulEzennolim] |
| **LinkedIn** | [https://www.linkedin.com/in/paul-ezennolim/] |

---

## References

### Foundation Model

- Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024).
  **MOMENT: A Family of Open Time-series Foundation Models.**
  *Proceedings of ICML 2024*. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)

### Baselines

- Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.**
  *Proceedings of KDD 2016*, 785–794.

- Breiman, L. (2001). **Random Forests.** *Machine Learning*, 45(1), 5–32.

- Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory.**
  *Neural Computation*, 9(8), 1735–1780.

### FLUXNET Dataset

- Pastorello, G., et al. (2020). **The FLUXNET2015 dataset and the ONEFlux processing pipeline
  for eddy covariance data.** *Scientific Data*, 7(1), 225.

- Reichstein, M., et al. (2005). **On the separation of net ecosystem exchange into assimilation
  and ecosystem respiration: review and improved algorithm.**
  *Global Change Biology*, 11(9), 1424–1439.

### Transfer Learning

- Pan, S. J., & Yang, Q. (2010). **A Survey on Transfer Learning.**
  *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359.

- Zhuang, F., et al. (2021). **A Comprehensive Study of Transfer Learning.**
  *Proceedings of the IEEE*, 109(1), 43–76.

### Carbon Flux Forecasting

- Jung, M., et al. (2020). **Scaling carbon fluxes from eddy covariance sites to globe:
  synthesis and evaluation of the FLUXCOM approach.**
  *Biogeosciences*, 17(5), 1343–1365.

- Tramontana, G., et al. (2016). **Predicting carbon dioxide and energy fluxes across global
  FLUXNET sites with regression algorithms.**
  *Biogeosciences*, 13(14), 4291–4313.

### Active Learning

- Settles, B. (2009). **Active Learning Literature Survey.**
  *Computer Sciences Technical Report 1648*, University of Wisconsin–Madison.

---

*Last updated: March 2026*
