"""
Data Preparation Pipeline for TEMPO Carbon Flux Forecasting
=============================================================

This script demonstrates how to prepare your multimodal carbon flux dataset
for use with the TEMPO foundation model.

Your data structure:
- 7 EC tower sites (FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt, UK-AMo, SE-Htm)
- Features: Meteorological variables + MODIS bands + temporal features
- Target: NEE_VUT_REF
- Train sites: 5 sites (FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt)
- Test sites: 2 sites (UK-AMo, SE-Htm) for cross-site generalization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch


def fill_feature_gaps(df, feature_cols, site_name=None,
                      cross_site_medians=None):
    """
    Enhanced gap-filling with cross-site fallback for
    completely missing features.

    Strategy (applied in order):
    1. Linear interpolation  (short gaps < 48 hours)
    2. Forward/backward fill (medium gaps < 14 days)
    3. Seasonal median       (long gaps - uses day-of-year)
    4. Cross-site median     (if entire column missing)
    5. Global zero/constant  (absolute last resort)
    """
    df_filled = df.copy()
    features_needing_crosssite = []
    n_cols_filled = 0
    zero_fallback_cols = []

    for col in feature_cols:
        nan_count = df_filled[col].isna().sum()
        if nan_count == 0:
            continue

        n_cols_filled += 1

        # STEP 1: Linear interpolation for short gaps (< 48 timesteps)
        df_filled[col] = df_filled[col].interpolate(
            method='linear',
            limit=48,
            limit_direction='both'
        )

        # STEP 2: Forward/backward fill for medium gaps
        df_filled[col] = df_filled[col].ffill(limit=336)
        df_filled[col] = df_filled[col].bfill(limit=336)

        # STEP 3: Seasonal median (day-of-year based)
        if df_filled[col].isna().sum() > 0:
            if hasattr(df_filled.index, 'dayofyear'):
                doy = df_filled.index.dayofyear
            else:
                doy = pd.to_datetime(df_filled.index).dayofyear

            temp_df = pd.DataFrame({
                'value': df_filled[col],
                'doy': doy
            })
            valid_count = temp_df['value'].notna().sum()
            if valid_count > 48:  # Need at least 2 days of data
                seasonal_med = temp_df.groupby('doy')['value'].transform('median')
                df_filled[col] = df_filled[col].fillna(seasonal_med)

        # STEP 4: Cross-site median (for 100% missing features)
        if df_filled[col].isna().sum() > 0:
            if cross_site_medians is not None and col in cross_site_medians:
                cross_median = cross_site_medians[col]
                df_filled[col] = df_filled[col].fillna(cross_median)
                features_needing_crosssite.append(col)
            else:
                features_needing_crosssite.append(col)

        # STEP 5: Absolute fallback - zero fill
        if df_filled[col].isna().sum() > 0:
            df_filled[col] = df_filled[col].fillna(0.0)
            zero_fallback_cols.append(col)

    remaining_nan = df_filled.isna().sum().sum()
    assert remaining_nan == 0, f"NaN remaining after gap-fill: {remaining_nan}"

    prefix = f"  [{site_name}] " if site_name else "  "
    status = f"{n_cols_filled} features gap-filled, 0 NaN remaining"
    if zero_fallback_cols:
        status += f"  ⚠ Zero-fallback applied to: {zero_fallback_cols}"
    print(f"{prefix}✓ {status}")

    return df_filled, features_needing_crosssite


class CarbonFluxDataProcessor:
    """
    Prepares EC tower data for TEMPO model training and inference.
    
    TEMPO expects:
    - Time series data with consistent temporal resolution
    - Normalized/scaled features
    - Proper train/val/test splits
    - Optional: decomposed components (trend, seasonal, residual)
    """
    
    def __init__(self, data_dir=str(Path(__file__).resolve().parent.parent / 'data' / 'raw')):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        
        # Define your feature groups
        self.meteorological_vars = [
            'SW_IN_F', 'LW_IN_F', 'VPD_F', 'TA_F',
            'PA_F', 'P_F', 'WS_F', 'G_F_MDS', 'LE_F_MDS', 'H_F_MDS'
        ]

        self.modis_bands = [
            f'MODIS_band_{i}' for i in range(1, 8)
        ]

        self.temporal_features = [
            'DOY', 'TOD'
        ]
        
    def load_site_data(self, site_name):
        """Load data for a single EC tower site."""
        # Map site codes to your uploaded files
        site_files = {
            'FI-Lom': '1.FI-Lom.csv',
            'GL-ZaF': '2.GL-ZaF.csv',
            'IE-Cra': '3.IE-Cra.xlsx',  # Note: Excel file
            'DE-Akm': '4.DE-Akm.csv',
            'FR-LGt': '5.FR-LGt.csv',
            'UK-AMo': '6.UK-AMo.csv',
            'SE-Htm': '7.SE-Htm.csv'
        }
        
        filepath = self.data_dir / site_files[site_name]
        
        if filepath.suffix == '.xlsx':
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
            
        return df
    
    def _set_datetime_index(self, df):
        """Ensure datetime index on a dataframe."""
        ts_col = 'TIMESTAMP' if 'TIMESTAMP' in df.columns else 'timestamp'
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True)
            df = df.set_index(ts_col)
        return df

    def preprocess_site(self, df, site_name=None, cross_site_medians=None):
        """
        Preprocess a single site's data.

        Steps:
        1. Handle missing values (with cross-site fallback)
        2. Normalize features
        3. Extract target variable
        """
        df = self._set_datetime_index(df)

        # Extract target
        target = df['NEE_VUT_REF'].ffill().bfill().values

        # Combine all features
        all_feature_cols = self.meteorological_vars + self.modis_bands + self.temporal_features
        features = df[all_feature_cols].copy()

        # Targeted gap-filling with cross-site fallback
        features, _ = fill_feature_gaps(
            features, all_feature_cols,
            site_name=site_name,
            cross_site_medians=cross_site_medians
        )

        # Normalize features using global scaler
        scaler = self.scalers['global']
        features_scaled = scaler.transform(features)

        return features_scaled, target, df.index
    
    def create_tempo_sequences(self, features, target, lookback=336, horizon=96):
        """
        Create input sequences for TEMPO.
        
        Args:
            features: Shape (T, F) - T timesteps, F features
            target: Shape (T,) - Target variable
            lookback: Historical window size (default 336 = 2 weeks hourly)
            horizon: Prediction horizon (default 96 = 4 days hourly)
        
        Returns:
            X: Input sequences
            y: Target sequences
        """
        X, y = [], []
        
        for i in range(len(features) - lookback - horizon + 1):
            # TEMPO can handle multivariate input
            X.append(features[i:i+lookback])
            y.append(target[i+lookback:i+lookback+horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_cross_site_splits(self):
        """
        Prepare data splits for cross-site generalization.

        Uses a two-pass approach:
        - Pass 1: Load all sites, collect cross-site feature statistics
        - Pass 2: Gap-fill using cross-site medians for 100% missing features

        Training: FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt
        Test:     UK-AMo, SE-Htm
        """
        train_sites = ['FI-Lom', 'GL-ZaF', 'IE-Cra', 'DE-Akm', 'FR-LGt']
        test_sites = ['UK-AMo', 'SE-Htm']
        all_feature_cols = self.meteorological_vars + self.modis_bands + self.temporal_features

        # ─── PASS 1: Load all sites and compute cross-site statistics ───
        print("[1/4] Loading all sites and computing cross-site statistics...")
        site_dataframes = {}
        cross_site_values = {}  # col -> list of valid values across sites

        for site in train_sites + test_sites:
            df = self.load_site_data(site)
            df = self._set_datetime_index(df)
            site_dataframes[site] = df

            # Collect valid values for each feature
            for col in all_feature_cols:
                if col not in df.columns:
                    continue
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    if col not in cross_site_values:
                        cross_site_values[col] = []
                    cross_site_values[col].extend(
                        valid_vals.sample(
                            min(1000, len(valid_vals)),
                            random_state=42
                        ).tolist()
                    )

        # Compute cross-site medians
        cross_site_medians = {
            col: float(np.median(vals))
            for col, vals in cross_site_values.items()
        }
        self.cross_site_medians = cross_site_medians
        print(f"  ✓ Cross-site medians computed for {len(cross_site_medians)} features")

        # ─── PASS 2: Gap-fill with cross-site fallback ──────────────────
        print("\n[2/4] Gap-filling training sites...")
        train_raw = {}
        all_features = []

        for site in train_sites:
            df = site_dataframes[site]
            target = df['NEE_VUT_REF'].ffill().bfill().values
            features = df[all_feature_cols].copy()

            features, missing_features = fill_feature_gaps(
                features, all_feature_cols,
                site_name=site,
                cross_site_medians=cross_site_medians
            )

            if missing_features:
                print(f"  [{site}] Cross-site fallback used for: {missing_features}")

            train_raw[site] = (features, target, df.index)
            all_features.append(features)

        # Fit global scaler on all gap-filled training data
        print("\n[3/4] Fitting global scaler and creating sequences...")
        global_scaler = StandardScaler()
        global_scaler.fit(pd.concat(all_features))
        self.scalers['global'] = global_scaler

        # Scale and create sequences
        train_features_list = []
        train_targets_list = []
        for site in train_sites:
            features, target, timestamps = train_raw[site]
            features_scaled = global_scaler.transform(features)
            X, y = self.create_tempo_sequences(features_scaled, target)
            train_features_list.append(X)
            train_targets_list.append(y)
            print(f"  ✓ {site:<10}  {len(X):>6,} sequences")

        X_train = np.concatenate(train_features_list, axis=0)
        y_train = np.concatenate(train_targets_list, axis=0)

        # Prepare test data (held-out sites)
        print("\n[4/4] Processing held-out test sites...")
        test_data = {}
        for site in test_sites:
            df = self.load_site_data(site)
            features, target, timestamps = self.preprocess_site(
                df, site_name=site, cross_site_medians=cross_site_medians
            )

            X_test, y_test = self.create_tempo_sequences(features, target)
            test_data[site] = {
                'X': X_test,
                'y': y_test,
                'timestamps': timestamps
            }
            print(f"  ✓ {site:<10}  {len(X_test):>6,} sequences")

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': test_data,
            'cross_site_medians': cross_site_medians
        }


# Example usage for your thesis
def main():
    """
    Main workflow for preparing data for TEMPO experiments.
    """
    import time
    from datetime import datetime

    print("=" * 72)
    print("DATA PREPARATION PIPELINE — CARBON FLUX FORECASTING")
    print(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy:   Cross-site generalisation (5 train / 2 test sites)")
    print("=" * 72 + "\n")

    t0 = time.time()
    processor = CarbonFluxDataProcessor()

    # Prepare cross-site splits
    data_splits = processor.prepare_cross_site_splits()

    # Save processed data
    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)

    print("\n" + "─" * 72)
    print("Saving processed arrays...")
    np.save(output_dir / 'train_X.npy', data_splits['train']['X'])
    np.save(output_dir / 'train_y.npy', data_splits['train']['y'])
    print(f"  ✓ data/processed/train_X.npy  "
          f"{data_splits['train']['X'].shape}")
    print(f"  ✓ data/processed/train_y.npy  "
          f"{data_splits['train']['y'].shape}")

    for site, data in data_splits['test'].items():
        np.save(output_dir / f'test_{site}_X.npy', data['X'])
        np.save(output_dir / f'test_{site}_y.npy', data['y'])
        print(f"  ✓ data/processed/test_{site}_X/y.npy  "
              f"{data['X'].shape}")

    medians_path = output_dir / 'cross_site_medians.json'
    with open(medians_path, 'w') as f:
        json.dump(data_splits['cross_site_medians'], f, indent=2)
    print(f"  ✓ data/processed/cross_site_medians.json  "
          f"({len(data_splits['cross_site_medians'])} features)")

    elapsed = time.time() - t0
    X_train = data_splits['train']['X']
    print(f"\n{'─' * 72}")
    print(f"  Training sequences:   {len(X_train):>8,}  "
          f"(lookback={X_train.shape[1]}, features={X_train.shape[2]})")
    for site, d in data_splits['test'].items():
        print(f"  Test {site:<10}:   {len(d['X']):>8,} sequences")
    print(f"  ⏱  Completed in {elapsed:.1f}s")
    print("=" * 72)

if __name__ == "__main__":
    main()