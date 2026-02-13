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

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch

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
    
    def preprocess_site(self, df):
        """
        Preprocess a single site's data.
        
        Steps:
        1. Handle missing values
        2. Create temporal features
        3. Normalize features
        4. Extract target variable
        """
        # Ensure datetime index
        ts_col = 'TIMESTAMP' if 'TIMESTAMP' in df.columns else 'timestamp'
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True)
            df = df.set_index(ts_col)
        
        # Extract target
        target = df['NEE_VUT_REF'].ffill().bfill().values
        
        # Combine all features
        features = df[self.meteorological_vars + self.modis_bands + self.temporal_features].copy()
        
        # Handle missing values
        # Option 1: Forward fill (common for EC data)
        features = features.ffill().bfill()
        
        # Option 2: Could use more sophisticated gap-filling
        # features = self.gapfill_advanced(features)
        
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
        
        Training strategy:
        - Train on 5 sites: FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt
        - Validate on portion of training sites
        - Test on 2 held-out sites: UK-AMo, SE-Htm
        """
        train_sites = ['FI-Lom', 'GL-ZaF', 'IE-Cra', 'DE-Akm', 'FR-LGt']
        test_sites = ['UK-AMo', 'SE-Htm']
        
        # First pass: load and gap-fill all training site data to fit a global scaler
        print("Loading training sites...")
        train_raw = {}
        all_features = []
        for site in train_sites:
            df = self.load_site_data(site)
            ts_col = 'TIMESTAMP' if 'TIMESTAMP' in df.columns else 'timestamp'
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True)
                df = df.set_index(ts_col)
            target = df['NEE_VUT_REF'].ffill().bfill().values
            features = df[self.meteorological_vars + self.modis_bands + self.temporal_features].copy()
            features = features.ffill().bfill()
            train_raw[site] = (features, target, df.index)
            all_features.append(features)

        # Fit global scaler on all training data
        global_scaler = StandardScaler()
        global_scaler.fit(pd.concat(all_features))
        self.scalers['global'] = global_scaler

        # Second pass: scale and create sequences
        train_features_list = []
        train_targets_list = []
        for site in train_sites:
            features, target, timestamps = train_raw[site]
            features_scaled = global_scaler.transform(features)
            X, y = self.create_tempo_sequences(features_scaled, target)
            train_features_list.append(X)
            train_targets_list.append(y)
            print(f"  {site}: {len(X)} sequences")

        # Concatenate all training data
        X_train = np.concatenate(train_features_list, axis=0)
        y_train = np.concatenate(train_targets_list, axis=0)

        # Prepare test data (held-out sites)
        test_data = {}
        print("\nLoading test sites...")
        for site in test_sites:
            df = self.load_site_data(site)
            features, target, timestamps = self.preprocess_site(df)
            
            X_test, y_test = self.create_tempo_sequences(features, target)
            test_data[site] = {
                'X': X_test,
                'y': y_test,
                'timestamps': timestamps
            }
            print(f"  {site}: {len(X_test)} sequences")
        
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': test_data
        }
    
    def prepare_for_tempo(self, X, y):
        """
        Convert numpy arrays to format expected by TEMPO.
        
        TEMPO expects either:
        1. Single time series: shape (T,) for univariate or (T, F) for multivariate
        2. Batch of series: list of arrays
        """
        # For fine-tuning, we'll use the multivariate approach
        # Each sequence in X is (lookback, features)
        
        # Option 1: Use only target variable (NEE) - univariate
        # This is simpler and TEMPO was primarily trained on univariate series
        
        # Option 2: Use all features - multivariate
        # TEMPO can handle this but may need more careful configuration
        
        return X, y


# Example usage for your thesis
def main():
    """
    Main workflow for preparing data for TEMPO experiments.
    """
    processor = CarbonFluxDataProcessor()
    
    # Prepare cross-site splits
    data_splits = processor.prepare_cross_site_splits()
    
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Training samples: {len(data_splits['train']['X'])}")
    print(f"Feature dimension: {data_splits['train']['X'].shape[-1]}")
    print(f"\nTest sites:")
    for site, data in data_splits['test'].items():
        print(f"  {site}: {len(data['X'])} samples")
    
    # Save processed data
    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / 'train_X.npy', data_splits['train']['X'])
    np.save(output_dir / 'train_y.npy', data_splits['train']['y'])

    for site, data in data_splits['test'].items():
        np.save(output_dir / f'test_{site}_X.npy', data['X'])
        np.save(output_dir / f'test_{site}_y.npy', data['y'])
    
    print("\n✓ Data saved successfully!")
    print("\nNext steps:")
    print("1. Run zero-shot TEMPO inference (no training)")
    print("2. Fine-tune TEMPO on your training data")
    print("3. Compare with baseline models (RF, XGBoost)")
    print("4. Evaluate cross-site generalization")

if __name__ == "__main__":
    main()