# baseline_models.py
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


class BaselineModels:
    """Traditional ML models for comparison."""

    def train_random_forest(self, X_train, y_train):
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        return rf

    def train_xgboost(self, X_train, y_train):
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        return xgb


def evaluate(model, X, y):
    """Compute RMSE, MAE, and R2 for a fitted model."""
    X_flat = X.reshape(X.shape[0], -1)
    y_pred = model.predict(X_flat)
    # Handle multi-output: average across forecast horizon
    if y_pred.ndim == 2:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
    else:
        rmse = np.sqrt(mean_squared_error(y.mean(axis=1), y_pred))
        mae = mean_absolute_error(y.mean(axis=1), y_pred)
        r2 = r2_score(y.mean(axis=1), y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'

    print("Loading processed data...")
    X_train = np.load(data_dir / 'train_X.npy')
    y_train = np.load(data_dir / 'train_y.npy')

    test_sites = ['UK-AMo', 'SE-Htm']
    test_data = {}
    for site in test_sites:
        test_data[site] = {
            'X': np.load(data_dir / f'test_{site}_X.npy'),
            'y': np.load(data_dir / f'test_{site}_y.npy'),
        }

    print(f"Training data: {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} timesteps, {X_train.shape[2]} features")

    baselines = BaselineModels()

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = baselines.train_random_forest(X_train, y_train)

    # Train XGBoost
    print("Training XGBoost...")
    xgb = baselines.train_xgboost(X_train, y_train)

    # Evaluate on test sites
    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS")
    print("=" * 60)

    for site in test_sites:
        X_test = test_data[site]['X']
        y_test = test_data[site]['y']

        print(f"\n--- {site} ({len(X_test)} samples) ---")

        rf_metrics = evaluate(rf, X_test, y_test)
        print(f"  Random Forest:  RMSE={rf_metrics['RMSE']:.4f}  "
              f"MAE={rf_metrics['MAE']:.4f}  R2={rf_metrics['R2']:.4f}")

        xgb_metrics = evaluate(xgb, X_test, y_test)
        print(f"  XGBoost:        RMSE={xgb_metrics['RMSE']:.4f}  "
              f"MAE={xgb_metrics['MAE']:.4f}  R2={xgb_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
