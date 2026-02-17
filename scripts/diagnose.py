# Quick diagnostic script - run in Python terminal
import numpy as np

# Load your data
train_X = np.load('data/processed/train_X.npy')
train_y = np.load('data/processed/train_y.npy')
test_X = np.load('data/processed/test_UK-AMo_X.npy')
test_y = np.load('data/processed/test_UK-AMo_y.npy')

# Diagnose
print("=== DATA DIAGNOSTICS ===")
print(f"train_X shape: {train_X.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_X shape:  {test_X.shape}")
print(f"test_y shape:  {test_y.shape}")

print(f"\nNaN in train_X: {np.isnan(train_X).sum()}")
print(f"NaN in train_y: {np.isnan(train_y).sum()}")
print(f"NaN in test_X:  {np.isnan(test_X).sum()}")
print(f"NaN in test_y:  {np.isnan(test_y).sum()}")

print(f"\ntrain_X range: [{np.nanmin(train_X):.3f}, {np.nanmax(train_X):.3f}]")
print(f"train_y range: [{np.nanmin(train_y):.3f}, {np.nanmax(train_y):.3f}]")