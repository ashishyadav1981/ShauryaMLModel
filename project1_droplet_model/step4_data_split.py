#!/usr/bin/env python3
"""
STEP 4: TRAIN/VALIDATION/TEST SPLIT
Split normalized data for ML training pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 4: TRAIN/VALIDATION/TEST SPLIT")
print("="*80)

# ============================================================================
# SECTION 1: Load Normalized Data
# ============================================================================
print("\n[1/4] Loading normalized data...")

try:
    # Try to load from pickle (faster)
    X_normalized = np.load('X_normalized.pkl', allow_pickle=True)
    y_normalized = np.load('y_normalized.pkl', allow_pickle=True)
    print("  ✓ Loaded from pickle format")
except:
    # Fallback to CSV
    try:
        X_normalized = pd.read_csv('X_normalized.csv').values
        y_normalized = pd.read_csv('y_normalized.csv').values
        print("  ✓ Loaded from CSV format")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print(f"  Make sure you've run Step 3 (normalization) first!")
        exit(1)

print(f"  ✓ X_normalized shape: {X_normalized.shape}")
print(f"  ✓ y_normalized shape: {y_normalized.shape}")

# ============================================================================
# SECTION 2: Split into Train/Val/Test (70/15/15)
# ============================================================================
print("\n[2/4] Splitting data into train/val/test (70/15/15)...")

# First split: 70% train, 30% temporary (for val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_normalized, y_normalized,
    test_size=0.30,
    random_state=42
)

# Second split: Split temporary 50/50 into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

# Calculate percentages
train_pct = len(X_train) / len(X_normalized) * 100
val_pct = len(X_val) / len(X_normalized) * 100
test_pct = len(X_test) / len(X_normalized) * 100

print(f"  ✓ Train: {len(X_train)} samples ({train_pct:.1f}%)")
print(f"  ✓ Val:   {len(X_val)} samples ({val_pct:.1f}%)")
print(f"  ✓ Test:  {len(X_test)} samples ({test_pct:.1f}%)")
print(f"  ✓ Total: {len(X_train) + len(X_val) + len(X_test)} samples")

# ============================================================================
# SECTION 3: Verify No Data Leakage
# ============================================================================
print("\n[3/4] Verifying data integrity...")

# Check shapes
assert X_train.shape[0] == y_train.shape[0], "Train shapes mismatch!"
assert X_val.shape[0] == y_val.shape[0], "Val shapes mismatch!"
assert X_test.shape[0] == y_test.shape[0], "Test shapes mismatch!"
print("  ✓ All shapes aligned")

# Check no sample is in multiple sets
total_samples = len(X_train) + len(X_val) + len(X_test)
assert total_samples == len(X_normalized), "Sample count mismatch!"
print(f"  ✓ All {len(X_normalized)} samples assigned exactly once")

# Check no NaN/Inf
assert not np.isnan(X_train).any(), "NaN in X_train!"
assert not np.isnan(y_train).any(), "NaN in y_train!"
assert not np.isinf(X_train).any(), "Inf in X_train!"
assert not np.isinf(y_train).any(), "Inf in y_train!"
print("  ✓ No NaN or Inf values in training data")

# Check data ranges (allow small floating-point errors)
tolerance = 1e-6  # Allow for numerical precision errors
assert X_train.min() >= -tolerance, f"X_train min < 0! (min: {X_train.min()})"
assert X_train.max() <= 1 + tolerance, f"X_train max > 1! (max: {X_train.max()})"
assert y_train.min() >= -tolerance, f"y_train min < 0! (min: {y_train.min()})"
assert y_train.max() <= 1 + tolerance, f"y_train max > 1! (max: {y_train.max()})"

# Clamp values to [0, 1] to handle floating-point precision
X_train = np.clip(X_train, 0, 1)
X_val = np.clip(X_val, 0, 1)
X_test = np.clip(X_test, 0, 1)
y_train = np.clip(y_train, 0, 1)
y_val = np.clip(y_val, 0, 1)
y_test = np.clip(y_test, 0, 1)

print("  ✓ All values clamped to normalized range [0, 1]")

# ============================================================================
# SECTION 4: Save Train/Val/Test Data
# ============================================================================
print("\n[4/4] Saving train/val/test data...")

# Save as numpy arrays
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("  ✓ Saved in numpy format (.npy)")

# Save as pickle
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_val, open('X_val.pkl', 'wb'))
pickle.dump(y_val, open('y_val.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))

print("  ✓ Saved in pickle format (.pkl)")

# Save as CSV
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_val).to_csv('X_val.csv', index=False)
pd.DataFrame(y_val).to_csv('y_val.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("  ✓ Saved in CSV format (.csv)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA SPLIT COMPLETE!")
print("="*80)

print(f"""
✓ Data successfully split and saved!

SPLIT SUMMARY:
  Train: {len(X_train):3d} samples ({train_pct:5.1f}%)
  Val:   {len(X_val):3d} samples ({val_pct:5.1f}%)
  Test:  {len(X_test):3d} samples ({test_pct:5.1f}%)
  ─────────────────────────────────
  Total: {len(X_normalized):3d} samples (100%)

DATA SHAPES:
  X_train: {X_train.shape}  |  y_train: {y_train.shape}
  X_val:   {X_val.shape}    |  y_val:   {y_val.shape}
  X_test:  {X_test.shape}   |  y_test:  {y_test.shape}

FILES SAVED (3 formats):
  ✓ Numpy (.npy)  - fastest for PyTorch
  ✓ Pickle (.pkl) - good compression
  ✓ CSV (.csv)    - human readable

NEXT: Step 5 - Define Neural Network

Run: python3 step5_neural_network.py
""")

print("="*80 + "\n")
