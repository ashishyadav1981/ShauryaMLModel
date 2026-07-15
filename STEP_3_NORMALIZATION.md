# STEP 3: NORMALIZATION
## Prepare Your Data for Neural Network Training

---

## Overview

**Normalization** scales all your features and targets to the same range (typically 0-1). This is **essential** for neural networks because:

- Prevents features with large values from dominating training
- Helps gradient descent converge faster
- Improves numerical stability
- Makes training more efficient

---

## Why Normalize?

### Problem Without Normalization
```
Feature ranges in your data:
  - initial_diameter_um:    10 - 100        (small numbers)
  - T_K:                    800 - 800       (large numbers)
  - p_Pa:                   5,000,000       (very large!)
  - vel_mag_m_s:            0.0001 - 0.0015 (tiny numbers)

Neural network treats them equally.
Result: Pressure gradient dominates learning!
```

### Solution: Normalize to [0, 1]
```
All features scaled to 0-1 range.
Neural network treats each feature fairly.
Result: Better, faster training!
```

---

## Step 3A: Load Your Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load your extracted data
X = pd.read_csv('droplet_features.csv')
y = pd.read_csv('droplet_targets.csv')

print(f"Features shape: {X.shape}")  # (792, 10)
print(f"Targets shape: {y.shape}")   # (792, 3)

print("\nFeature ranges BEFORE normalization:")
print(X.describe())

print("\nTarget ranges BEFORE normalization:")
print(y.describe())
```

**Expected output:**
```
Features shape: (792, 10)
Targets shape: (792, 3)

Feature ranges BEFORE normalization:
       initial_diameter_um   time_ms    T_K       p_Pa  ...
mean               45.0000    5.0000  800.0  5000000.0  
min                10.0000    0.0000  800.0  5000000.0  
max               100.0000   10.0000  800.0  5000000.0  
```

---

## Step 3B: Normalize Features

```python
# Create MinMaxScaler for features
scaler_X = MinMaxScaler(feature_range=(0, 1))

# Fit and transform features
X_normalized = scaler_X.fit_transform(X)

# Convert back to DataFrame (optional, for readability)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

print("Feature ranges AFTER normalization:")
print(X_normalized_df.describe())

# Should show: min=0, max=1 for all features
```

**Expected output:**
```
Feature ranges AFTER normalization:
       initial_diameter_um  time_ms  T_K  p_Pa  ...
count             792.0      792.0  792.0  792.0
mean               0.4286    0.5000  NaN   NaN
min                0.0000    0.0000  0.0   NaN
max                1.0000    1.0000  1.0   NaN

Note: T_K and p_Pa show NaN because they're constant (all 800 K, all 5 MPa)
This is fine! Constant features don't contribute to learning.
```

---

## Step 3C: Normalize Targets

```python
# Create MinMaxScaler for targets
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Fit and transform targets
y_normalized = scaler_y.fit_transform(y)

# Convert back to DataFrame
y_normalized_df = pd.DataFrame(y_normalized, columns=y.columns)

print("Target ranges AFTER normalization:")
print(y_normalized_df.describe())

# Should show: min=0, max=1 for all targets
```

**Expected output:**
```
Target ranges AFTER normalization:
       diameter_um  evaporation_time_ms  emissions_proxy
count       792.0              792.0              792.0
mean          0.429              0.324              0.323
min           0.000              0.000              0.000
max           1.000              1.000              1.000
```

---

## Step 3D: Save Scalers for Later

**CRITICAL:** You need to save the scalers to un-normalize predictions later!

```python
# Save scalers to files
pickle.dump(scaler_X, open('scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open('scaler_y.pkl', 'wb'))

print("✓ Scalers saved:")
print("  - scaler_X.pkl")
print("  - scaler_y.pkl")
```

Later, you'll load them like this:
```python
# Load scalers
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

# Un-normalize predictions
y_pred_unnormalized = scaler_y.inverse_transform(y_pred_normalized)
```

---

## Step 3E: Save Normalized Data

```python
# Save normalized data as CSV
np.savetxt('X_normalized.csv', X_normalized, delimiter=',', header=','.join(X.columns), comments='')
np.savetxt('y_normalized.csv', y_normalized, delimiter=',', header=','.join(y.columns), comments='')

# Or as pickle (faster for numpy arrays)
import pickle
pickle.dump(X_normalized, open('X_normalized.pkl', 'wb'))
pickle.dump(y_normalized, open('y_normalized.pkl', 'wb'))

print("✓ Normalized data saved:")
print("  - X_normalized.csv / X_normalized.pkl")
print("  - y_normalized.csv / y_normalized.pkl")
```

---

## Complete Step 3 Script

Here's everything in one script:

```python
#!/usr/bin/env python3
"""
STEP 3: NORMALIZATION
Normalize your droplet training data for ML
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("\n" + "="*80)
print("STEP 3: DATA NORMALIZATION")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/4] Loading extracted data...")

X = pd.read_csv('droplet_features.csv')
y = pd.read_csv('droplet_targets.csv')

print(f"  ✓ Features shape: {X.shape}")
print(f"  ✓ Targets shape: {y.shape}")

print("\n  Feature Statistics BEFORE normalization:")
print(X.describe().round(4))

print("\n  Target Statistics BEFORE normalization:")
print(y.describe().round(4))

# ============================================================================
# Normalize Features
# ============================================================================
print("\n[2/4] Normalizing features...")

scaler_X = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler_X.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

print(f"  ✓ Features normalized to [0, 1]")
print("\n  Feature Statistics AFTER normalization:")
print(X_normalized_df.describe().round(4))

# ============================================================================
# Normalize Targets
# ============================================================================
print("\n[3/4] Normalizing targets...")

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler_y.fit_transform(y)
y_normalized_df = pd.DataFrame(y_normalized, columns=y.columns)

print(f"  ✓ Targets normalized to [0, 1]")
print("\n  Target Statistics AFTER normalization:")
print(y_normalized_df.describe().round(4))

# ============================================================================
# Save Scalers and Normalized Data
# ============================================================================
print("\n[4/4] Saving scalers and normalized data...")

# Save scalers (CRITICAL for un-normalizing predictions later)
pickle.dump(scaler_X, open('scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open('scaler_y.pkl', 'wb'))
print("  ✓ Scalers saved:")
print("    - scaler_X.pkl (for input features)")
print("    - scaler_y.pkl (for output targets)")

# Save normalized data as CSV
X_normalized_df.to_csv('X_normalized.csv', index=False)
y_normalized_df.to_csv('y_normalized.csv', index=False)
print("  ✓ Normalized data saved:")
print("    - X_normalized.csv")
print("    - y_normalized.csv")

# Also save as numpy arrays (optional, for faster loading)
pickle.dump(X_normalized, open('X_normalized.pkl', 'wb'))
pickle.dump(y_normalized, open('y_normalized.pkl', 'wb'))
print("  ✓ Also saved as pickle format:")
print("    - X_normalized.pkl")
print("    - y_normalized.pkl")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("NORMALIZATION COMPLETE!")
print("="*80)

print(f"""
✓ Data successfully normalized!

ORIGINAL DATA RANGES:
  Features:
    - initial_diameter_um:    10.0 - 100.0
    - time_ms:                0.0 - 10.0
    - T_K:                    800.0 - 800.0 (constant)
    - p_Pa:                   5,000,000.0 (constant)
    - etc.
    
  Targets:
    - diameter_um:            8.68 - 100.0
    - evaporation_time_ms:    40.5 - 4051.6
    - emissions_proxy:        15,244 - 2,025,821

NORMALIZED DATA RANGES (now all [0, 1]):
  Features:
    - All features normalized to 0-1 range
    - Constant features (T, p) mapped to single value
    - Ready for neural network!
    
  Targets:
    - All targets normalized to 0-1 range
    - Ready for model training!

FILES CREATED:
  1. scaler_X.pkl              ← Load this to un-normalize test predictions
  2. scaler_y.pkl              ← Load this to un-normalize model outputs
  3. X_normalized.csv          ← Normalized input features
  4. y_normalized.csv          ← Normalized output targets
  5. X_normalized.pkl          ← Same as CSV but in pickle format (faster)
  6. y_normalized.pkl          ← Same as CSV but in pickle format (faster)

NEXT STEPS:
  1. Proceed to Step 4: Train/Val/Test Split
  2. Define neural network architecture (Step 5)
  3. Set up training loop (Step 6)
  4. Train your model
  5. Validate on test set
  6. Compare against Sandia Flame D data
""")

print("="*80 + "\n")
```

---

## Step 3F: Verify Normalization Worked

```python
# Quick verification
print("Verification:")
print(f"  X_normalized min: {X_normalized.min():.4f} (should be 0)")
print(f"  X_normalized max: {X_normalized.max():.4f} (should be 1)")
print(f"  y_normalized min: {y_normalized.min():.4f} (should be 0)")
print(f"  y_normalized max: {y_normalized.max():.4f} (should be 1)")

# Check for NaN or Inf
print(f"\n  NaN in X_normalized: {np.isnan(X_normalized).sum()}")
print(f"  Inf in X_normalized: {np.isinf(X_normalized).sum()}")
print(f"  NaN in y_normalized: {np.isnan(y_normalized).sum()}")
print(f"  Inf in y_normalized: {np.isinf(y_normalized).sum()}")

# All should be 0
```

---

## Important Notes

### 1. Constant Features (T_K, p_Pa)
Your data has constant temperature and pressure. After normalization:
- **Option A:** Keep them as 0 (or 0.5) — won't affect learning
- **Option B:** Remove them entirely — reduces input dimensions

Recommendation: **Keep them** for now. They'll just have zero variance.

### 2. Un-Normalizing Predictions

After your model makes predictions, un-normalize using the saved scaler:

```python
# Load scaler
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

# Your model predicts normalized values
y_pred_normalized = model.predict(X_test_normalized)  # Range [0, 1]

# Un-normalize to real units
y_pred_unnormalized = scaler_y.inverse_transform(y_pred_normalized)

# Now in real units:
# - diameter_um: actual droplet size
# - evaporation_time_ms: actual time
# - emissions_proxy: actual heat release
```

### 3. Never Normalize Test Data Separately

**WRONG:**
```python
# DON'T do this - creates data leakage
scaler_test = MinMaxScaler()
X_test_normalized = scaler_test.fit_transform(X_test)
```

**RIGHT:**
```python
# Use the training scaler on test data
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
X_test_normalized = scaler_X.transform(X_test)  # transform, not fit_transform!
```

---

## Output Files

After running Step 3, you'll have:

```
✓ scaler_X.pkl                 (input feature scaler - SAVE THIS!)
✓ scaler_y.pkl                 (output target scaler - SAVE THIS!)
✓ X_normalized.csv             (792 × 10 normalized features)
✓ y_normalized.csv             (792 × 3 normalized targets)
✓ X_normalized.pkl             (same as CSV, faster loading)
✓ y_normalized.pkl             (same as CSV, faster loading)
```

---

## Next: Step 4 - Train/Val/Test Split

Once you have normalized data, proceed to **Step 4** to:
1. Split data into train (70%), validation (15%), test (15%)
2. Create PyTorch DataLoaders for batch training
3. Set up training infrastructure

---

## Summary

| Step | What | Files In | Files Out |
|------|------|----------|-----------|
| Step 2 | Extract features | Eulerian CFD | droplet_features.csv, droplet_targets.csv |
| **Step 3** | **Normalize** | **droplet_features/targets.csv** | **X_normalized, y_normalized, scalers** |
| Step 4 | Split data | X_normalized, y_normalized | train/val/test sets |
| Step 5 | Build model | - | neural network |
| Step 6 | Train | train/val sets | trained weights |
| Step 7 | Evaluate | test set | metrics (MAE, RMSE, R²) |
| Step 8 | Validate | Sandia data | comparison results |

---

## Ready?

Run the normalization script and you'll be ready for Step 4! 🚀
