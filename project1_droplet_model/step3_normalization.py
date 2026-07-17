#!/usr/bin/env python3
"""
STEP 3: DATA NORMALIZATION
Normalize extracted droplet training data for ML training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 3: DATA NORMALIZATION")
print("="*80)

# ============================================================================
# SECTION 1: Load Data
# ============================================================================
print("\n[1/4] Loading extracted data...")

try:
    X = pd.read_csv('droplet_features.csv')
    y = pd.read_csv('droplet_targets.csv')
    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Targets shape: {y.shape}")
except FileNotFoundError as e:
    print(f"  ✗ Error: {e}")
    print(f"  Make sure you've run Step 2 (feature extraction) first!")
    exit(1)

print("\n  Feature Statistics BEFORE normalization:")
print(X.describe().round(4))

print("\n  Target Statistics BEFORE normalization:")
print(y.describe().round(4))

# ============================================================================
# SECTION 2: Normalize Features
# ============================================================================
print("\n[2/4] Normalizing features...")

scaler_X = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler_X.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

print(f"  ✓ Features normalized to [0, 1]")
print(f"  ✓ Feature scaler fitted on {len(X)} training samples")

print("\n  Feature Statistics AFTER normalization:")
print(X_normalized_df.describe().round(4))

# ============================================================================
# SECTION 3: Normalize Targets
# ============================================================================
print("\n[3/4] Normalizing targets...")

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler_y.fit_transform(y)
y_normalized_df = pd.DataFrame(y_normalized, columns=y.columns)

print(f"  ✓ Targets normalized to [0, 1]")
print(f"  ✓ Target scaler fitted on {len(y)} samples")

print("\n  Target Statistics AFTER normalization:")
print(y_normalized_df.describe().round(4))

# ============================================================================
# SECTION 4: Save Scalers and Data
# ============================================================================
print("\n[4/4] Saving scalers and normalized data...")

# Save scalers (CRITICAL for un-normalizing predictions)
pickle.dump(scaler_X, open('scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open('scaler_y.pkl', 'wb'))
print("  ✓ Scalers saved:")
print("    - scaler_X.pkl (for input features)")
print("    - scaler_y.pkl (for output targets)")

# Save normalized data as CSV (human-readable)
X_normalized_df.to_csv('X_normalized.csv', index=False)
y_normalized_df.to_csv('y_normalized.csv', index=False)
print("  ✓ Normalized data saved as CSV:")
print("    - X_normalized.csv ({} samples × {} features)".format(X_normalized.shape[0], X_normalized.shape[1]))
print("    - y_normalized.csv ({} samples × {} targets)".format(y_normalized.shape[0], y_normalized.shape[1]))

# Save normalized data as pickle (faster for numpy arrays)
pickle.dump(X_normalized, open('X_normalized.pkl', 'wb'))
pickle.dump(y_normalized, open('y_normalized.pkl', 'wb'))
print("  ✓ Normalized data saved as Pickle (faster loading):")
print("    - X_normalized.pkl")
print("    - y_normalized.pkl")

# ============================================================================
# SECTION 5: Verification
# ============================================================================
print("\n[VERIFICATION] Checking normalized data quality...")

# Check ranges
print(f"  ✓ X_normalized min: {X_normalized.min():.6f} (should be 0)")
print(f"  ✓ X_normalized max: {X_normalized.max():.6f} (should be 1)")
print(f"  ✓ y_normalized min: {y_normalized.min():.6f} (should be 0)")
print(f"  ✓ y_normalized max: {y_normalized.max():.6f} (should be 1)")

# Check for NaN/Inf
nan_X = np.isnan(X_normalized).sum()
nan_y = np.isnan(y_normalized).sum()
inf_X = np.isinf(X_normalized).sum()
inf_y = np.isinf(y_normalized).sum()

print(f"  ✓ NaN in X_normalized: {nan_X} (should be 0)")
print(f"  ✓ NaN in y_normalized: {nan_y} (should be 0)")
print(f"  ✓ Inf in X_normalized: {inf_X} (should be 0)")
print(f"  ✓ Inf in y_normalized: {inf_y} (should be 0)")

if nan_X + nan_y + inf_X + inf_y > 0:
    print("  ⚠️  WARNING: Found NaN or Inf values!")
else:
    print("  ✓ Data quality check PASSED")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("NORMALIZATION COMPLETE!")
print("="*80)

print(f"""
✓ Data successfully normalized!

ORIGINAL DATA RANGES:
  Features:
    - initial_diameter_um:    {X['initial_diameter_um'].min():.2f} - {X['initial_diameter_um'].max():.2f}
    - time_ms:                {X['time_ms'].min():.2f} - {X['time_ms'].max():.2f}
    - T_K:                    {X['T_K'].min():.2f} - {X['T_K'].max():.2f} (constant)
    - p_Pa:                   {X['p_Pa'].min():.0f} (constant)
    - rho_kg_m3:              {X['rho_kg_m3'].min():.4f} - {X['rho_kg_m3'].max():.4f}
    - O2_fraction:            {X['O2_fraction'].min():.4f} - {X['O2_fraction'].max():.4f}
    - vel_mag_m_s:            {X['vel_mag_m_s'].min():.6f} - {X['vel_mag_m_s'].max():.6f}
    - etc.
    
  Targets:
    - diameter_um:            {y['diameter_um'].min():.2f} - {y['diameter_um'].max():.2f}
    - evaporation_time_ms:    {y['evaporation_time_ms'].min():.2f} - {y['evaporation_time_ms'].max():.2f}
    - emissions_proxy:        {y['emissions_proxy'].min():.0f} - {y['emissions_proxy'].max():.0f}

NORMALIZED DATA RANGES (all [0, 1]):
  Features:
    - All {X_normalized.shape[1]} features normalized to 0-1 range
    - Constant features (T, p) mapped to constant values
    - Ready for neural network training!
    
  Targets:
    - All {y_normalized.shape[1]} targets normalized to 0-1 range
    - Ready for model learning!

FILES CREATED:
  [Scalers - SAVE THESE!]
    1. scaler_X.pkl              ← For un-normalizing test/validation predictions
    2. scaler_y.pkl              ← For un-normalizing model outputs
  
  [Normalized Data - CSV format (human-readable)]
    3. X_normalized.csv          ← {X_normalized.shape[0]} × {X_normalized.shape[1]} features
    4. y_normalized.csv          ← {y_normalized.shape[0]} × {y_normalized.shape[1]} targets
  
  [Normalized Data - Pickle format (faster loading)]
    5. X_normalized.pkl          ← Same as CSV, optimized for Python
    6. y_normalized.pkl          ← Same as CSV, optimized for Python

KEY INFORMATION:
  - Samples: {len(X_normalized)}
  - Input features: {X_normalized.shape[1]}
  - Output targets: {y_normalized.shape[1]}
  - Feature range: [0, 1]
  - Target range: [0, 1]

HOW TO USE THESE FILES:
  
  1. Load normalized data for training:
     >>> X_train = pd.read_csv('X_normalized.csv')
     >>> y_train = pd.read_csv('y_normalized.csv')
     
  2. Or load faster with pickle:
     >>> X_train = pickle.load(open('X_normalized.pkl', 'rb'))
     >>> y_train = pickle.load(open('y_normalized.pkl', 'rb'))
  
  3. Later, when you have predictions, un-normalize:
     >>> scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
     >>> y_pred_real = scaler_y.inverse_transform(y_pred_normalized)
     >>> # Now y_pred_real is in original units (diameter in μm, etc.)

NEXT STEPS:
  1. Review normalized data (open X_normalized.csv in Excel)
  2. Proceed to Step 4: Train/Val/Test Split
  3. Define neural network architecture (Step 5)
  4. Set up training loop (Step 6)
  5. Train and evaluate your model
  6. Validate against Sandia Flame D data

STATUS:
  ✓ Step 1: Data Exploration - COMPLETE
  ✓ Step 2: Feature Extraction - COMPLETE
  ✓ Step 3: Normalization - COMPLETE
  → Step 4: Train/Val/Test Split - READY
""")

print("="*80 + "\n")

print("To continue to Step 4, run:\n  python3 step4_data_split.py\n")
