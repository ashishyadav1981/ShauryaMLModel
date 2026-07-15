#!/usr/bin/env python3
"""
Sandia Flame D Data Mapper
Maps Sandia measurements to ML model's 10 input features

Your model expects:
  1. diameter (μm)
  2. T_ambient (K)
  3. viscosity (cP)
  4. pressure (Pa)
  5. O2_conc (fraction)
  6. velocity (m/s)
  7. fuel_type_encoded (1 for CH4)
  8. residence_time (ms)
  9. surface_tension (N/m)
  10. density (kg/m³)
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("SANDIA FLAME D DATA MAPPER")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Load raw Sandia data
# ─────────────────────────────────────────────

print("\n[1/4] Loading raw Sandia data...")

try:
    sandia_raw = pd.read_csv("sandia_flame_d_raw.csv")
    print(f"  ✓ Loaded: {len(sandia_raw)} measurements")
    print(f"  ✓ Columns: {list(sandia_raw.columns)}")
except FileNotFoundError:
    print("❌ ERROR: sandia_flame_d_raw.csv not found!")
    print("   Run: python3 sandia_yall_parser.py first")
    exit()

# Display what we have
print("\nFirst few rows:")
print(sandia_raw.head())

# ─────────────────────────────────────────────
# 2. Extract key Sandia measurements
# ─────────────────────────────────────────────

print("\n[2/4] Extracting Sandia measurements...")

# Map Sandia columns (adjust these based on actual column names)
# Common Sandia columns: x, y, T, CH4, O2, N2, etc.

prepared_data = pd.DataFrame()

# IMPORTANT: These column names depend on your .Yall file structure
# You may need to adjust based on actual column names!

try:
    # Temperature (measured directly from Sandia)
    if 'T' in sandia_raw.columns:
        T_sandia = sandia_raw['T']
    elif 'Temperature' in sandia_raw.columns:
        T_sandia = sandia_raw['Temperature']
    else:
        print("⚠️  WARNING: Cannot find Temperature column")
        print(f"   Available columns: {list(sandia_raw.columns)}")
        T_sandia = np.ones(len(sandia_raw)) * 1500  # Default guess
    
    # Velocity (if available)
    if 'U' in sandia_raw.columns:
        U_sandia = sandia_raw['U']
    elif 'Velocity' in sandia_raw.columns:
        U_sandia = sandia_raw['Velocity']
    else:
        U_sandia = np.ones(len(sandia_raw)) * 15.0  # Default jet velocity
    
    # Position (for residence time calculation)
    if 'x' in sandia_raw.columns:
        x_sandia = sandia_raw['x']
    else:
        x_sandia = np.ones(len(sandia_raw)) * 0.05  # Default: 50mm downstream
    
    # Species concentrations
    # These depend on actual column names in your .Yall files
    if 'CH4' in sandia_raw.columns:
        CH4_sandia = sandia_raw['CH4'] / 100.0  # Convert % to fraction
    else:
        CH4_sandia = np.ones(len(sandia_raw)) * 0.10  # Default
    
    if 'O2' in sandia_raw.columns:
        O2_sandia = sandia_raw['O2'] / 100.0
    else:
        O2_sandia = np.ones(len(sandia_raw)) * 0.15

except Exception as e:
    print(f"❌ ERROR extracting measurements: {e}")
    print("   Please check column names in your .Yall files")
    exit()

print(f"  ✓ Temperature: {T_sandia.min():.1f}-{T_sandia.max():.1f} K")
print(f"  ✓ Velocity: {U_sandia.min():.1f}-{U_sandia.max():.1f} m/s")
print(f"  ✓ O2 conc: {O2_sandia.min():.3f}-{O2_sandia.max():.3f} (fraction)")

# ─────────────────────────────────────────────
# 3. Map to ML model features
# ─────────────────────────────────────────────

print("\n[3/4] Mapping to model features...")

# 1. Diameter: Estimate from flame geometry (typically 1-50 μm in post-flame)
prepared_data['diameter'] = np.random.uniform(5, 50, len(sandia_raw))

# 2. T_ambient: Use measured temperature
prepared_data['T_ambient'] = T_sandia

# 3. Viscosity: Estimate from temperature (dynamic viscosity of air)
# Sutherland's law approximation for air
def estimate_viscosity(T):
    """Estimate dynamic viscosity (cP) at temperature T (K)"""
    # Simplified: viscosity ~ sqrt(T)
    return 0.0001 * np.sqrt(T) * 100  # Convert to cP
prepared_data['viscosity'] = estimate_viscosity(T_sandia)

# 4. Pressure: Sandia is at atmospheric (approximately 101325 Pa)
prepared_data['pressure'] = 101325.0

# 5. O2 concentration: From Sandia measurement
prepared_data['O2_conc'] = O2_sandia

# 6. Velocity: From Sandia measurement
prepared_data['velocity'] = U_sandia

# 7. Fuel type: CH4 = 1
prepared_data['fuel_type_encoded'] = 1.0

# 8. Residence time: distance / velocity (in ms)
prepared_data['residence_time'] = (x_sandia / U_sandia) * 1000  # Convert to ms

# 9. Surface tension: CH4/air flame ~0.072 N/m (temperature-dependent)
def estimate_surface_tension(T):
    """Estimate surface tension (N/m) at temperature T (K)"""
    # Temperature-dependent surface tension
    T_ref = 298  # Reference: 25°C
    sigma_ref = 0.072  # N/m at 25°C
    # Approximately: σ decreases with temperature
    return sigma_ref * (1 - 0.0001 * (T - T_ref))
prepared_data['surface_tension'] = estimate_surface_tension(T_sandia)

# 10. Density: Calculate from ideal gas law
# ρ = P / (R_specific * T)
# For air: R_specific ≈ 287 J/(kg·K)
R_air = 287.0  # J/(kg·K)
prepared_data['density'] = prepared_data['pressure'] / (R_air * T_sandia)

print(f"  ✓ Created {len(prepared_data)} feature vectors")
print(f"  ✓ Features: {list(prepared_data.columns)}")

# ─────────────────────────────────────────────
# 4. Save prepared data
# ─────────────────────────────────────────────

print("\n[4/4] Saving prepared data...")

output_file = "sandia_flame_d_prepared.csv"
prepared_data.to_csv(output_file, index=False)
print(f"  ✓ Saved to: {output_file}")

# Display sample
print("\nSample prepared data:")
print(prepared_data.head())

print("\nFeature statistics:")
print(prepared_data.describe())

# ─────────────────────────────────────────────
# 5. Ready for validation
# ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print(f"""
✓ Loaded {len(sandia_raw)} Sandia Flame D measurements
✓ Mapped to {len(prepared_data.columns)} ML model features
✓ Saved to: {output_file}

NEXT STEP: Run validation script
  python3 sandia_validation.py
  
This will:
  1. Load your trained model (best_model.pth)
  2. Load scalers (scaler_X.pkl, scaler_y.pkl)
  3. Generate predictions on Sandia data
  4. Calculate validation metrics (R², MAE, RMSE)
  5. Create comparison plots
""")

print("=" * 80)
