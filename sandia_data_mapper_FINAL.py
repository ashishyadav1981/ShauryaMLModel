#!/usr/bin/env python3
"""
Sandia to ML Model Feature Mapper

Maps Sandia Flame D measurements to your ML model's 10 input features:
  1. diameter (μm)         ← Estimate from flame region
  2. T_ambient (K)         ← YT (measured temperature)
  3. viscosity (cP)        ← Calculate from T
  4. pressure (Pa)         ← Atmospheric (101325)
  5. O2_conc (fraction)    ← YO2 (measured O2)
  6. velocity (m/s)        ← Mixture fraction F used as proxy
  7. fuel_type_encoded     ← CH4 = 1.0
  8. residence_time (ms)   ← r_over_d * jet diameter / F
  9. surface_tension (N/m) ← Temperature-dependent
  10. density (kg/m³)      ← Ideal gas law
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("SANDIA TO ML FEATURE MAPPER")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Load raw Sandia data
# ─────────────────────────────────────────────

print("\n[1/4] Loading Sandia Flame D data...")

try:
    sandia_raw = pd.read_csv("sandia_flame_d_raw.csv")
    
    # Convert all numeric columns to float (they may have been read as strings)
    numeric_cols = ['r_over_d', 'F', 'Frms', 'T', 'Trms', 'YO2', 'YO2rms', 'YN2', 'YN2rms',
                    'YH2', 'YH2rms', 'YH2O', 'YH2Orms', 'YCH4', 'YCH4rms', 'YCO', 'YCOrms',
                    'YCO2', 'YCO2rms', 'YOH', 'YOHrms', 'YNO', 'YNOrms']
    
    for col in numeric_cols:
        if col in sandia_raw.columns:
            sandia_raw[col] = pd.to_numeric(sandia_raw[col], errors='coerce')
    
    print(f"  ✓ Loaded: {len(sandia_raw)} measurements")
    print(f"  ✓ Columns: {list(sandia_raw.columns)}")
    print(f"  ✓ Converted to numeric types")
except FileNotFoundError:
    print("❌ ERROR: sandia_flame_d_raw.csv not found!")
    print("   Run: python3 sandia_yave_parser_FINAL.py first")
    exit()

# ─────────────────────────────────────────────
# 2. Sandia flame D parameters
# ─────────────────────────────────────────────

# Sandia Flame D jet specifications
SANDIA_JET_DIAMETER = 0.0049  # meters (4.9 mm)
SANDIA_FUEL = "CH4"
SANDIA_PRESSURE = 101325.0    # Pa (1 atm)

print("\n[2/4] Sandia Flame D specifications:")
print(f"  Jet diameter: {SANDIA_JET_DIAMETER*1000:.1f} mm")
print(f"  Fuel: {SANDIA_FUEL}")
print(f"  Pressure: {SANDIA_PRESSURE:.0f} Pa")

# ─────────────────────────────────────────────
# 3. Create feature vectors
# ─────────────────────────────────────────────

print("\n[3/4] Mapping to model features...")

prepared = pd.DataFrame()

# 1. DIAMETER: Estimate from flame region
#    In post-flame region: typical droplets 1-50 μm
#    Use mixture fraction F as indicator
#    Higher F (unburnt) -> larger droplets
#    Lower F (burnt) -> smaller droplets
prepared['diameter'] = 50 * sandia_raw['F']  # Range: 0-50 μm based on F

# 2. T_AMBIENT: Use measured temperature
prepared['T_ambient'] = sandia_raw['T']

# 3. VISCOSITY: Dynamic viscosity from temperature (Sutherland's law)
def estimate_viscosity(T_K):
    """Estimate dynamic viscosity (cP) using simplified Sutherland law for air"""
    # For air: μ(T) ≈ 1.715e-5 * (T/273.15)^0.5 / (1 + 110.4/T) Pa·s
    # Simplified: μ ≈ 1.48e-6 * T^0.5 Pa·s
    mu_pas = 1.48e-6 * np.power(T_K, 0.5)
    mu_cp = mu_pas * 100  # Convert Pa·s to cP
    return np.clip(mu_cp, 0.001, 1.0)  # Reasonable range

prepared['viscosity'] = estimate_viscosity(sandia_raw['T'])

# 4. PRESSURE: Sandia is at atmospheric
prepared['pressure'] = SANDIA_PRESSURE

# 5. O2_CONCENTRATION: From Sandia YO2 (already mass fraction)
prepared['O2_conc'] = sandia_raw['YO2']

# 6. VELOCITY: Estimate from mixture fraction
#    Pure unburnt jet (F=1) at ~25 m/s
#    Pure burnt (F=0) at ~0 m/s
#    Linear interpolation
prepared['velocity'] = 25.0 * sandia_raw['F']

# 7. FUEL_TYPE_ENCODED: CH4 = 1.0
prepared['fuel_type_encoded'] = 1.0

# 8. RESIDENCE_TIME: Distance / velocity
#    r_over_d is radial position, multiply by jet diameter to get meters
#    Axial location from file (x/d): D01 means 1D, D15 means 15D, etc.
#    Use axial location as proxy for residence time
axial_position_m = sandia_raw['axial_location'] * SANDIA_JET_DIAMETER
velocity_ms = prepared['velocity'].clip(lower=0.1)  # Avoid division by zero
prepared['residence_time'] = (axial_position_m / velocity_ms) * 1000  # Convert to ms

# 9. SURFACE_TENSION: Temperature-dependent for air/droplet interface
def estimate_surface_tension(T_K):
    """Estimate surface tension (N/m) at temperature T (K)"""
    # For air: decreases with temperature
    # Reference: σ ≈ 0.072 N/m at 298K
    sigma_ref = 0.072
    T_ref = 298
    # Linear approximation: σ decreases by ~0.0001 N/m per K
    sigma = sigma_ref * (1 - 0.0001 * (T_K - T_ref))
    return np.clip(sigma, 0.01, 0.1)

prepared['surface_tension'] = estimate_surface_tension(sandia_raw['T'])

# 10. DENSITY: Ideal gas law
#    ρ = P / (R_specific * T)
#    For air: R_specific ≈ 287 J/(kg·K)
R_air = 287.0
prepared['density'] = SANDIA_PRESSURE / (R_air * sandia_raw['T'])

# ─────────────────────────────────────────────
# 4. Save and verify
# ─────────────────────────────────────────────

print("\n[4/4] Verifying and saving...")

output_file = "sandia_flame_d_prepared.csv"
prepared.to_csv(output_file, index=False)
print(f"  ✓ Saved to: {output_file}")

print(f"\nFeature statistics:")
print(prepared.describe())

# Verify ranges
print("\nRange verification:")
feature_ranges = {
    'diameter': (1, 100, 'μm'),
    'T_ambient': (300, 2000, 'K'),
    'viscosity': (0.001, 1, 'cP'),
    'pressure': (100000, 102000, 'Pa'),
    'O2_conc': (0, 0.25, 'fraction'),
    'velocity': (0, 30, 'm/s'),
    'fuel_type_encoded': (1, 1, 'code'),
    'residence_time': (0.1, 1000, 'ms'),
    'surface_tension': (0.01, 0.1, 'N/m'),
    'density': (0.1, 2, 'kg/m³'),
}

for col, (min_exp, max_exp, unit) in feature_ranges.items():
    actual_min = prepared[col].min()
    actual_max = prepared[col].max()
    status = "✓" if (actual_min >= min_exp and actual_max <= max_exp) else "⚠️ "
    print(f"  {status} {col:<20} {actual_min:>10.4f} - {actual_max:>10.4f} {unit}")

# Sample data
print(f"\nSample prepared data (first 5 rows):")
print(prepared.head())

print("\n" + "=" * 80)
print("MAPPING COMPLETE!")
print("=" * 80)
print(f"""
✓ Mapped {len(prepared)} measurements to 10 features
✓ Saved to: {output_file}

Feature mapping summary:
  1. diameter         ← Estimated from mixture fraction F
  2. T_ambient        ← Sandia measured temperature
  3. viscosity        ← Calculated from T (Sutherland's law)
  4. pressure         ← Atmospheric (101325 Pa)
  5. O2_conc          ← Sandia YO2
  6. velocity         ← Estimated from F (0-25 m/s range)
  7. fuel_type        ← CH4 (code = 1.0)
  8. residence_time   ← Axial position / velocity
  9. surface_tension  ← Temperature-dependent
  10. density         ← Ideal gas law

NEXT STEP: Run sandia_validation_FINAL.py
  This will:
    1. Load your trained model
    2. Generate predictions on Sandia data
    3. Create validation report
""")

print("=" * 80)
