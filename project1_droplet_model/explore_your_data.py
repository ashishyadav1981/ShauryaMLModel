#!/usr/bin/env python3
"""
Step 2: Data Exploration for Your Eulerian CFD Data
Run this script to understand your data before ML training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# SECTION 1: LOAD AND BASIC EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA EXPLORATION - YOUR EULERIAN CFD DATA")
print("="*80 + "\n")

# Load your data
data_path = 'Eulerian_1_-_eulerian1_1.csv'  # Update path if needed
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

print(f"✓ Data loaded successfully!")
print(f"\nDataset shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ============================================================================
# SECTION 2: COLUMN INFORMATION
# ============================================================================
print("\n" + "="*80)
print("COLUMN INFORMATION")
print("="*80 + "\n")

print("Data types:")
print(data.dtypes)

print("\nColumn categories:")
print("  Flow properties:  T (temperature), p (pressure), rho (density)")
print("  Velocity:         U:0, U:1, U:2 (X, Y, Z components in m/s)")
print("  Species:          C7H16, CO2, H2O, N2, O2 (mass fractions)")
print("  Spatial:          Points:0, Points:1, Points:2 (X, Y, Z in m)")
print("  Metadata:         TimeStep")

# ============================================================================
# SECTION 3: MISSING DATA CHECK
# ============================================================================
print("\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80 + "\n")

print("Missing values per column:")
missing = data.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(missing[missing > 0])

print("\nInfinite values per column:")
inf_count = (np.isinf(data.select_dtypes(include=[np.number]))).sum()
if inf_count.sum() == 0:
    print("  ✓ No infinite values!")
else:
    print(inf_count[inf_count > 0])

# ============================================================================
# SECTION 4: STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEMPERATURE & PRESSURE")
print("="*80 + "\n")

print(f"Temperature (T):")
print(f"  Range:     {data['T'].min():.2f} - {data['T'].max():.2f} K")
print(f"  Mean ± Std: {data['T'].mean():.2f} ± {data['T'].std():.2f} K")
print(f"  Variance:  {(data['T'].max() - data['T'].min()) / data['T'].mean() * 100:.2f}%")

print(f"\nPressure (p):")
print(f"  Range:     {data['p'].min():.2e} - {data['p'].max():.2e} Pa")
print(f"  In MPa:    {data['p'].min()/1e6:.2f} - {data['p'].max()/1e6:.2f} MPa")
print(f"  Variance:  {(data['p'].max() - data['p'].min()) / data['p'].mean() * 100:.3f}%")
print(f"  ⚠️  Note: Pressure is nearly constant (good for single case)")

# ============================================================================
# SECTION 5: FLUID PROPERTIES
# ============================================================================
print("\n" + "="*80)
print("FLUID PROPERTIES")
print("="*80 + "\n")

print(f"Density (rho):")
print(f"  Range:     {data['rho'].min():.4f} - {data['rho'].max():.4f} kg/m³")
print(f"  Mean:      {data['rho'].mean():.4f} kg/m³")

vel_mag = np.sqrt(data['U:0']**2 + data['U:1']**2 + data['U:2']**2)
print(f"\nVelocity Magnitude:")
print(f"  X (U:0):   {data['U:0'].min():.4f} - {data['U:0'].max():.4f} m/s")
print(f"  Y (U:1):   {data['U:1'].min():.4f} - {data['U:1'].max():.4f} m/s")
print(f"  Z (U:2):   {data['U:2'].min():.4f} - {data['U:2'].max():.4f} m/s")
print(f"  Magnitude: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s (mean: {vel_mag.mean():.4f})")

# ============================================================================
# SECTION 6: SPECIES COMPOSITION
# ============================================================================
print("\n" + "="*80)
print("SPECIES COMPOSITION")
print("="*80 + "\n")

print(f"C7H16 (Heptane fuel):")
print(f"  Range:        {data['C7H16'].min()} - {data['C7H16'].max()}")
print(f"  Non-zero pts: {(data['C7H16'] > 0).sum():,} / {len(data):,}")

print(f"\nO2 (Oxygen):")
print(f"  Range:  {data['O2'].min():.4f} - {data['O2'].max():.4f}")
print(f"  Mean:   {data['O2'].mean():.4f}")

print(f"\nN2 (Nitrogen):")
print(f"  Range:  {data['N2'].min():.4f} - {data['N2'].max():.4f}")
print(f"  Mean:   {data['N2'].mean():.4f}")

print(f"\nCO2 (Carbon dioxide):")
print(f"  Range:  {data['CO2'].min()} - {data['CO2'].max()}")

print(f"\nH2O (Water vapor):")
print(f"  Range:  {data['H2O'].min():.4f} - {data['H2O'].max():.4f}")

# Species conservation check
species_sum = data['C7H16'] + data['CO2'] + data['H2O'] + data['N2'] + data['O2']
print(f"\nMass fraction sum (C7H16 + CO2 + H2O + N2 + O2):")
print(f"  Min:  {species_sum.min():.6f}")
print(f"  Max:  {species_sum.max():.6f}")
print(f"  Mean: {species_sum.mean():.6f}")
if species_sum.min() < 0.95 or species_sum.max() > 1.05:
    print(f"  ⚠️  Sum ≠ 1.0 (Other species or trace compounds exist)")

# ============================================================================
# SECTION 7: SPATIAL DOMAIN
# ============================================================================
print("\n" + "="*80)
print("SPATIAL DOMAIN")
print("="*80 + "\n")

print(f"Domain dimensions:")
print(f"  X (Points:0): {data['Points:0'].min():.4f} - {data['Points:0'].max():.4f} m")
print(f"  Y (Points:1): {data['Points:1'].min():.4f} - {data['Points:1'].max():.4f} m")
print(f"  Z (Points:2): {data['Points:2'].min():.4f} - {data['Points:2'].max():.4f} m")

dx = data['Points:0'].unique()
dy = data['Points:1'].unique()
dz = data['Points:2'].unique()
print(f"\nGrid resolution:")
print(f"  X points: {len(dx)}")
print(f"  Y points: {len(dy)}")
print(f"  Z points: {len(dz)}")
print(f"  Total grid cells: {len(dx)} × {len(dy)} × {len(dz)} = {len(dx)*len(dy)*len(dz):,} (actual: {len(data):,})")

# ============================================================================
# SECTION 8: TIME METADATA
# ============================================================================
print("\n" + "="*80)
print("TIME METADATA")
print("="*80 + "\n")

print(f"Unique TimeSteps: {data['TimeStep'].unique()}")
print(f"⚠️  Only 1 time step present (snapshot at t={data['TimeStep'].iloc[0]})")
print(f"   For droplet training, you'll need multiple time steps or multiple cases")

# ============================================================================
# SECTION 9: SUMMARY STATISTICS TABLE
# ============================================================================
print("\n" + "="*80)
print("COMPLETE STATISTICS TABLE")
print("="*80 + "\n")

numeric_cols = data.select_dtypes(include=[np.number]).columns
print(data[numeric_cols].describe().round(6))

# ============================================================================
# SECTION 10: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS FOR ML TRAINING")
print("="*80 + "\n")

print("""
1. CURRENT STATE:
   ✓ One spatial snapshot (Eulerian field) with 178K grid points
   ✓ Complete and clean data (no missing values)
   ✓ Single case: pressure ~5 MPa, temperature ~800 K, heptane fuel
   
2. WHAT'S MISSING FOR ML:
   ✗ Multiple cases (different P, T, fuel, droplet sizes)
   ✗ Time evolution (how field changes over time)
   ✗ Droplet-specific outputs (evaporation time, final size, emissions)
   
3. NEXT STEPS:
   a) Obtain multiple CFD simulations from Dr. Mishra
   b) Collect them in a single dataset (CSV with cases as rows)
   c) Define inputs (boundary conditions) and outputs (droplet properties)
   d) Proceed to Step 3 (Normalization)
   
4. EXAMPLE FORMAT (Target):
   
   Diameter  Pressure  Temp  Fuel_Type  Evap_Time  Emissions
   10μm      5 MPa     800K  C7H16      1.2 ms     45 ppm
   20μm      5 MPa     800K  C7H16      2.5 ms     42 ppm
   50μm      5 MPa     1200K C7H16      5.1 ms     38 ppm
   ...
   
5. RESOURCES:
   - Sandia Flame D (validation): Find published datasets online
   - Your CFD solver documentation for extracting droplet properties
   - Dr. Mishra's prior publications for methodology
""")

# ============================================================================
# SECTION 11: SAVE DATA SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SAVING DATA SUMMARY")
print("="*80 + "\n")

summary_dict = {
    'Variable': ['TimeStep', 'T (K)', 'p (Pa)', 'rho (kg/m³)', 'C7H16', 'O2', 'N2', 
                 'U_mag (m/s)', 'Vel_X', 'Vel_Y', 'Vel_Z', 'X_coord', 'Y_coord', 'Z_coord'],
    'Min': [
        data['TimeStep'].min(), data['T'].min(), data['p'].min(), data['rho'].min(),
        data['C7H16'].min(), data['O2'].min(), data['N2'].min(),
        vel_mag.min(), data['U:0'].min(), data['U:1'].min(), data['U:2'].min(),
        data['Points:0'].min(), data['Points:1'].min(), data['Points:2'].min()
    ],
    'Max': [
        data['TimeStep'].max(), data['T'].max(), data['p'].max(), data['rho'].max(),
        data['C7H16'].max(), data['O2'].max(), data['N2'].max(),
        vel_mag.max(), data['U:0'].max(), data['U:1'].max(), data['U:2'].max(),
        data['Points:0'].max(), data['Points:1'].max(), data['Points:2'].max()
    ],
    'Mean': [
        data['TimeStep'].mean(), data['T'].mean(), data['p'].mean(), data['rho'].mean(),
        data['C7H16'].mean(), data['O2'].mean(), data['N2'].mean(),
        vel_mag.mean(), data['U:0'].mean(), data['U:1'].mean(), data['U:2'].mean(),
        data['Points:0'].mean(), data['Points:1'].mean(), data['Points:2'].mean()
    ]
}

summary_df = pd.DataFrame(summary_dict)
summary_df.to_csv('data_summary.csv', index=False)
print("✓ Summary saved to 'data_summary.csv'")

print("\n" + "="*80)
print("EXPLORATION COMPLETE!")
print("="*80)
print("\nNext: Run visualization script (explore_visualizations.py)")
print("      or proceed to Step 3 (Normalization) once you have multiple cases")
