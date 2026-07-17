#!/usr/bin/env python3
"""
Sandia Flame D .Yave Parser
Parses ensemble-averaged flame data files

File format:
  r/d    F       Frms    T(K)   Trms   YO2   YO2rms  YN2   YN2rms  YH2   YH2rms  
  YH2O   YH2Orms YCH4    YCH4rms YCO   YCOrms  YCO2  YCO2rms YOH   YOHrms  
  YNO    YNOrms  YCOLIF  YCOrms  TNDR

Output: Combines all D*.Yave files into single CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SANDIA FLAME D .YAVE PARSER")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Find and list Sandia .Yave files
# ─────────────────────────────────────────────

print("\n[1/3] Finding Sandia Flame D data files...")

sandia_path = Path("pmCDEF/pmD.stat")

if not sandia_path.exists():
    print(f"❌ ERROR: Cannot find {sandia_path}")
    print("Expected: pmCDEF/pmD.stat/ folder with D*.Yave files")
    exit()

yave_files = sorted(sandia_path.glob("D*.Yave"))

if not yave_files:
    print(f"❌ No D*.Yave files found in {sandia_path}")
    exit()

print(f"✓ Found {len(yave_files)} Sandia Flame D measurement files:")
for f in yave_files:
    print(f"  {f.name}")

# ─────────────────────────────────────────────
# 2. Parse .Yave files
# ─────────────────────────────────────────────

print("\n[2/3] Parsing .Yave files...")

# Column names from Sandia file header
column_names = [
    'r_over_d',      # Radial position normalized by jet diameter
    'F',             # Mixture fraction (scalar dissipation proxy)
    'Frms',          # Mixture fraction rms
    'T',             # Temperature (K)
    'Trms',          # Temperature rms
    'YO2',           # O2 mass fraction
    'YO2rms',        # O2 rms
    'YN2',           # N2 mass fraction
    'YN2rms',        # N2 rms
    'YH2',           # H2 mass fraction
    'YH2rms',        # H2 rms
    'YH2O',          # H2O mass fraction
    'YH2Orms',       # H2O rms
    'YCH4',          # CH4 mass fraction (fuel)
    'YCH4rms',       # CH4 rms
    'YCO',           # CO mass fraction
    'YCOrms',        # CO rms
    'YCO2',          # CO2 mass fraction
    'YCO2rms',       # CO2 rms
    'YOH',           # OH mass fraction
    'YOHrms',        # OH rms
    'YNO',           # NO mass fraction (emissions!)
    'YNOrms',        # NO rms
    'YCOLIF',        # CO LIF measurement
    'YCOrms2',       # CO rms (duplicate?)
    'TNDR'           # Additional parameter
]

all_data = []

for yave_file in yave_files:
    try:
        # Read file, skipping header lines
        df = pd.read_csv(
            yave_file,
            delim_whitespace=True,
            skiprows=3,  # Skip title and description
            names=column_names,
            na_values=['NaN', 'nan', '-999', '---']
        )
        
        # Add source file identifier
        df['source_file'] = yave_file.stem  # D01, D02, etc.
        
        # Extract axial location from filename (D01 -> 1, D15 -> 15, etc.)
        axial_id = int(yave_file.stem[1:])
        df['axial_location'] = axial_id
        
        all_data.append(df)
        print(f"  ✓ {yave_file.name}: {len(df)} measurements")
        
    except Exception as e:
        print(f"  ⚠️  Error reading {yave_file.name}: {e}")

if not all_data:
    print("❌ No data successfully parsed!")
    exit()

# ─────────────────────────────────────────────
# 3. Combine and save
# ─────────────────────────────────────────────

print("\n[3/3] Combining and saving data...")

combined = pd.concat(all_data, ignore_index=True)
print(f"  ✓ Total measurements: {len(combined)}")

# Save to CSV
output_file = "sandia_flame_d_raw.csv"
combined.to_csv(output_file, index=False)
print(f"  ✓ Saved to: {output_file}")

# ─────────────────────────────────────────────
# 4. Display summary
# ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print(f"\nShape: {combined.shape}")
print(f"Columns: {list(combined.columns)}")

print("\nKey variables (physical units):")
print(f"  Temperature (T):     {combined['T'].min():.1f} - {combined['T'].max():.1f} K")
print(f"  O2 mass fraction:    {combined['YO2'].min():.6f} - {combined['YO2'].max():.6f}")
print(f"  CH4 mass fraction:   {combined['YCH4'].min():.6f} - {combined['YCH4'].max():.6f}")
print(f"  N2 mass fraction:    {combined['YN2'].min():.6f} - {combined['YN2'].max():.6f}")
print(f"  NO mass fraction:    {combined['YNO'].min():.6f} - {combined['YNO'].max():.6f}")
print(f"  H2O mass fraction:   {combined['YH2O'].min():.6f} - {combined['YH2O'].max():.6f}")

print("\nFirst few rows:")
print(combined[['source_file', 'r_over_d', 'T', 'YCH4', 'YO2', 'YNO']].head(10))

print("\n" + "=" * 80)
print("PARSING COMPLETE!")
print("=" * 80)
print(f"""
✓ Parsed {len(yave_files)} .Yave files
✓ Total measurements: {len(combined)}
✓ Saved to: {output_file}

NEXT STEP: Run sandia_data_mapper.py
  This will map Sandia variables to your model's 10 input features
""")

print("=" * 80)
