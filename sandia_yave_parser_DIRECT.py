#!/usr/bin/env python3
"""
Sandia Flame D Parser - Direct Path
Uses known path: pmCDEF\pmCDEFarchives\pmD.stat
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

print("=" * 80)
print("SANDIA FLAME D .YAVE PARSER - DIRECT PATH")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Set direct path
# ─────────────────────────────────────────────

print("\n[1/4] Locating Sandia data...")

# Direct path based on your file structure
sandia_path = Path.cwd() / "pmCDEF" / "pmCDEFarchives" / "pmD.stat"

print(f"Looking for: {sandia_path}")

if not sandia_path.exists():
    # Try alternative paths
    alt_paths = [
        Path("pmCDEF/pmCDEFarchives/pmD.stat"),
        Path.home() / "Downloads" / "ShauryaMLModel" / "pmCDEF" / "pmCDEFarchives" / "pmD.stat",
    ]
    
    found = False
    for alt_path in alt_paths:
        if alt_path.exists():
            sandia_path = alt_path
            found = True
            print(f"✓ Found at: {sandia_path}")
            break
    
    if not found:
        print(f"❌ ERROR: Cannot find pmD.stat")
        print(f"\nTried:")
        print(f"  {sandia_path}")
        for alt in alt_paths:
            print(f"  {alt}")
        print(f"\nYour current directory:")
        print(f"  {Path.cwd()}")
        exit()
else:
    print(f"✓ Found at: {sandia_path}")

# ─────────────────────────────────────────────
# 2. List .Yave files
# ─────────────────────────────────────────────

print("\n[2/4] Finding .Yave files...")

yave_files = sorted(sandia_path.glob("D*.Yave"))

if not yave_files:
    print(f"❌ No D*.Yave files found in {sandia_path}")
    # List what's there
    print("\nFiles in directory:")
    for item in sorted(sandia_path.glob("*"))[:20]:
        print(f"  {item.name}")
    exit()

print(f"✓ Found {len(yave_files)} Flame D measurement files:")
for f in yave_files:
    print(f"  {f.name}")

# ─────────────────────────────────────────────
# 3. Parse all .Yave files
# ─────────────────────────────────────────────

print("\n[3/4] Parsing .Yave files...")

column_names = [
    'r_over_d', 'F', 'Frms', 'T', 'Trms', 'YO2', 'YO2rms', 'YN2', 'YN2rms',
    'YH2', 'YH2rms', 'YH2O', 'YH2Orms', 'YCH4', 'YCH4rms', 'YCO', 'YCOrms',
    'YCO2', 'YCO2rms', 'YOH', 'YOHrms', 'YNO', 'YNOrms', 'YCOLIF', 'YCOrms2', 'TNDR'
]

all_data = []

for yave_file in yave_files:
    try:
        df = pd.read_csv(
            yave_file,
            sep='\s+',  # Use regex for whitespace separation (replaces delim_whitespace)
            skiprows=3,
            names=column_names,
            na_values=['NaN', 'nan', '-999', '---'],
            engine='python'
        )
        
        df['source_file'] = yave_file.stem
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
# 4. Combine and save
# ─────────────────────────────────────────────

print("\n[4/4] Combining and saving data...")

combined = pd.concat(all_data, ignore_index=True)

output_file = "sandia_flame_d_raw.csv"
combined.to_csv(output_file, index=False)

print(f"  ✓ Total measurements: {len(combined)}")
print(f"  ✓ Saved to: {output_file}")

# Display summary
print("\n" + "=" * 80)
print("PARSING COMPLETE!")
print("=" * 80)
print(f"""
✓ Parsed {len(yave_files)} .Yave files
✓ Total measurements: {len(combined)}
✓ Saved to: {output_file}

Key variables found:
  Temperature (T):  {combined['T'].min():.1f} - {combined['T'].max():.1f} K
  O2 fraction:      {combined['YO2'].min():.6f} - {combined['YO2'].max():.6f}
  CH4 fraction:     {combined['YCH4'].min():.6f} - {combined['YCH4'].max():.6f}
  NO fraction:      {combined['YNO'].min():.6f} - {combined['YNO'].max():.6f}

NEXT: Run sandia_data_mapper_FINAL.py
""")

print("=" * 80)
