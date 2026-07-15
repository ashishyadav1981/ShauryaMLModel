#!/usr/bin/env python3
"""
Sandia Flame D Data Parser
Converts .Yall format files to CSV for ML model validation

Reads Flame D .Yall files and extracts:
  - Position: x, y, z
  - Temperature: T
  - Species: CH4, O2, N2, CO2, H2O, H2, CO, OH, NO
  - Velocity: U (if available)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SANDIA FLAME D DATA PARSER")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Find Sandia data folder
# ─────────────────────────────────────────────

sandia_path = Path("pmCDEF") / "pmD.scat"

if not sandia_path.exists():
    print(f"\n❌ ERROR: Cannot find {sandia_path}")
    print("Expected structure:")
    print("  pmCDEF/")
    print("    pmD.scat/")
    print("      D15.Yall")
    print("      D30.Yall")
    print("      ... etc")
    exit()

print(f"\n✓ Found Sandia data at: {sandia_path.absolute()}")

# ─────────────────────────────────────────────
# 2. List available .Yall files for Flame D
# ─────────────────────────────────────────────

yall_files = sorted(sandia_path.glob("D*.Yall"))

if not yall_files:
    print(f"❌ No .Yall files found in {sandia_path}")
    exit()

print(f"\n✓ Found {len(yall_files)} Flame D measurement files:")
for f in yall_files:
    print(f"  - {f.name}")

# ─────────────────────────────────────────────
# 3. Parse .Yall file format
# ─────────────────────────────────────────────

def parse_yall_file(filepath):
    """
    Parse a .Yall format file
    
    Format (typical):
    # Header information
    x(m)  y(m)  T(K)  CH4(%)  O2(%)  N2(%)  Products...
    ----  ----  ----  ------  -----  -----  ---------
    data values...
    """
    
    try:
        # Try reading with pandas (handles whitespace-separated)
        df = pd.read_csv(filepath, delim_whitespace=True, comment='#', 
                         skip_blank_lines=True, engine='python')
        return df
    except Exception as e:
        print(f"  ⚠️  Error parsing {filepath.name}: {e}")
        return None

# ─────────────────────────────────────────────
# 4. Parse all Flame D files
# ─────────────────────────────────────────────

print("\n[1/3] Parsing .Yall files...")

all_data = []

for yall_file in yall_files:
    df = parse_yall_file(yall_file)
    
    if df is not None:
        # Add source file identifier
        df['source_file'] = yall_file.stem  # D15, D30, D45, etc.
        all_data.append(df)
        print(f"  ✓ {yall_file.name}: {len(df)} measurements")
    else:
        print(f"  ✗ {yall_file.name}: FAILED")

if not all_data:
    print("❌ No data successfully parsed!")
    exit()

# ─────────────────────────────────────────────
# 5. Combine all measurements
# ─────────────────────────────────────────────

print("\n[2/3] Combining measurements...")

combined_data = pd.concat(all_data, ignore_index=True)
print(f"  ✓ Total measurements: {len(combined_data)}")
print(f"  ✓ Columns: {list(combined_data.columns)}")

# ─────────────────────────────────────────────
# 6. Display sample data
# ─────────────────────────────────────────────

print("\n[3/3] Sample data:")
print(combined_data.head(10))

print(f"\nData shape: {combined_data.shape}")
print(f"Column types:\n{combined_data.dtypes}")

# ─────────────────────────────────────────────
# 7. Save to CSV for inspection
# ─────────────────────────────────────────────

output_file = "sandia_flame_d_raw.csv"
combined_data.to_csv(output_file, index=False)
print(f"\n✓ Saved raw data to: {output_file}")

# ─────────────────────────────────────────────
# 8. Display statistics
# ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("DATA STATISTICS")
print("=" * 80)

numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
print(combined_data[numeric_cols].describe())

# ─────────────────────────────────────────────
# 9. Next steps
# ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print(f"""
✓ Successfully parsed Sandia Flame D data
✓ Saved raw data to: {output_file}
✓ Total measurements: {len(combined_data)}

NEXT: Review the columns to understand your data structure.
      Then proceed to: sandia_data_mapper.py
      
This script will:
  1. Map Sandia columns to your model's 10 input features
  2. Create sandia_flame_d_prepared.csv
  3. Run validation with your trained model
""")

print("=" * 80)
