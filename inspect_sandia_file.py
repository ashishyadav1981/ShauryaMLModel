#!/usr/bin/env python3
"""
Sandia .Yave File Inspector
Reads and displays structure of Sandia Flame D data files
"""

from pathlib import Path
import pandas as pd

print("=" * 80)
print("SANDIA FILE INSPECTOR")
print("=" * 80)

# ─────────────────────────────────────────────
# 1. Find Sandia data folder
# ─────────────────────────────────────────────

print("\n[1/3] Looking for Sandia data...")

possible_paths = [
    Path("pmCDEF/pmD.stat"),
    Path("pmCDEF") / "pmD.stat",
    Path("Downloads") / "pmCDEF" / "pmD.stat",
]

sandia_folder = None
for path in possible_paths:
    if path.exists():
        sandia_folder = path
        break

if not sandia_folder:
    # Search recursively
    import os
    for root, dirs, files in os.walk("."):
        if "pmD.stat" in dirs:
            sandia_folder = Path(root) / "pmD.stat"
            break

if not sandia_folder or not sandia_folder.exists():
    print("❌ Cannot find pmD.stat folder")
    print("\nExpected location:")
    print("  C:\\Users\\Ashish\\Downloads\\ShauryaMLModel\\pmCDEF\\pmD.stat\\")
    print("\nOr anywhere in your project folder with structure:")
    print("  pmCDEF/pmD.stat/D01.Yave, D02.Yave, etc.")
    exit()

print(f"✓ Found: {sandia_folder.absolute()}")

# ─────────────────────────────────────────────
# 2. List available files
# ─────────────────────────────────────────────

print("\n[2/3] Listing available files...")

yave_files = sorted(sandia_folder.glob("*.Yave"))

if not yave_files:
    print("❌ No .Yave files found!")
    # Show what's there
    all_files = sorted(sandia_folder.glob("*"))
    print(f"\nFiles in {sandia_folder}:")
    for f in all_files[:20]:
        print(f"  {f.name}")
    exit()

print(f"✓ Found {len(yave_files)} .Yave files:")
for f in yave_files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<20} ({size_kb:>6.1f} KB)")

# ─────────────────────────────────────────────
# 3. Read and inspect first file
# ─────────────────────────────────────────────

print("\n[3/3] Inspecting first file...")

test_file = yave_files[0]
print(f"\nFile: {test_file.name}")
print("─" * 80)

# Try reading as text first to see header
print("\nFIRST 20 LINES (text):")
print("─" * 80)

with open(test_file, 'r') as f:
    for i, line in enumerate(f):
        if i < 20:
            print(line.rstrip())
        else:
            break

# Try reading as pandas dataframe
print("\n" + "─" * 80)
print("PARSED DATA:")
print("─" * 80)

try:
    df = pd.read_csv(test_file, 
                     sep=r'\s+',
                     comment='#',
                     engine='python',
                     na_values=['NaN', 'nan', '-999'])
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Save sample
    df.to_csv("sandia_sample_inspection.csv", index=False)
    print(f"\n✓ Sample saved to: sandia_sample_inspection.csv")
    
except Exception as e:
    print(f"⚠️  Error parsing as CSV: {e}")
    print("\nTrying alternative parsing methods...")
    
    # Try reading with different separator
    try:
        df = pd.read_csv(test_file, sep='\s+', comment='#', engine='python')
        print(f"✓ Successfully read with regex separator")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e2:
        print(f"❌ Could not parse file: {e2}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("INSPECTION SUMMARY")
print("=" * 80)
print(f"""
✓ Located Sandia data folder
✓ Found {len(yave_files)} .Yave files
✓ Inspected: {test_file.name}

NEXT STEPS:
1. Check the columns listed above
2. Tell me what columns you see (especially T, CH4, O2, U, etc.)
3. I'll update the parser to handle your specific file format
4. Run validation!

If columns look good, run:
  python3 sandia_yall_parser.py

This will parse all files and create:
  sandia_flame_d_raw.csv
""")

print("=" * 80)
