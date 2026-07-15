# PATH 2: Complete Implementation Guide
## Extract Droplet Features from Eulerian Field → ML Training Ready

---

## Overview

You're implementing **Path 2** because:
✓ You have one Eulerian CFD snapshot (178K grid points)  
✓ You want to extract features to train a droplet size evolution model  
✓ You need synthetic training data before proceeding to ML training  

This guide walks through the **complete workflow** from your CFD file to ML-ready training data.

---

## Workflow: 4 Main Phases

```
Phase 1: Load Eulerian Field
         ↓
Phase 2: Create Virtual Droplet Parcels
         ↓
Phase 3: Extract Features & Generate Targets
         ↓
Phase 4: Export Training Data
         ↓
        Ready for Step 3 (Normalization)
```

---

## Phase 1: Load Your Eulerian Data

Your file: `Eulerian_1_-_eulerian1_1.csv`

**What it contains:**
- 178,164 grid points across a 100mm³ combustion domain
- Field variables: T (777–800 K), p (5 MPa), ρ, species (C7H16, O2, N2, CO2, H2O)
- Velocity components: U:0, U:1, U:2
- Spatial coordinates: Points:0, Points:1, Points:2

**Key insight:** Each grid point represents a location in the domain. We'll use these to create "parcel" regions where droplets would experience similar conditions.

---

## Phase 2: Create Virtual Droplet Parcels

**Concept:** Divide the domain into small regions (parcels).  
Each parcel = one droplet at one location = one training example.

```
Domain (100mm × 100mm × 100mm)
│
├─ Parcel 1 (15mm × 15mm × 5mm region)
│  ├─ Averaged T = 799 K
│  ├─ Averaged O2 = 0.20
│  ├─ Averaged velocity = 0.05 m/s
│  └─ Location: (20mm, 50mm, 5mm)
│
├─ Parcel 2 (next region)
│  ├─ Averaged T = 780 K
│  ├─ Averaged O2 = 0.15
│  └─ ...
│
└─ ... ~1000 parcels total
```

**Code logic:**
```python
# Divide domain into 15mm × 15mm × 5mm parcels
parcel_dx = 0.015  # 15 mm
parcel_dy = 0.015
parcel_dz = 0.005

for each parcel location:
    - Extract all grid points in this parcel
    - Compute average T, p, ρ, species, velocity
    - Record spatial location
```

**Result:** ~1000 parcels, each with averaged field properties

---

## Phase 3: Extract Features & Generate Targets

### Features (Model Inputs)

For each parcel, extract these 10 features:

| Feature | Meaning | Unit | Example |
|---------|---------|------|---------|
| `initial_diameter_um` | Starting droplet size | μm | 10, 20, 50, 100 |
| `time_ms` | Time since injection | ms | 0, 1, 2, ..., 10 |
| `T_K` | Local temperature | K | 800 |
| `p_Pa` | Local pressure | Pa | 5e6 |
| `rho_kg_m3` | Local density | kg/m³ | 21.7 |
| `O2_fraction` | Oxygen availability | - | 0.23 |
| `C7H16_fraction` | Fuel availability | - | 0.0 |
| `vel_mag_m_s` | Flow velocity magnitude | m/s | 0.05 |
| `distance_from_inlet_m` | Axial distance | m | 0.03 |
| `radial_distance_m` | Radial distance from centerline | m | 0.02 |

### Targets (Model Outputs)

For each parcel, compute these 3 target values:

| Target | Meaning | Unit | Formula |
|--------|---------|------|---------|
| `diameter_um` | Current droplet size | μm | D² Law: d² = d₀² - K·t |
| `evaporation_time_ms` | Time to complete evaporation | ms | d₀² / K |
| `emissions_proxy` | Heat release / NOx indicator | - | (T - 300) × (t_evap - t) |

**Physics Model: D² Law**

The droplet diameter squared decreases linearly with time:

```
d² = d₀² - K·t

Where:
  d₀ = initial diameter
  K = evaporation constant
  t = time
  
K depends on:
  - Local temperature (higher → faster evaporation)
  - Oxygen availability (combustion support)
  - Fuel availability (evaporation driving force)
  - Local velocity (relative velocity enhancement)
```

**Example calculation:**

```
Parcel at 800 K, high O2, high temperature:
  d₀ = 50 μm
  K = 1e-9 × (1 + 2×heat_factor) × (1 + O2_factor) × velocity_enhance
  K ≈ 1.5e-9 m²/s
  
  Evaporation time = d₀² / K = (50e-6)² / (1.5e-9) = 1.67 seconds = 1670 ms
  
At t = 100 ms:
  d² = (50e-6)² - (1.5e-9) × 0.1
  d² = 2.5e-9 - 1.5e-10 = 2.35e-9
  d = 48.5 μm (still large because evaporation is slow at 800K)
  
At t = 1000 ms:
  d² = (50e-6)² - (1.5e-9) × 1.0
  d² = 2.5e-9 - 1.5e-9 = 1.0e-9
  d = 31.6 μm (approximately 63% evaporated)
```

---

## Phase 4: Generate Training Data

The script `path2_feature_extraction.py` does this:

```python
# For each of 4 initial diameters [10, 20, 50, 100] μm:
#   For each of ~1000 parcels:
#     For each of 11 time steps [0, 1, 2, ..., 10] ms:
#       Create 1 training sample

Total samples = 4 × 1000 × 11 = 44,000 samples
```

**Output files:**

1. **droplet_trajectories_full.csv** (44,000 rows)
   - Complete data: features + targets + metadata
   - For detailed analysis and visualization

2. **droplet_features.csv** (44,000 rows, 10 cols)
   - Input features only (for training)
   - Use for X (model input)

3. **droplet_targets.csv** (44,000 rows, 3 cols)
   - Target outputs (diameter, evap time, emissions)
   - Use for y (model target)

4. **parcel_metadata.csv** (~1000 rows)
   - Spatial and field properties of each parcel
   - Useful for analysis and debugging

---

## How to Run Path 2

### Step 1: Copy Your Data
```bash
# Ensure your Eulerian data is in the same directory
cp Eulerian_1_-_eulerian1_1.csv .
```

### Step 2: Run the Extraction Script
```bash
python3 path2_feature_extraction.py
```

**Expected output:**
```
================================================================================
PATH 2: DROPLET FEATURE EXTRACTION FROM EULERIAN FIELD
================================================================================

[1/6] Loading Eulerian CFD data...
  ✓ Loaded 178,164 grid points
  Domain: X=[-0.01, 0.10] m
  ...

[2/6] Creating virtual droplet parcels...
  ✓ Created 1024 parcels

[3/6] Generating droplet size evolution targets...
  ✓ Generated 44,000 droplet evolution samples

[4/6] Preparing ML training data...
  Feature matrix shape: (44000, 10)
  Target matrix shape: (44000, 3)

[5/6] Saving data files...
  ✓ droplet_trajectories_full.csv
  ✓ droplet_features.csv
  ✓ droplet_targets.csv
  ✓ parcel_metadata.csv

[6/6] Creating visualizations...
  ✓ 01_Droplet_Evolution.png
  ✓ 02_Temperature_Effects.png
  ✓ 03_Feature_Distributions.png

✓ Created 44,000 training samples from 1 Eulerian snapshot
```

### Step 3: Verify the Data
```bash
# Check file sizes
ls -lh *.csv

# Preview the data
head droplet_features.csv
head droplet_targets.csv

# Check dimensions
wc -l droplet_features.csv  # Should be 44,001 (header + 44,000 rows)
```

---

## Understanding the Generated Data

### Example: One Training Sample

```
Features (Input):
  initial_diameter_um = 50
  time_ms = 2.5
  T_K = 799
  p_Pa = 5000000
  rho_kg_m3 = 21.7
  O2_fraction = 0.23
  C7H16_fraction = 0.0
  vel_mag_m_s = 0.05
  distance_from_inlet_m = 0.03
  radial_distance_m = 0.02

Targets (Output):
  diameter_um = 47.3
  evaporation_time_ms = 1670
  emissions_proxy = 125.4
```

**Interpretation:**
- A 50 μm droplet at time t=2.5ms in a 799K environment
- Has evaporated to 47.3 μm (97% of original)
- Will take 1670 ms total to fully evaporate
- Generates heat/NOx with magnitude 125.4

---

## Next Steps: Proceed to Step 3 (Normalization)

Once you have the 4 CSV files, you're ready for ML:

```
Step 3: Normalize Data
  ├─ Load droplet_features.csv and droplet_targets.csv
  ├─ Apply Min-Max scaling to all features
  ├─ Apply Min-Max scaling to all targets
  └─ Save scalers for later un-normalization

Step 4: Split Data
  ├─ 70% training (30,800 samples)
  ├─ 15% validation (6,600 samples)
  └─ 15% test (6,600 samples)

Step 5+: Train ML Model
  ├─ Define neural network architecture
  ├─ Set up training loop with early stopping
  ├─ Validate on test set
  └─ Test on Sandia Flame D benchmark
```

---

## Validation Against Sandia Flame D

Once your model is trained, you'll validate it against real experimental data.

**Sandia Flame D Link:** https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/

See `SANDIA_VALIDATION_GUIDE.md` for complete workflow.

---

## Important Notes

### 1. Physics-Based vs Data-Driven
This implementation uses **physics-informed synthetic targets** (D² law).
- **Pro:** Targets are physically realistic
- **Con:** Model will learn the D² law, not novel patterns

**For future work:** Replace D² law with actual CFD-derived droplet properties from multiple simulations.

### 2. Single Snapshot Limitation
You're extracting from **one Eulerian snapshot** at one time instant.
- ~1000 spatial locations (parcels)
- 4 initial diameters (representative sizes)
- 11 time steps (synthetic evolution)
= 44,000 training samples

**Ideal (future):** Multiple CFD cases with different conditions would give millions of samples.

### 3. Feature Engineering Opportunities
Current 10 features are basic. You could add:
- Reynolds number (inertial effects)
- Damköhler number (combustion speed)
- Evaporation number (ratio of evaporation to convection)
- Mixture fraction (fuel-air mixing)
- Turbulent intensity (mixing rate)

---

## Troubleshooting

### Problem: "File not found: Eulerian_1_-_eulerian1_1.csv"
**Solution:** Ensure the file is in the same directory where you run the script.
```bash
ls Eulerian_1_-_eulerian1_1.csv
```

### Problem: "Not enough points in parcel"
**Solution:** Increase `parcel_dx`, `parcel_dy`, `parcel_dz` if parcels are empty.
```python
parcel_dx = 0.020  # Increase from 0.015 to 0.020
```

### Problem: "Memory error with 178K points"
**Solution:** Process in chunks or reduce parcel grid resolution.
```python
# Sample grid instead of using all points
data = data.sample(frac=0.5, random_state=42)  # Use 50% of points
```

---

## Quality Checks

After running the script, verify:

### 1. Data Completeness
```bash
# Check no NaN or Inf values
python3 << 'EOF'
import pandas as pd
import numpy as np

features = pd.read_csv('droplet_features.csv')
targets = pd.read_csv('droplet_targets.csv')

print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")
print(f"NaN in features: {features.isnull().sum().sum()}")
print(f"NaN in targets: {targets.isnull().sum().sum()}")
print(f"Inf in features: {np.isinf(features).sum().sum()}")
print(f"Inf in targets: {np.isinf(targets).sum().sum()}")
EOF
```

### 2. Feature Ranges
```bash
python3 << 'EOF'
import pandas as pd

features = pd.read_csv('droplet_features.csv')
print("Feature Statistics:")
print(features.describe())
EOF
```

### 3. Target Ranges
```bash
python3 << 'EOF'
import pandas as pd

targets = pd.read_csv('droplet_targets.csv')
print("Target Statistics:")
print(targets.describe())
print(f"\nDiameter range: {targets['diameter_um'].min():.2f} - {targets['diameter_um'].max():.2f} μm")
print(f"Evap time range: {targets['evaporation_time_ms'].min():.2f} - {targets['evaporation_time_ms'].max():.2f} ms")
EOF
```

---

## Files Summary

| File | Type | Rows | Columns | Purpose |
|------|------|------|---------|---------|
| `droplet_features.csv` | CSV | 44,000 | 10 | Model inputs (X) |
| `droplet_targets.csv` | CSV | 44,000 | 3 | Model targets (y) |
| `droplet_trajectories_full.csv` | CSV | 44,000 | 20+ | Complete data + metadata |
| `parcel_metadata.csv` | CSV | ~1,000 | 15 | Parcel properties |
| `01_Droplet_Evolution.png` | Image | - | - | Size evolution plots |
| `02_Temperature_Effects.png` | Image | - | - | Temperature influence |
| `03_Feature_Distributions.png` | Image | - | - | Histograms of features |

---

## Example: Loading Data for Next Steps

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load extracted data
X = pd.read_csv('droplet_features.csv')
y = pd.read_csv('droplet_targets.csv')

print(f"Training samples: {len(X)}")
print(f"Features: {list(X.columns)}")
print(f"Targets: {list(y.columns)}")

# Convert to numpy
X_numpy = X.values
y_numpy = y.values

# Ready for Step 3: Normalization
print("\n✓ Data ready for Step 3 (Normalization)")
print("  Next: Apply Min-Max scaling and split into train/val/test")
```

---

## Success Criteria

✓ All 4 CSV files generated  
✓ Each file has correct number of rows (44,000 for features/targets, ~1,000 for parcel metadata)  
✓ No NaN or Inf values  
✓ Feature ranges are reasonable (T in 700-850K, p in 4.9-5.1 MPa, etc.)  
✓ Visualizations show expected patterns (temperature gradient, diameter decay)  

**If all above met:** Proceed to Step 3 (Normalization) →

---

## Questions?

Refer to:
1. **Feature extraction details:** See inline comments in `path2_feature_extraction.py`
2. **Physics model:** Section "Phase 3" above
3. **Sandia validation:** `SANDIA_VALIDATION_GUIDE.md`
4. **ML training continuation:** Original complete guide (Steps 3-10)

---

**You're now ready to train your droplet size evolution model! 🚀**
