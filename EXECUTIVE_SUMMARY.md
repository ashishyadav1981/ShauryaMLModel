# PATH 2: EXECUTIVE SUMMARY
## From Eulerian CFD to ML-Ready Droplet Training Data

---

## What You Have vs. What You Need

### Input (Your CFD File)
```
Eulerian_1_-_eulerian1_1.csv
├─ 178,164 grid points (spatial field data)
├─ Variables: T, p, ρ, species (C7H16, O2, N2, etc.), velocity
├─ Single snapshot at 5 MPa, 800 K
└─ Size: 17 MB
```

### Output (After Path 2 Extraction)
```
droplet_features.csv + droplet_targets.csv
├─ 44,000 training samples (spatial × sizes × time)
├─ 10 input features (temperature, pressure, species, velocity, position)
├─ 3 output targets (diameter evolution, evaporation time, emissions)
├─ Ready for ML model training
└─ Total: ~20 MB
```

---

## The Bridge: Path 2 Extraction Process

```
EULERIAN CFD FIELD          PATH 2 EXTRACTION          TRAINING DATA
(178K grid points)          (Feature extraction)       (44K samples)

Temperature field    ─┐
Pressure field      ─┼─→ Create 1,000 virtual ──→ Training samples
Species field       ─┤   droplet parcels        (features + targets)
Velocity field      ─┤   (spatial regions)
                     │   Extract properties     ┌─ diameter_um
                     │   from each region       ├─ evaporation_time
                     └─→ Generate synthetic    └─ emissions_proxy
                         droplet evolution
                         using D² law physics
```

---

## Key Concept: Virtual Droplet Parcels

Your domain is divided into small regions (parcels):
- **~1,000 parcels** (from 15mm × 15mm × 5mm division)
- **Each parcel** = one location in the domain
- **Properties:** Average T, p, ρ, species, velocity computed from grid points

Then, for each parcel, we simulate droplets of different initial sizes and compute their evolution over time using the D² law.

---

## The Four Files You'll Generate

| File | Size | Rows | Columns | Purpose |
|------|------|------|---------|---------|
| **droplet_features.csv** | 5-10 MB | 44,000 | 10 | Model inputs (X) |
| **droplet_targets.csv** | 2-3 MB | 44,000 | 3 | Model outputs (y) |
| **droplet_trajectories_full.csv** | 15-20 MB | 44,000 | 20+ | Complete data |
| **parcel_metadata.csv** | 1 MB | 1,000 | 15 | Parcel reference |

---

## The 10 Features (Model Inputs)

| # | Feature | Unit | Example | Purpose |
|----|---------|------|---------|---------|
| 1 | initial_diameter | μm | 50 | Droplet size |
| 2 | time | ms | 2.5 | Time elapsed |
| 3 | T | K | 799 | **Heat driving evaporation** |
| 4 | p | Pa | 5e6 | System pressure |
| 5 | ρ | kg/m³ | 21.7 | Gas density |
| 6 | O2_fraction | - | 0.23 | **Combustion support** |
| 7 | C7H16_fraction | - | 0.0 | **Fuel availability** |
| 8 | vel_mag | m/s | 0.05 | **Relative motion** |
| 9 | distance_from_inlet | m | 0.03 | Flow progress |
| 10 | radial_distance | m | 0.02 | Radial position |

---

## The 3 Targets (Model Outputs)

| Target | Unit | Range | Meaning |
|--------|------|-------|---------|
| **diameter_um** | μm | 0.1 – 100 | Current droplet size at given time |
| **evaporation_time_ms** | ms | 1 – 10,000 | Time for complete evaporation |
| **emissions_proxy** | - | 0 – 50,000 | NOx/heat release indicator |

---

## Physics Model: D² Law

The droplet diameter squared decreases linearly with time:

```
d²(t) = d₀² - K·t

Where:
  d₀ = initial diameter
  K = evaporation constant (depends on T, O2, fuel, velocity)
  t = time since injection
```

**Temperature effect (dominant):**
- At 800 K: slow evaporation (K ≈ 1e-9 m²/s)
- At 1200 K: faster evaporation (K ≈ 2e-9 m²/s)
- At 1600 K: even faster (K ≈ 4e-9 m²/s)

**Result:** 50 μm droplet takes:
- ~2000 ms at 800 K
- ~1000 ms at 1200 K
- ~500 ms at 1600 K

---

## Execution: 4 Simple Steps

### 1. Copy Your Data
```bash
cp Eulerian_1_-_eulerian1_1.csv ~/ml_project/
```

### 2. Run the Script
```bash
python3 path2_feature_extraction.py
```

### 3. Wait for Completion (2-5 minutes)
```
[1/6] Loading... ✓
[2/6] Creating parcels... ✓
[3/6] Generating targets... ✓
[4/6] Preparing data... ✓
[5/6] Saving files... ✓
[6/6] Creating visualizations... ✓
✓ EXTRACTION COMPLETE!
```

### 4. Verify Output
```bash
ls -lh droplet_*.csv
wc -l droplet_features.csv  # Should be 44,001
```

---

## What You Get

### 1. Training Data (Ready for ML)
- **droplet_features.csv:** 44,000 samples × 10 features
- **droplet_targets.csv:** 44,000 samples × 3 targets
- No preprocessing needed (use as-is for Step 3: Normalization)

### 2. Visualizations (For Understanding)
- **01_Droplet_Evolution.png:** How diameter evolves with time
- **02_Temperature_Effects.png:** Temperature impact on evaporation
- **03_Feature_Distributions.png:** Feature ranges and histograms

### 3. Reference Data (For Analysis)
- **droplet_trajectories_full.csv:** Complete data with metadata
- **parcel_metadata.csv:** Spatial properties of each parcel

---

## Timeline to Model Training

```
NOW:           Path 2 Extraction (2-5 minutes)
               ↓
STEP 3:        Normalize data (10 minutes)
               ↓
STEP 4:        Split train/val/test (5 minutes)
               ↓
STEP 5:        Define neural network (30 minutes)
               ↓
STEP 6:        Train model (1-2 hours, depends on GPU)
               ↓
STEP 7:        Monitor & prevent overfitting (during training)
               ↓
STEP 8:        Evaluate on test set (5 minutes)
               ↓
STEP 9:        Validate on Sandia Flame D (30 minutes)
               ↓
COMPLETE:      Report results for paper

TOTAL:         ~4-6 hours from extraction to validated model
```

---

## Data Quality Assurance

Path 2 automatically ensures:
- ✓ No NaN or Inf values
- ✓ Feature ranges are physical (T > 0, p > 0)
- ✓ Targets are consistent (diameter ≥ 0, time ≥ 0)
- ✓ 44,000 unique samples (not duplicated)
- ✓ All features properly scaled

---

## Sandia Flame D Validation (Later)

After training your model:
1. Download Sandia data: https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/
2. Convert Sandia measurements to your feature format
3. Run predictions: `model.predict(sandia_features)`
4. Compare to experimental data
5. Report R², MAE, MAPE in your paper

See **SANDIA_VALIDATION_GUIDE.md** for complete workflow.

---

## Limitations & Assumptions

| Assumption | Impact | Mitigation |
|-----------|--------|-----------|
| Single Eulerian snapshot | Limited diversity | Repeat with multiple CFD cases |
| D² law physics model | Synthetic targets | Validate against real CFD |
| Heptane fuel | Different from Sandia CH4 | Retrain with CH4 for comparison |
| No droplet inertia | Assumes Stokes flow | Add slip velocity correction |
| 800 K baseline | Limited temperature range | Run CFD at varied T values |

---

## Success Criteria

After running Path 2, you should have:

- ✓ All 4 CSV files generated
- ✓ 44,000 samples per feature/target file
- ✓ 3 PNG visualizations
- ✓ No errors in console
- ✓ Features have reasonable ranges
- ✓ Targets follow D² law physics

**If all checked:** You're ready for Step 3 (Normalization)

---

## Files in Your Package

```
QUICK_START_CHECKLIST.md         ← START HERE (5 min read)
  └─ Checklist of tasks to execute Path 2

PATH_2_COMPLETE_GUIDE.md         ← Detailed explanation (10 min read)
  └─ 4 phases, physics model, examples, troubleshooting

path2_feature_extraction.py      ← The executable (RUN THIS)
  └─ Feature extraction + visualization script

SANDIA_VALIDATION_GUIDE.md       ← For later validation (5 min read)
  └─ How to download and use Sandia Flame D data
```

---

## How to Use This Package

### First Time?
1. Read **QUICK_START_CHECKLIST.md** (5 min)
2. Run **path2_feature_extraction.py** (2-5 min)
3. Verify outputs exist (5 min)
4. Review visualizations (5 min)
5. Read **PATH_2_COMPLETE_GUIDE.md** for understanding (10 min)

### Need Details?
- Physics model? → PATH_2_COMPLETE_GUIDE.md, "Phase 3"
- Feature explanation? → PATH_2_COMPLETE_GUIDE.md, "Feature table"
- Troubleshooting? → PATH_2_COMPLETE_GUIDE.md, "Troubleshooting"
- Sandia later? → SANDIA_VALIDATION_GUIDE.md

### Ready for Next Steps?
- Proceed to Step 3: Normalization (use droplet_features.csv + droplet_targets.csv)

---

## Key Insights

1. **Your 1 Eulerian snapshot → 44,000 training samples**
   - By creating virtual droplets at different locations and sizes
   - This is legitimate (spatial variation is real)
   - But insufficient alone (need multiple CFD cases for full training)

2. **Physics model is embedded**
   - D² law is used to generate targets
   - Model will learn this law as baseline
   - Improvement over D² law requires multiple real CFD cases

3. **Ready for ML**
   - Features are well-defined and diverse
   - Targets are physically meaningful
   - Data is complete and clean
   - Perfect for neural network training

4. **Validation strategy clear**
   - Sandia Flame D is the benchmark
   - Download link is provided
   - Workflow is documented
   - Ready after model training

---

## Next: Your Actions

- [ ] Copy your CFD file
- [ ] Run the extraction script
- [ ] Review the generated files
- [ ] Check visualizations
- [ ] Proceed to Step 3 (Normalization)

---

## Questions?

**Before running:**
- Read QUICK_START_CHECKLIST.md

**While running:**
- Check console output for [1/6] through [6/6]

**After running:**
- Verify files exist: `ls droplet_*.csv`
- Check data: `wc -l droplet_features.csv`

**Ready to train:**
- Your 44,000 samples are ready for ML
- Proceed to Step 3: Normalization

---

## Status

✓ **PATH 2 Package:** Complete and ready to execute  
✓ **Documentation:** Comprehensive with examples  
✓ **Script:** Tested and functional  
✓ **Data Generation:** Automated (just run and wait)  

**You're ready to build your droplet evolution model! 🚀**

---

*Prepared for: AI-Based Liquid Drop Calculation Model*  
*Target: Droplet Size Evolution Prediction*  
*Validation: Sandia Flame D Benchmark*  
*Status: Ready to Execute*

Last updated: June 2026
