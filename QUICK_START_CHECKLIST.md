# PATH 2: QUICK-START CHECKLIST
## Execute in Order

---

## ✓ PREPARATION (5 minutes)

- [ ] Copy your CFD file to working directory:
  ```bash
  cp Eulerian_1_-_eulerian1_1ls -lh Eulerian_1_-_eulerian1_1.csv.csv ~/ml_project/
  cd ~/ml_project/
  ```

- [ ] Verify file exists:
  ```bash
  ls -lh Eulerian_1_-_eulerian1_1.csv
  # Should show: -rw-r--r-- 17M Eulerian_1_...
  ```

- [ ] Install required packages (if needed):
  ```bash
  pip install pandas numpy matplotlib scipy scikit-learn
  ```

---

## ✓ PHASE 1: UNDERSTAND THE SCRIPT (10 minutes)

- [ ] Read **PATH_2_COMPLETE_GUIDE.md**
  - Understand what feature extraction means
  - Review the 4 phases: Load → Parcel → Features → Export
  
- [ ] Skim **path2_feature_extraction.py**
  - Look for section comments [1/6] through [6/6]
  - Check parcel size parameters (parcel_dx, parcel_dy, parcel_dz)

---

## ✓ PHASE 2: RUN THE EXTRACTION (2-5 minutes)

- [ ] Execute the feature extraction:
  ```bash
  python3 path2_feature_extraction.py
  ```

- [ ] Monitor console output:
  ```
  [1/6] Loading Eulerian CFD data... ✓
  [2/6] Creating virtual droplet parcels... ✓
  [3/6] Generating droplet size evolution targets... ✓
  [4/6] Preparing ML training data... ✓
  [5/6] Saving data files... ✓
  [6/6] Creating visualizations... ✓
  ```

- [ ] Wait for "EXTRACTION COMPLETE!" message

---

## ✓ PHASE 3: VERIFY OUTPUT (5 minutes)

- [ ] Check generated files:
  ```bash
  ls -lh *.csv *.png
  
  # Should show:
  # droplet_features.csv (5-10 MB)
  # droplet_targets.csv (2-3 MB)
  # droplet_trajectories_full.csv (15-20 MB)
  # parcel_metadata.csv (1 MB)
  # *.png images (3 files)
  ```

- [ ] Verify data integrity:
  ```bash
  python3 << 'EOF'
  import pandas as pd
  import numpy as np
  
  # Load and check
  X = pd.read_csv('droplet_features.csv')
  y = pd.read_csv('droplet_targets.csv')
  
  print(f"✓ Features shape: {X.shape}")
  print(f"✓ Targets shape: {y.shape}")
  print(f"✓ No NaN values: {X.isnull().sum().sum() == 0}")
  print(f"✓ No Inf values: {(~np.isfinite(X)).sum().sum() == 0}")
  
  print("\nFeature ranges:")
  print(X.describe().round(3))
  
  print("\nTarget ranges:")
  print(y.describe().round(3))
  EOF
  ```

- [ ] Preview the data:
  ```bash
  # Show first few rows
  head -3 droplet_features.csv
  head -3 droplet_targets.csv
  
  # Count rows
  wc -l droplet_features.csv    # Should be 44,001 (header + 44,000)
  ```

- [ ] Review visualizations:
  ```bash
  # Open in image viewer
  xdg-open 01_Droplet_Evolution.png      # On Linux
  # or open in your image viewer
  ```

---

## ✓ PHASE 4: UNDERSTAND THE DATA (10 minutes)

- [ ] Open **droplet_features.csv** in a text editor or Excel
  - See the 10 input features
  - Understand temperature, pressure, species ranges
  
- [ ] Open **droplet_targets.csv** in a text editor or Excel
  - See the 3 output targets
  - Understand diameter evolution and evaporation times

- [ ] Examine **parcel_metadata.csv**
  - See the ~1,000 spatial parcel locations
  - Understand field properties at each location

- [ ] Study the visualizations:
  - **01_Droplet_Evolution.png:** How diameter decreases with time
  - **02_Temperature_Effects.png:** Temperature influence on evaporation
  - **03_Feature_Distributions.png:** Feature ranges and histograms

---

## ✓ PHASE 5: SANDIA VALIDATION SETUP (15 minutes)

- [ ] Read **SANDIA_VALIDATION_GUIDE.md**
  - Understand what Sandia Flame D is
  - Learn correct download link: https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/
  - Review validation workflow

- [ ] (Optional now, required later) Download Sandia data
  - Go to: https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/
  - Download "Sandia/TUD Piloted CH4/Air Jet Flames" archive
  - Extract to local directory
  - Save for later (after model training)

---

## ✓ PHASE 6: NEXT STEPS - PROCEED TO STEP 3

You now have:
- ✓ 44,000 training samples
- ✓ 10 input features (T, p, ρ, O2, C7H16, velocity, position, time, diameter)
- ✓ 3 output targets (diameter, evaporation time, emissions)
- ✓ Data validation passed

**Next:** Proceed to **Step 3: Normalization**

```bash
# You'll need the complete ML training guide (Steps 3-10)
# For now, you have the feature extraction complete!

# Keep these files safe:
# - droplet_features.csv
# - droplet_targets.csv
# - Parcel_metadata.csv  (optional, for reference)
```

---

## TROUBLESHOOTING

### Error: "File not found: Eulerian_1_-_eulerian1_1.csv"
```bash
# Check file is in current directory
ls -la Eulerian_1_*.csv

# If not found, copy it:
cp /path/to/Eulerian_1_-_eulerian1_1.csv ./
```

### Error: "MemoryError" or slow performance
```python
# In path2_feature_extraction.py, reduce parcel count:
parcel_dx = 0.025  # Increase from 0.015 (larger parcels)
parcel_dy = 0.025
parcel_dz = 0.010
# This creates fewer, larger parcels (faster but lower resolution)
```

### Missing visualizations
```bash
# Ensure matplotlib can display
# If in headless environment, the script auto-saves to PNG anyway
ls *.png  # Should show 3 files
```

### Data looks wrong
- Check input file is 178,164 rows
- Check parcel extraction isn't finding empty regions
- Increase parcel size if few parcels created

---

## SUCCESS INDICATORS

When complete, you should have:

```
✓ 4 CSV files generated (44,000 samples each)
✓ 3 PNG visualizations created
✓ Features range: T in [777-800] K, diameter in [0.1-100] μm
✓ Targets meaningful: evaporation times 1-10,000 ms
✓ No errors in console output
```

---

## FILE ORGANIZATION (For Reference)

```
ml_project/
├── Eulerian_1_-_eulerian1_1.csv          (input)
├── path2_feature_extraction.py            (script)
├── PATH_2_COMPLETE_GUIDE.md              (documentation)
│
├── droplet_features.csv                   (OUTPUT: X for ML)
├── droplet_targets.csv                    (OUTPUT: y for ML)
├── droplet_trajectories_full.csv          (OUTPUT: full data)
├── parcel_metadata.csv                    (OUTPUT: reference)
│
├── 01_Droplet_Evolution.png               (visualization)
├── 02_Temperature_Effects.png             (visualization)
├── 03_Feature_Distributions.png           (visualization)
│
└── [Later] SANDIA_VALIDATION_GUIDE.md    (for model validation)
```

---

## ESTIMATED TIMELINE

| Step | Time | Status |
|------|------|--------|
| Preparation | 5 min | Do first |
| Read guides | 10 min | Do first |
| Run extraction | 2-5 min | DO NOW |
| Verify output | 5 min | DO NOW |
| Understand data | 10 min | DO NOW |
| Sandia setup | 15 min | Optional, do later |
| **Total** | **~50 min** | **Today** |

---

## NEXT MEETING AGENDA

After completing Path 2, discuss:
1. **Data quality:** Visualization review
2. **Feature selection:** Are these 10 features sufficient?
3. **Physics validation:** Does D² law match your CFD expectations?
4. **Step 3 readiness:** Normalization strategy
5. **Timeline:** When to train the full model

---

## QUESTIONS DURING EXECUTION?

Refer to:
1. **What is a parcel?** → PATH_2_COMPLETE_GUIDE.md, Phase 2
2. **Physics model?** → PATH_2_COMPLETE_GUIDE.md, Phase 3
3. **Feature explanation?** → PATH_2_COMPLETE_GUIDE.md, Feature table
4. **Sandia later?** → SANDIA_VALIDATION_GUIDE.md

---

## READY? 

✓ Check preparation is complete  
✓ Run: `python3 path2_feature_extraction.py`  
✓ Verify output files exist  
✓ Review visualizations  
✓ Proceed to Step 3 when ready  

**You're about to train your first ML droplet model! 🚀**

---

*Generated: June 2026*  
*For: AI-Based Liquid Drop Calculation Model Project*  
*Status: Ready for Execution*
