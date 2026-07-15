# STEP 3: QUICK START

## Your Current Status ✅

**Path 2 (Feature Extraction):** COMPLETE
- ✅ 792 training samples generated
- ✅ 10 input features extracted
- ✅ 3 output targets created
- ✅ All visualizations working

**Files you have:**
```
✓ droplet_features.csv         (792 × 10)
✓ droplet_targets.csv          (792 × 3)
✓ droplet_trajectories_full.csv
✓ parcel_metadata.csv
✓ 3 PNG visualizations
```

---

## What is Step 3?

**Normalization** = scale all data to 0-1 range

**Why?**
- Neural networks train better on normalized data
- Prevents large-value features from dominating
- Improves convergence speed

**What it does:**
```
Original:  diameter ∈ [8.68, 100], T ∈ [800, 800], p ∈ [5e6, 5e6]
                          ↓ NORMALIZE
Normalized: diameter ∈ [0, 1], T ∈ [0, 1], p ∈ [0, 1]
```

---

## Run Step 3 NOW

```powershell
cd C:\Users\Ashish\Downloads\ShauryaMLModel
python3 step3_normalization.py
```

### What It Does

1. ✅ Loads `droplet_features.csv` and `droplet_targets.csv`
2. ✅ Scales all features to [0, 1]
3. ✅ Scales all targets to [0, 1]
4. ✅ Saves scalers for later un-normalization
5. ✅ Saves normalized data as CSV and Pickle

### What You'll Get

```
✓ scaler_X.pkl                 ← SAVE THIS! (un-normalize test data)
✓ scaler_y.pkl                 ← SAVE THIS! (un-normalize predictions)
✓ X_normalized.csv             (normalized features)
✓ y_normalized.csv             (normalized targets)
✓ X_normalized.pkl             (same, faster loading)
✓ y_normalized.pkl             (same, faster loading)
```

---

## Expected Output

```
================================================================================
STEP 3: DATA NORMALIZATION
================================================================================

[1/4] Loading extracted data...
  ✓ Features shape: (792, 10)
  ✓ Targets shape: (792, 3)

[2/4] Normalizing features...
  ✓ Features normalized to [0, 1]

[3/4] Normalizing targets...
  ✓ Targets normalized to [0, 1]

[4/4] Saving scalers and normalized data...
  ✓ Scalers saved:
    - scaler_X.pkl
    - scaler_y.pkl
  ✓ Normalized data saved as CSV:
    - X_normalized.csv
    - y_normalized.csv
  ✓ Also saved as pickle format:
    - X_normalized.pkl
    - y_normalized.pkl

[VERIFICATION] Checking normalized data quality...
  ✓ X_normalized min: 0.000000 (should be 0)
  ✓ X_normalized max: 1.000000 (should be 1)
  ✓ y_normalized min: 0.000000 (should be 0)
  ✓ y_normalized max: 1.000000 (should be 1)
  ✓ NaN in X_normalized: 0 (should be 0)
  ✓ NaN in y_normalized: 0 (should be 0)
  ✓ Inf in X_normalized: 0 (should be 0)
  ✓ Inf in y_normalized: 0 (should be 0)
  ✓ Data quality check PASSED

================================================================================
NORMALIZATION COMPLETE!
================================================================================
```

---

## After Normalization

### Files to Keep Safe

```
These 2 files are CRITICAL - save them!
  scaler_X.pkl  ← Load when evaluating on new data
  scaler_y.pkl  ← Load when un-normalizing model outputs
```

### How to Un-Normalize Later

```python
import pickle

# Load scalers
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

# Your model predicts normalized values [0, 1]
y_pred_normalized = model.predict(X_test_normalized)

# Un-normalize to real units
y_pred_real = scaler_y.inverse_transform(y_pred_normalized)

# Now in original units:
# - diameter_um: 8.68 - 100
# - evaporation_time_ms: 40.5 - 4051.6
# - emissions_proxy: 15,244 - 2,025,821
```

---

## Your Progress

```
Step 1: Understand CFD Data        ✅ COMPLETE
Step 2: Feature Extraction          ✅ COMPLETE  (792 samples)
Step 3: Normalization               → RUN NOW
Step 4: Train/Val/Test Split        Coming next
Step 5: Define Neural Network       Coming next
Step 6: Training Loop               Coming next
Step 7: Monitor & Prevent Overfitting Coming next
Step 8: Evaluate on Test Set        Coming next
Step 9: Validate on Sandia          Coming next (after model training)
Step 10: Report Results             Coming next
```

---

## Timeline

- **Step 2 (Extraction):** 2-5 minutes ✅
- **Step 3 (Normalization):** 1-2 minutes ⏰ (about to run)
- **Step 4 (Split):** 1-2 minutes
- **Step 5 (Model):** 10-20 minutes
- **Step 6 (Training):** 10-60 minutes (depends on GPU)
- **Steps 7-10:** 1-2 hours total

**Total: ~2-3 hours from start to validation**

---

## Files to Review

- **`STEP_3_NORMALIZATION.md`** ← Read this for detailed explanation
- **`step3_normalization.py`** ← Run this script
- **`EXTRACTION_COMPLETE.md`** ← Already read, reference for data insights

---

## Next: Step 4 - Train/Val/Test Split

After Step 3 completes successfully:

1. You'll have normalized data ready
2. Next: Split into training (70%), validation (15%), test (15%)
3. Then: Build neural network, train, evaluate

**But first, complete Step 3!**

---

## Action Items

- [ ] Read: `STEP_3_NORMALIZATION.md` (10 min)
- [ ] Run: `python3 step3_normalization.py` (1-2 min)
- [ ] Verify: All 6 output files created
- [ ] Check: Data quality (no NaN/Inf)
- [ ] Save: `scaler_X.pkl` and `scaler_y.pkl` (CRITICAL!)
- [ ] Ready: For Step 4

---

## Run It Now! 🚀

```powershell
python3 step3_normalization.py
```

Then proceed to Step 4 (coming soon)!
