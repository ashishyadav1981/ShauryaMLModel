# STEP 7: EVALUATE MODEL PERFORMANCE

## Overview
After training, you need to assess how well your model performs on unseen test data. This step calculates detailed metrics, generates visualizations, and prepares for final Sandia validation.

---

## What This Step Does

### Input Files
- `best_model.pth` (from Step 6 training)
- `X_test.npy`, `y_test.npy` (from Step 4 splitting)
- `scaler_X.pkl`, `scaler_y.pkl` (from Step 3 normalization)

### Output Files
- `evaluation_metrics.txt` — Detailed metrics report
- `evaluation_plots.png` — Visualizations (predicted vs actual, residuals)
- `predictions_vs_actual.csv` — Sample-by-sample predictions

---

## Running Step 7

### Command
```bash
python3 step7_evaluate.py
```

### Expected Runtime
~30 seconds

### Expected Output
```
================================================================================
STEP 7: MODEL EVALUATION
================================================================================

[1/5] Loading test data ...
  ✓ X_test shape: (119, 10)
  ✓ y_test shape: (119, 3)

[2/5] Loading scalers ...
  ✓ Scalers loaded

[3/5] Loading best model and making predictions ...
  ✓ Predictions shape: (119, 3)

[4/5] Calculating evaluation metrics ...

  NORMALISED SPACE (0-1 range):
  ─────────────────────────────
  MSE  : 0.001234
  MAE  : 0.025678
  RMSE : 0.035123
  R²   : 0.965432

  REAL SPACE (Physical Units):
  ─────────────────────────────
  Target Name                 MAE           RMSE          R²
  ──────────────────────────────────────────────────────────────────
  Diameter (μm)                5.2345       7.1234     0.9876
  Evap Time (ms)              12.3456      18.9876     0.9543
  Emissions (ppm)              2.3456       3.4567     0.8765

  PERCENTAGE ERRORS (Real Space):
  ─────────────────────────────────
  Diameter (μm)                 3.24%
  Evap Time (ms)                5.67%
  Emissions (ppm)               8.91%
```

---

## Understanding the Metrics

### 1. MSE (Mean Squared Error)
- **Formula**: MSE = mean((true - predicted)²)
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Penalizes large errors heavily
- **Example**: MSE = 0.001 (in normalised space) is good

### 2. MAE (Mean Absolute Error)
- **Formula**: MAE = mean(|true - predicted|)
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Average absolute error (easier to interpret than MSE)
- **Example**: MAE = 5.2 μm means on average predictions are off by 5.2 micrometers

### 3. RMSE (Root Mean Squared Error)
- **Formula**: RMSE = √MSE
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Same units as target, penalizes outliers
- **Example**: RMSE = 7.1 μm

### 4. R² Score (Coefficient of Determination)
- **Formula**: R² = 1 - (SS_res / SS_tot)
- **Range**: -∞ to 1 (higher is better, 1.0 is perfect)
- **Interpretation**: Fraction of variance explained by the model

#### R² Interpretation Guide
| R² Value | Quality | Action |
|----------|---------|--------|
| > 0.95   | Excellent | Publication ready |
| 0.85-0.95 | Very Good | Good for paper, maybe tune slightly |
| 0.75-0.85 | Good | Acceptable, consider improvements |
| 0.60-0.75 | Moderate | Needs work, increase model capacity |
| 0.40-0.60 | Poor | Significant improvements needed |
| < 0.40   | Very Poor | Restart with new architecture |

---

## Interpreting the Plots

### File: `evaluation_plots.png`

#### Top Row: Predicted vs Actual Scatter Plots
- **X-axis**: True values from test set
- **Y-axis**: Model predictions
- **Red dashed line**: Perfect prediction (y=x)

**What to look for:**
- ✓ Points clustered tightly around the red line → Good!
- ⚠ Scatter above/below the line → Model has systematic bias
- ✗ Points far from line → Model struggling on extreme values

#### Bottom Row: Residual Distributions
- **X-axis**: Prediction error (true - predicted)
- **Y-axis**: Frequency (histogram)
- **Red dashed line**: Zero error (perfect predictions)

**What to look for:**
- ✓ Bell curve centered at 0 → Well-calibrated model
- ⚠ Skewed toward positive/negative → Systematic bias
- ✗ Very wide distribution → High prediction variance

---

## Real Space vs Normalised Space

### Normalised Space (0-1 range)
- What the neural network actually sees
- Good for debugging model behavior
- Less intuitive for domain experts

**Example:**
```
MSE (normalised) = 0.001234
```

### Real Space (Physical Units)
- What matters for your application
- Human-interpretable results
- Good for papers and presentations

**Example:**
```
Diameter (μm): MAE = 5.2345 μm, RMSE = 7.1234 μm
Evap Time (ms): MAE = 12.3456 ms, RMSE = 18.9876 ms
Emissions (ppm): MAE = 2.3456 ppm, RMSE = 3.4567 ppm
```

---

## Percentage Error

Relative error in each target:
```
% Error = (MAE / mean_true_value) × 100%
```

**Interpretation:**
- ✓ < 5%: Excellent
- ✓ 5-10%: Good
- ⚠ 10-20%: Acceptable
- ✗ > 20%: Needs improvement

---

## Troubleshooting Poor Performance

### If R² < 0.7

**Check these in order:**

#### 1. **Training Curve Overfitting?**
   - Open `training_history.png`
   - If validation loss rises while training loss falls: **overfitting**
   - **Fix**: Increase dropout (0.2 → 0.3), reduce hidden size (128 → 64)

#### 2. **Model Too Small?**
   - Current: 10 → 128 → 64 → 32 → 3
   - **Try**: 10 → 256 → 128 → 64 → 3
   - **Re-run**: Steps 5 & 6

#### 3. **Training Time Too Short?**
   - Check `training_log.csv` — did it early stop too early?
   - **Fix**: Increase `MAX_EPOCHS` or `PATIENCE` in step6_training_loop.py

#### 4. **Data Quality Issues?**
   - Open `predictions_vs_actual.csv`
   - Look for outliers or systematic errors
   - Check feature distributions with: `python3 explore_your_data.py`

#### 5. **Learning Rate Wrong?**
   - Current: 1e-3
   - **Try**: 1e-2 or 5e-4
   - Modify `LEARNING_RATE` in step6_training_loop.py

---

## Reading the Metrics Report

### Structure of `evaluation_metrics.txt`

```
NORMALISED SPACE METRICS (0-1 range)
─────────────────────────────────────
Mean Squared Error (MSE)     : 0.001234
Mean Absolute Error (MAE)    : 0.025678
Root Mean Squared Error (RMSE): 0.035123
R² Score (mean)              : 0.965432

REAL SPACE METRICS (Physical Units)
────────────────────────────────────
Target                      MAE          RMSE         R²
Diameter (μm)               5.2345       7.1234      0.9876
Evap Time (ms)             12.3456      18.9876      0.9543
Emissions (ppm)             2.3456       3.4567      0.8765

PERCENTAGE ERRORS (Real Space)
──────────────────────────────
Target                      % Error
Diameter (μm)                3.24%
Evap Time (ms)               5.67%
Emissions (ppm)              8.91%

DATA SAMPLE STATISTICS
──────────────────────
Test set size: 119 samples
Features: 10
Targets: 3

TARGET VALUE RANGES (Real Space)
─────────────────────────────────
Target                      Min         Max          Mean
Diameter (μm)               8.68        100.00       45.23
Evap Time (ms)             40.49       4051.64     1234.56
Emissions (ppm)              0.00        10.00        5.23
```

---

## What Happens Next

### Good Performance (R² > 0.8)?
1. ✓ Your model is ready for validation
2. Prepare for **Sandia Flame D validation** (see separate guide)
3. Document results for your paper

### Moderate Performance (R² 0.6-0.8)?
1. ⚠ Model is usable but could improve
2. Try retraining with larger model/longer training
3. Still proceed to Sandia validation as is
4. Document limitations in paper

### Poor Performance (R² < 0.6)?
1. ✗ Model needs significant work
2. Return to Step 5 and try larger architecture
3. Or return to Step 2 and check data quality
4. Don't proceed to Sandia validation until improved

---

## Key Files After Step 7

| File | Purpose | Keep? |
|------|---------|-------|
| `best_model.pth` | Best trained model | ✓ YES (needed for Sandia validation) |
| `evaluation_metrics.txt` | Metrics report | ✓ YES (for paper) |
| `evaluation_plots.png` | Visual results | ✓ YES (for paper) |
| `predictions_vs_actual.csv` | Detailed predictions | ✓ YES (for analysis) |

---

## Checkpoints Before Moving to Sandia

Before proceeding to Sandia Flame D validation, verify:

- [ ] Read `evaluation_metrics.txt` and understand the metrics
- [ ] Opened `evaluation_plots.png` and visually inspected results
- [ ] R² score is > 0.65 on at least 2 of 3 targets
- [ ] No obvious systematic bias in residual plots
- [ ] Percentage errors are < 20% for main outputs
- [ ] Documented hyperparameters (learning rate, batch size, etc.)

---

## Next: Sandia Flame D Validation

Once satisfied with test performance, proceed to validate against real experimental data:

**File**: `SANDIA_VALIDATION_GUIDE.md`

This will show you how to:
1. Download Sandia Flame D benchmark data
2. Prepare it for your model
3. Compare your ML predictions against experiments
4. Generate publication-quality validation plots

---

## Summary

| Step | Status | What You Get |
|------|--------|-------------|
| 1-6 | ✓ Complete | Trained model |
| 7 | → **You are here** | Evaluation metrics + plots |
| 8 | Next | Sandia validation |
| 9 | Final | Paper-ready results |

🎉 You're nearly done! Keep going!
