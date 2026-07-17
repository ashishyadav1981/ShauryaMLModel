#!/usr/bin/env python3
"""
STEP 7: EVALUATE MODEL PERFORMANCE
Evaluates the best trained model on test set and generates detailed metrics.

Requires:
  - best_model.pth            (from Step 6)
  - X_test.npy, y_test.npy    (from Step 4)
  - scaler_X.pkl, scaler_y.pkl (from Step 3)

Outputs:
  - evaluation_metrics.txt    ← detailed metrics report
  - evaluation_plots.png      ← comprehensive visualizations
  - predictions_vs_actual.csv ← predictions for inspection
"""

import os
import numpy as np
import pickle
import csv
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 0.  Reproduce the model class
# ─────────────────────────────────────────────
class DropletEvolutionModel(nn.Module):
    def __init__(self, input_size=10, output_size=3):
        super(DropletEvolutionModel, self).__init__()
        self.fc1      = nn.Linear(input_size, 128)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2      = nn.Linear(128, 64)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3      = nn.Linear(64, 32)
        self.relu3    = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4      = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        return self.fc4(x)


print("=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

# ─────────────────────────────────────────────
# 1.  Load test data
# ─────────────────────────────────────────────
print("\n[1/5] Loading test data ...")

def load_npy(name):
    path = f"{name}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}\nRun step4_data_split.py first.")
    return np.load(path)

X_test = load_npy("X_test")
y_test = load_npy("y_test")

print(f"  ✓ X_test shape: {X_test.shape}")
print(f"  ✓ y_test shape: {y_test.shape}")

# ─────────────────────────────────────────────
# 2.  Load scalers (for un-normalisation)
# ─────────────────────────────────────────────
print("\n[2/5] Loading scalers ...")

if not os.path.exists("scaler_X.pkl") or not os.path.exists("scaler_y.pkl"):
    print("  ⚠  Warning: Scalers not found. Working with normalised metrics only.")
    scaler_X = None
    scaler_y = None
else:
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    print("  ✓ Scalers loaded")

# ─────────────────────────────────────────────
# 3.  Load best model and make predictions
# ─────────────────────────────────────────────
print("\n[3/5] Loading best model and making predictions ...")

if not os.path.exists("best_model.pth"):
    raise FileNotFoundError("best_model.pth not found. Run step6_training_loop.py first.")

model = DropletEvolutionModel(input_size=X_test.shape[1],
                               output_size=y_test.shape[1])
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

X_test_t = torch.FloatTensor(X_test.astype(np.float32))
y_test_t = torch.FloatTensor(y_test.astype(np.float32))

with torch.no_grad():
    y_pred_norm = model(X_test_t).numpy()

print(f"  ✓ Predictions shape: {y_pred_norm.shape}")

# ─────────────────────────────────────────────
# 4.  Calculate metrics (normalised & real)
# ─────────────────────────────────────────────
print("\n[4/5] Calculating evaluation metrics ...")

# Normalised space
loss_fn = nn.MSELoss()
mse_norm  = loss_fn(torch.FloatTensor(y_pred_norm), y_test_t).item()
mae_norm  = mean_absolute_error(y_test, y_pred_norm)
rmse_norm = np.sqrt(mean_squared_error(y_test, y_pred_norm))
r2_norm   = r2_score(y_test, y_pred_norm, multioutput='raw_values')

print(f"\n  NORMALISED SPACE (0-1 range):")
print(f"  ─────────────────────────────")
print(f"  MSE  : {mse_norm:.6f}")
print(f"  MAE  : {mae_norm:.6f}")
print(f"  RMSE : {rmse_norm:.6f}")
print(f"  R²   : {r2_norm.mean():.6f}")

# Real space (un-normalised)
if scaler_y is not None:
    y_test_real = scaler_y.inverse_transform(y_test)
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)

    target_names = ["Diameter (μm)", "Evap Time (ms)", "Emissions (ppm)"]
    mae_real  = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2_real   = r2_score(y_test_real, y_pred_real, multioutput='raw_values')

    print(f"\n  REAL SPACE (Physical Units):")
    print(f"  ─────────────────────────────")
    print(f"  Target Name                 MAE           RMSE          R²")
    print(f"  " + "─" * 70)
    for i, name in enumerate(target_names):
        mae_i   = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
        rmse_i  = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
        r2_i    = r2_score(y_test_real[:, i], y_pred_real[:, i])

        print(f"  {name:<25} {mae_i:>13.4f}  {rmse_i:>13.4f}  {r2_i:>7.4f}")

    # Percentage errors
    print(f"\n  PERCENTAGE ERRORS (Real Space):")
    print(f"  ─────────────────────────────────")
    for i, name in enumerate(target_names):
        true_mean = np.mean(np.abs(y_test_real[:, i]))
        mae_i = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
        pct_error = (mae_i / true_mean * 100) if true_mean > 0 else 0
        print(f"  {name:<25} {pct_error:>6.2f}%")

# ─────────────────────────────────────────────
# 5.  Save metrics report
# ─────────────────────────────────────────────
print("\n[5/5] Generating evaluation report ...")

report_lines = [
    "=" * 80,
    "STEP 7: MODEL EVALUATION REPORT",
    "=" * 80,
    "",
    "NORMALISED SPACE METRICS (0-1 range)",
    "─" * 80,
    f"Mean Squared Error (MSE)     : {mse_norm:.6f}",
    f"Mean Absolute Error (MAE)    : {mae_norm:.6f}",
    f"Root Mean Squared Error (RMSE): {rmse_norm:.6f}",
    f"R² Score (mean)              : {r2_norm.mean():.6f}",
    "",
]

if scaler_y is not None:
    report_lines.extend([
        "REAL SPACE METRICS (Physical Units)",
        "─" * 80,
        f"{'Target':<30} {'MAE':<15} {'RMSE':<15} {'R²':<10}",
        "─" * 80,
    ])

    target_names = ["Diameter (μm)", "Evap Time (ms)", "Emissions (ppm)"]
    for i, name in enumerate(target_names):
        mae_i   = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
        rmse_i  = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
        r2_i    = r2_score(y_test_real[:, i], y_pred_real[:, i])
        report_lines.append(
            f"{name:<30} {mae_i:<15.4f} {rmse_i:<15.4f} {r2_i:<10.4f}"
        )

    report_lines.extend([
        "",
        "PERCENTAGE ERRORS (Real Space)",
        "─" * 80,
        f"{'Target':<30} {'% Error':<15}",
        "─" * 80,
    ])

    for i, name in enumerate(target_names):
        true_mean = np.mean(np.abs(y_test_real[:, i]))
        mae_i = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
        pct_error = (mae_i / true_mean * 100) if true_mean > 0 else 0
        report_lines.append(f"{name:<30} {pct_error:<15.2f}%")

report_lines.extend([
    "",
    "DATA SAMPLE STATISTICS",
    "─" * 80,
    f"Test set size: {len(y_test)} samples",
    f"Features: {X_test.shape[1]}",
    f"Targets: {y_test.shape[1]}",
    "",
])

if scaler_y is not None:
    report_lines.extend([
        "TARGET VALUE RANGES (Real Space)",
        "─" * 80,
        f"{'Target':<30} {'Min':<15} {'Max':<15} {'Mean':<15}",
        "─" * 80,
    ])
    for i, name in enumerate(target_names):
        min_val = y_test_real[:, i].min()
        max_val = y_test_real[:, i].max()
        mean_val = y_test_real[:, i].mean()
        report_lines.append(
            f"{name:<30} {min_val:<15.4f} {max_val:<15.4f} {mean_val:<15.4f}"
        )

report_lines.extend([
    "",
    "INTERPRETATION GUIDE",
    "─" * 80,
    "✓ R² close to 1.0  : Model explains 100% of variance (excellent)",
    "✓ R² > 0.8         : Model is good",
    "⚠ R² 0.5-0.8       : Model is moderate",
    "✗ R² < 0.5         : Model needs improvement",
    "",
    "NEXT STEPS",
    "─" * 80,
    "1. Check evaluation_plots.png for visual inspection",
    "2. If R² < 0.7: Re-train with larger model or more epochs",
    "3. If overfitting visible in training_history.png: increase dropout",
    "4. Once satisfied, prepare for Sandia Flame D validation",
    "=" * 80,
])

report_text = "\n".join(report_lines)
print(report_text)

with open("evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("\n  ✓ Report saved: evaluation_metrics.txt")

# ─────────────────────────────────────────────
# 6.  Create evaluation plots
# ─────────────────────────────────────────────
print("\n  Generating evaluation plots ...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if scaler_y is not None:
        # Use real space for plotting
        y_true = y_test_real
        y_pred = y_pred_real
        target_names = ["Diameter (μm)", "Evap Time (ms)", "Emissions (ppm)"]
        use_real_space = True
    else:
        y_true = y_test
        y_pred = y_pred_norm
        target_names = ["Output 0", "Output 1", "Output 2"]
        use_real_space = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Predicted vs Actual scatter plots
    for i, name in enumerate(target_names):
        ax = axes[0, i]

        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50)

        # Perfect prediction line
        min_val, max_val = y_true[:, i].min(), y_true[:, i].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect prediction')

        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name}\n(Pred vs Actual)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Residual plots (error distribution)
    for i, name in enumerate(target_names):
        ax = axes[1, i]

        residuals = y_true[:, i] - y_pred[:, i]
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))

        ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\n(MAE={mae:.4f}, RMSE={rmse:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Plots saved: evaluation_plots.png")

except Exception as e:
    print(f"  ⚠  Could not create plots: {e}")

# ─────────────────────────────────────────────
# 7.  Save predictions to CSV
# ─────────────────────────────────────────────
print("  Saving predictions to CSV ...")

csv_rows = [
    ["sample_id", "true_output_0", "true_output_1", "true_output_2",
     "pred_output_0", "pred_output_1", "pred_output_2",
     "error_0", "error_1", "error_2"]
]

if scaler_y is not None:
    y_true_export = y_test_real
    y_pred_export = y_pred_real
else:
    y_true_export = y_test
    y_pred_export = y_pred_norm

for i in range(len(y_test)):
    error = y_true_export[i] - y_pred_export[i]
    row = [i] + list(y_true_export[i]) + list(y_pred_export[i]) + list(error)
    csv_rows.append(row)

with open("predictions_vs_actual.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("  ✓ Predictions saved: predictions_vs_actual.csv")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print(f"""
  Files saved:
    ✓ evaluation_metrics.txt      ← detailed metrics report
    ✓ evaluation_plots.png        ← predicted vs actual + residuals
    ✓ predictions_vs_actual.csv   ← sample-by-sample predictions

  Key Metrics:
    MAE (normalised)   : {mae_norm:.6f}
    RMSE (normalised)  : {rmse_norm:.6f}
    R² (normalised)    : {r2_norm.mean():.6f}

NEXT STEPS:
  1. Review evaluation_metrics.txt — check overall quality
  2. Open evaluation_plots.png — visual inspection
  3. If satisfied with performance:
     → Prepare for Sandia Flame D validation (see SANDIA_VALIDATION_GUIDE.md)
  4. Document hyperparameters and results for your paper
""")

print("=" * 80)
