#!/usr/bin/env python3
"""
Generate Sandia Predictions
Creates sandia_predictions.csv if it doesn't exist
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

print("=" * 80)
print("SANDIA PREDICTIONS GENERATOR")
print("=" * 80)

# ─────────────────────────────────────────────
# 0. Model definition
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

# ─────────────────────────────────────────────
# 1. Load prepared data
# ─────────────────────────────────────────────

print("\n[1/4] Loading prepared Sandia data...")

if not os.path.exists("sandia_flame_d_prepared.csv"):
    print("❌ ERROR: sandia_flame_d_prepared.csv not found!")
    print("   Run sandia_data_mapper_FINAL.py first")
    exit()

sandia_prep = pd.read_csv("sandia_flame_d_prepared.csv")
X_sandia = sandia_prep.values

print(f"  ✓ Loaded: {len(sandia_prep)} measurements")
print(f"  ✓ Features: {X_sandia.shape[1]}")

# ─────────────────────────────────────────────
# 2. Load model and scalers
# ─────────────────────────────────────────────

print("\n[2/4] Loading model and scalers...")

if not os.path.exists("best_model.pth"):
    print("❌ ERROR: best_model.pth not found!")
    exit()

if not os.path.exists("scaler_X.pkl") or not os.path.exists("scaler_y.pkl"):
    print("❌ ERROR: Scalers not found!")
    exit()

try:
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    
    model = DropletEvolutionModel(input_size=X_sandia.shape[1], output_size=3)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    print("  ✓ Model loaded")
    print("  ✓ Scalers loaded")
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit()

# ─────────────────────────────────────────────
# 3. Generate predictions
# ─────────────────────────────────────────────

print("\n[3/4] Generating predictions...")

X_sandia_norm = scaler_X.transform(X_sandia)
X_sandia_t = torch.FloatTensor(X_sandia_norm.astype(np.float32))

with torch.no_grad():
    y_pred_norm = model(X_sandia_t).numpy()

y_pred_real = scaler_y.inverse_transform(y_pred_norm)

print(f"  ✓ Generated {len(y_pred_real)} predictions")

# ─────────────────────────────────────────────
# 4. Save predictions
# ─────────────────────────────────────────────

print("\n[4/4] Saving predictions...")

results = pd.DataFrame(
    y_pred_real,
    columns=["Diameter_pred_um", "EvapTime_pred_ms", "Emissions_pred_ppm"]
)

results.to_csv("sandia_predictions.csv", index=False)

print(f"  ✓ Saved to: sandia_predictions.csv")
print(f"  ✓ Shape: {results.shape}")

# Display sample
print(f"\nSample predictions:")
print(results.head(10))

print("\n" + "=" * 80)
print("✅ PREDICTIONS GENERATED!")
print("=" * 80)
print(f"""
✓ Predictions saved: sandia_predictions.csv
✓ Ready for comparison

NEXT: Run sandia_comparison.py
  python3 sandia_comparison.py

This will show you how your model performs on Sandia data!
""")
