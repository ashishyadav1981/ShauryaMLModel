# STEP 4: TRAIN/VALIDATION/TEST SPLIT
## Prepare Data for ML Training Pipeline

---

## Overview

**Data splitting** divides your dataset into three parts:

- **Training set (70%):** Used to train the neural network
- **Validation set (15%):** Used to monitor overfitting during training
- **Test set (15%):** Used for final evaluation (never seen by model during training)

This prevents **data leakage** and ensures your model generalizes to new data.

---

## Why Three Sets?

### Training Set (70%)
- Model learns from this data
- Weights are updated based on training loss
- Model should achieve low loss here

### Validation Set (15%)
- Model doesn't learn from this data (no weight updates)
- Used to check for overfitting during training
- Model should achieve similar loss as training
- **If validation loss > training loss = overfitting!**

### Test Set (15%)
- Final evaluation of model performance
- Never seen by model or training loop
- Represents real-world performance
- Use this for reporting final metrics (MAE, RMSE, R²)

---

## How to Split

### Your Data
```
Total samples: 792
Training: 70% = 554 samples
Validation: 15% = 119 samples
Test: 15% = 119 samples
```

### Code: Manual Split

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Load normalized data
X = np.load('X_normalized.pkl', allow_pickle=True)
y = np.load('y_normalized.pkl', allow_pickle=True)

# Split 1: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split 2: Split temp into 50% val, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Train: {len(X_train)} samples")  # 554
print(f"Val:   {len(X_val)} samples")    # 119
print(f"Test:  {len(X_test)} samples")   # 119
```

---

## Step 4A: Create Training/Validation/Test Sets

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

print("\n" + "="*80)
print("STEP 4: TRAIN/VALIDATION/TEST SPLIT")
print("="*80)

# ============================================================================
# Load Normalized Data
# ============================================================================
print("\n[1/4] Loading normalized data...")

# Option 1: Load from pickle (faster)
X_normalized = np.load('X_normalized.pkl', allow_pickle=True)
y_normalized = np.load('y_normalized.pkl', allow_pickle=True)

# Option 2: Load from CSV
# X_normalized = pd.read_csv('X_normalized.csv').values
# y_normalized = pd.read_csv('y_normalized.csv').values

print(f"  ✓ Loaded X: {X_normalized.shape}")
print(f"  ✓ Loaded y: {y_normalized.shape}")

# ============================================================================
# Split into Train/Val/Test
# ============================================================================
print("\n[2/4] Splitting data (70/15/15)...")

# Split 1: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X_normalized, y_normalized,
    test_size=0.30,
    random_state=42  # For reproducibility
)

# Split 2: Split temp into 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print(f"  ✓ Train: {len(X_train)} samples ({len(X_train)/len(X_normalized)*100:.1f}%)")
print(f"  ✓ Val:   {len(X_val)} samples ({len(X_val)/len(X_normalized)*100:.1f}%)")
print(f"  ✓ Test:  {len(X_test)} samples ({len(X_test)/len(X_normalized)*100:.1f}%)")
print(f"  ✓ Total: {len(X_train) + len(X_val) + len(X_test)} samples")

# ============================================================================
# Verify No Data Leakage
# ============================================================================
print("\n[3/4] Verifying no data leakage...")

train_indices = set(range(len(X_train)))
val_indices = set(range(len(X_train), len(X_train) + len(X_val)))
test_indices = set(range(len(X_train) + len(X_val), len(X_normalized)))

# Check no overlap
assert len(train_indices & val_indices) == 0, "Train/Val overlap!"
assert len(train_indices & test_indices) == 0, "Train/Test overlap!"
assert len(val_indices & test_indices) == 0, "Val/Test overlap!"

print(f"  ✓ No overlap between sets")
print(f"  ✓ All {len(X_normalized)} samples assigned exactly once")

# ============================================================================
# Save Train/Val/Test Data
# ============================================================================
print("\n[4/4] Saving train/val/test data...")

# Save as numpy arrays (fastest for PyTorch)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("  ✓ Saved as numpy format (.npy):")
print("    - X_train.npy, y_train.npy")
print("    - X_val.npy, y_val.npy")
print("    - X_test.npy, y_test.npy")

# Also save as pickle
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_val, open('X_val.pkl', 'wb'))
pickle.dump(y_val, open('y_val.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))

print("  ✓ Also saved as pickle format (.pkl)")

# Also save as CSV (for inspection)
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_val).to_csv('X_val.csv', index=False)
pd.DataFrame(y_val).to_csv('y_val.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("  ✓ Also saved as CSV format (.csv) for inspection")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("DATA SPLIT COMPLETE!")
print("="*80)

print(f"""
✓ Data successfully split into train/val/test!

SPLIT SUMMARY:
  Train: {len(X_train)} samples ({len(X_train)/len(X_normalized)*100:.1f}%)
  Val:   {len(X_val)} samples ({len(X_val)/len(X_normalized)*100:.1f}%)
  Test:  {len(X_test)} samples ({len(X_test)/len(X_normalized)*100:.1f}%)
  ─────────────────
  Total: {len(X_normalized)} samples (100%)

DATA SHAPES:
  X_train: {X_train.shape}
  y_train: {y_train.shape}
  X_val:   {X_val.shape}
  y_val:   {y_val.shape}
  X_test:  {X_test.shape}
  y_test:  {y_test.shape}

FEATURE STATISTICS (Train Set):
  Mean: {X_train.mean(axis=0)}
  Std:  {X_train.std(axis=0)}
  Min:  {X_train.min(axis=0)}
  Max:  {X_train.max(axis=0)}

TARGET STATISTICS (Train Set):
  Mean: {y_train.mean(axis=0)}
  Std:  {y_train.std(axis=0)}
  Min:  {y_train.min(axis=0)}
  Max:  {y_train.max(axis=0)}

FILES CREATED:
  [Numpy format - fastest for PyTorch]
    ✓ X_train.npy, y_train.npy
    ✓ X_val.npy, y_val.npy
    ✓ X_test.npy, y_test.npy
  
  [Pickle format - faster than CSV]
    ✓ X_train.pkl, y_train.pkl
    ✓ X_val.pkl, y_val.pkl
    ✓ X_test.pkl, y_test.pkl
  
  [CSV format - human readable]
    ✓ X_train.csv, y_train.csv
    ✓ X_val.csv, y_val.csv
    ✓ X_test.csv, y_test.csv

HOW TO USE:
  
  # Load for PyTorch training (recommended):
  X_train = np.load('X_train.npy')
  y_train = np.load('y_train.npy')
  X_val = np.load('X_val.npy')
  y_val = np.load('y_val.npy')
  X_test = np.load('X_test.npy')
  y_test = np.load('y_test.npy')
  
  # Or load as pandas DataFrames:
  X_train = pd.read_csv('X_train.csv')
  y_train = pd.read_csv('y_train.csv')

IMPORTANT NOTES:
  
  1. Train/Val/Test sets are independent
     - No sample appears in multiple sets
     - Prevents data leakage
  
  2. Use correct set for each phase:
     - Training: X_train, y_train
     - Monitoring: X_val, y_val
     - Final evaluation: X_test, y_test
  
  3. Never train on test set!
     - Test set is for final evaluation only
     - Use validation set to monitor during training
  
  4. Random state = 42
     - Results are reproducible
     - Same split every time you run

NEXT STEPS:
  1. Review split statistics above
  2. Proceed to Step 5: Define Neural Network
  3. Implement training loop (Step 6)
  4. Monitor and prevent overfitting (Step 7)
  5. Evaluate on test set (Step 8)
  6. Validate against Sandia data (Step 9)
""")

print("="*80 + "\n")
print("To continue to Step 5, run:\n  python3 step5_neural_network.py\n")
```

---

## Step 4B: Load the Split Data

```python
import numpy as np
import pandas as pd

# Load split data (numpy format - fastest)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(f"Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
print(f"Val:   {X_val.shape[0]} samples × {X_val.shape[1]} features")
print(f"Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
```

---

## Step 4C: Convert to PyTorch Format (Optional)

If you're using PyTorch for neural networks:

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders (for batch training)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")
```

---

## Important: Data Leakage Prevention

### ❌ WRONG - Creates Data Leakage
```python
# DON'T normalize after splitting!
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)  # Scaler sees train data
X_test_norm = scaler.transform(X_test)        # But leaks train statistics!
```

### ✅ RIGHT - Prevents Data Leakage
```python
# Normalize BEFORE splitting!
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)              # Scaler learns from all data
X_train = X_norm[:554]                        # Then split
X_val = X_norm[554:673]
X_test = X_norm[673:]
```

*(You already did this correctly in Step 3!)*

---

## Visual: Data Flow

```
Raw CFD Data (178K grid points)
        ↓ [Step 2: Extract Features]
droplet_features.csv + droplet_targets.csv (792 samples)
        ↓ [Step 3: Normalize]
X_normalized.pkl + y_normalized.pkl (792 samples, scaled 0-1)
        ↓ [Step 4: Split] ← YOU ARE HERE
┌───────┴────────┬────────────┬──────────────┐
↓                ↓            ↓              ↓
X_train        X_val        X_test      (Features)
y_train        y_val        y_test      (Targets)
554 samples    119 samples  119 samples
        ↓ [Step 5: Neural Network]
    Define model architecture
        ↓ [Step 6: Training Loop]
    Train on (X_train, y_train)
    Monitor with (X_val, y_val)
        ↓ [Step 8: Evaluation]
    Evaluate on X_test, y_test
        ↓ [Step 9: Sandia Validation]
    Compare against Sandia Flame D
```

---

## Summary

| Set | Samples | Purpose | When to Use |
|-----|---------|---------|------------|
| **Train** | 554 (70%) | Model learns | During training loop |
| **Validation** | 119 (15%) | Check overfitting | Every epoch during training |
| **Test** | 119 (15%) | Final evaluation | Only at the very end |

---

## Next Steps

Once Step 4 completes:
1. ✅ You have train/val/test sets
2. ✅ No data leakage
3. → Proceed to **Step 5: Define Neural Network**

---

## Key Files for Step 5

```python
# Step 5 will use:
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
```

No need to load test set yet - that's for final evaluation!

