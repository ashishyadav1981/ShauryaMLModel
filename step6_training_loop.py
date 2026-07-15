#!/usr/bin/env python3
"""
STEP 6: TRAINING LOOP
Trains the DropletEvolutionModel on the prepared data.

Requires:
  - X_train.npy, y_train.npy
  - X_val.npy,   y_val.npy
  - X_test.npy,  y_test.npy
  - model_initial_weights.pth  (from Step 5)
  - scaler_y.pkl               (from Step 3, for un-normalization)

Outputs:
  - best_model.pth             ← best checkpoint (by val loss)
  - training_history.png       ← train vs val loss curve
  - training_log.csv           ← per-epoch metrics
"""

import os
import numpy as np
import pickle
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# 0.  Reproduce the model class from Step 5
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
# 1.  Configuration
# ─────────────────────────────────────────────
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5   # L2 regularisation
BATCH_SIZE     = 32
MAX_EPOCHS     = 300
PATIENCE       = 30     # early-stopping patience
PRINT_EVERY    = 10     # print progress every N epochs

print("=" * 70)
print("STEP 6: TRAINING LOOP")
print("=" * 70)

# ─────────────────────────────────────────────
# 2.  Load split data
# ─────────────────────────────────────────────
print("\n[1/6] Loading train/val/test data ...")

def load_npy(name):
    path = f"{name}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Make sure you ran step4_data_split.py first."
        )
    return np.load(path)

X_train = load_npy("X_train")
y_train = load_npy("y_train")
X_val   = load_npy("X_val")
y_val   = load_npy("y_val")
X_test  = load_npy("X_test")
y_test  = load_npy("y_test")

print(f"  ✓ Train : {X_train.shape[0]} samples")
print(f"  ✓ Val   : {X_val.shape[0]} samples")
print(f"  ✓ Test  : {X_test.shape[0]} samples")
print(f"  ✓ Features: {X_train.shape[1]}   Targets: {y_train.shape[1]}")

# ─────────────────────────────────────────────
# 3.  Build PyTorch DataLoader
# ─────────────────────────────────────────────
print("\n[2/6] Building data loaders ...")

def to_tensor(arr):
    return torch.FloatTensor(arr.astype(np.float32))

train_ds     = TensorDataset(to_tensor(X_train), to_tensor(y_train))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

X_val_t  = to_tensor(X_val);  y_val_t  = to_tensor(y_val)
X_test_t = to_tensor(X_test); y_test_t = to_tensor(y_test)

print(f"  ✓ Batch size : {BATCH_SIZE}")
print(f"  ✓ Batches/epoch: {len(train_loader)}")

# ─────────────────────────────────────────────
# 4.  Initialise model, optimiser, loss
# ─────────────────────────────────────────────
print("\n[3/6] Initialising model ...")

model = DropletEvolutionModel(input_size=X_train.shape[1],
                               output_size=y_train.shape[1])

# Load initial weights from Step 5 if available
if os.path.exists("model_initial_weights.pth"):
    model.load_state_dict(torch.load("model_initial_weights.pth"))
    print("  ✓ Loaded initial weights from model_initial_weights.pth")
else:
    print("  ℹ  No initial weights file found — using random initialisation")

optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)

# Learning-rate scheduler: reduce LR when val loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

loss_fn = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ✓ Trainable parameters : {total_params:,}")
print(f"  ✓ Optimiser            : Adam  (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"  ✓ Loss function        : MSELoss")

# ─────────────────────────────────────────────
# 5.  Training loop
# ─────────────────────────────────────────────
print(f"\n[4/6] Training for up to {MAX_EPOCHS} epochs (patience={PATIENCE}) ...")
print("-" * 70)
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}  {'Status'}")
print("-" * 70)

train_losses = []
val_losses   = []
log_rows     = [["epoch", "train_loss", "val_loss", "lr"]]

best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Training phase ──
    model.train()
    epoch_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss        = loss_fn(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # ── Validation phase ──
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss  = loss_fn(val_preds, y_val_t).item()
    val_losses.append(val_loss)

    # ── LR scheduler ──
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)

    # ── Logging ──
    log_rows.append([epoch, epoch_train_loss, val_loss, current_lr])

    # ── Early stopping & checkpointing ──
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        status = "✓ saved"
    else:
        patience_counter += 1
        status = f"patience {patience_counter}/{PATIENCE}"

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"{epoch:>6}  {epoch_train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{current_lr:>10.2e}  {status}")

    if patience_counter >= PATIENCE:
        print(f"\n  ⚠  Early stopping triggered at epoch {epoch}")
        break

print("-" * 70)
print(f"\n  ✓ Best validation loss : {best_val_loss:.6f}")
print(f"  ✓ Model saved to       : best_model.pth")

# ─────────────────────────────────────────────
# 6.  Save training log CSV
# ─────────────────────────────────────────────
with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(log_rows)
print("  ✓ Training log saved   : training_log.csv")

# ─────────────────────────────────────────────
# 7.  Plot training history
# ─────────────────────────────────────────────
print("\n[5/6] Plotting training history ...")

try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 – full loss curves
    axes[0].plot(epochs_range, train_losses, label='Train Loss',
                 color='steelblue', linewidth=2)
    axes[0].plot(epochs_range, val_losses,   label='Val Loss',
                 color='coral',     linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 – log scale (easier to see convergence)
    axes[1].semilogy(epochs_range, train_losses, label='Train Loss',
                     color='steelblue', linewidth=2)
    axes[1].semilogy(epochs_range, val_losses,   label='Val Loss',
                     color='coral',     linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss (log scale)')
    axes[1].set_title('Training vs Validation Loss (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot saved : training_history.png")

except Exception as e:
    print(f"  ⚠  Could not create plot: {e}")
    print("     (Training still completed successfully)")

# ─────────────────────────────────────────────
# 8.  Quick evaluation on test set
# ─────────────────────────────────────────────
print("\n[6/6] Quick evaluation on test set ...")

model_best = DropletEvolutionModel(input_size=X_train.shape[1],
                                    output_size=y_train.shape[1])
model_best.load_state_dict(torch.load("best_model.pth"))
model_best.eval()

with torch.no_grad():
    test_preds_norm = model_best(X_test_t).numpy()

# Try to load scaler_y for un-normalised metrics
if os.path.exists("scaler_y.pkl"):
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    test_preds_real = scaler_y.inverse_transform(test_preds_norm)
    y_test_real     = scaler_y.inverse_transform(y_test)

    target_names = ["Diameter (μm)", "Evap Time (ms)", "Emissions (ppm)"]
    print(f"\n  {'Target':<22}  {'MAE':>12}  {'RMSE':>12}")
    print(f"  {'-'*50}")
    for i, name in enumerate(target_names):
        mae  = np.mean(np.abs(y_test_real[:, i] - test_preds_real[:, i]))
        rmse = np.sqrt(np.mean((y_test_real[:, i] - test_preds_real[:, i])**2))
        print(f"  {name:<22}  {mae:>12.4f}  {rmse:>12.4f}")
else:
    # Normalised space metrics
    test_loss_norm = loss_fn(torch.FloatTensor(test_preds_norm),
                             y_test_t).item()
    print(f"  ✓ Test MSE (normalised): {test_loss_norm:.6f}")
    print("  (scaler_y.pkl not found — showing normalised metrics)")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"""
  Files saved:
    ✓ best_model.pth        ← best checkpoint (by validation loss)
    ✓ training_history.png  ← loss curves
    ✓ training_log.csv      ← per-epoch metrics

  Best validation loss : {best_val_loss:.6f}
  Total epochs trained : {len(train_losses)}

NEXT STEPS:
  1. Open training_history.png — check for overfitting
     - Good: both curves go down together
     - Overfitting: val loss rises while train loss keeps falling
  2. If overfitting: increase dropout (0.2→0.3) or reduce hidden size
  3. If underfitting: increase hidden size (128→256) or add more layers
  4. To continue: python3 step7_evaluate.py
""")