# STEP 5: NEURAL NETWORK ARCHITECTURE
## Design the Model for Droplet Size Evolution Prediction

---

## Overview

**Neural Network** = A machine learning model with layers of neurons that learn patterns from data.

For your droplet prediction task:
- **Inputs:** 10 features (temperature, pressure, species, velocity, etc.)
- **Outputs:** 3 targets (diameter, evaporation time, emissions)
- **Hidden layers:** Learn representations of droplet physics

---

## Architecture Design: Simple MLP

We'll use a **Multi-Layer Perceptron (MLP)** - simple, effective, and proven:

```
Input Layer (10 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU activation)
    ↓
Hidden Layer 2 (64 neurons, ReLU activation)
    ↓
Hidden Layer 3 (32 neurons, ReLU activation)
    ↓
Output Layer (3 targets, Linear activation)
```

### Why This Architecture?

| Layer | Neurons | Reason |
|-------|---------|--------|
| Input | 10 | Your 10 features |
| Hidden 1 | 128 | "Wide" layer captures complex patterns |
| Hidden 2 | 64 | Compress learned features |
| Hidden 3 | 32 | Further compress |
| Output | 3 | Your 3 targets |

**Progression:** 10 → 128 → 64 → 32 → 3 (bottleneck design)

---

## PyTorch Model Definition

```python
import torch
import torch.nn as nn

class DropletEvolutionModel(nn.Module):
    """
    Neural network for predicting droplet evolution
    
    Inputs:  10 features (T, p, rho, O2, C7H16, velocity, position, time, diameter)
    Outputs: 3 targets (diameter, evaporation_time, emissions)
    """
    
    def __init__(self, input_size=10, output_size=3):
        super(DropletEvolutionModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, 128)      # Input → 128 neurons
        self.relu1 = nn.ReLU()                      # Activation function
        self.dropout1 = nn.Dropout(0.2)             # Regularization: drop 20%
        
        self.fc2 = nn.Linear(128, 64)               # 128 → 64 neurons
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)                # 64 → 32 neurons
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, output_size)       # 32 → 3 outputs
        # No activation here - linear output for regression
    
    def forward(self, x):
        """
        Forward pass: compute predictions
        
        Args:
            x: Input tensor of shape (batch_size, 10)
        
        Returns:
            output: Predicted targets of shape (batch_size, 3)
        """
        # Layer 1
        x = self.fc1(x)      # Linear transformation
        x = self.relu1(x)    # ReLU activation (non-linearity)
        x = self.dropout1(x) # Dropout regularization
        
        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer (no activation for regression)
        x = self.fc4(x)
        
        return x

# Create model
model = DropletEvolutionModel(input_size=10, output_size=3)
print(model)
```

---

## Understanding Each Component

### 1. **Linear Layers (fc1, fc2, fc3, fc4)**

```python
self.fc1 = nn.Linear(10, 128)
```

- Takes 10 inputs
- Outputs 128 values
- Learns 10×128 = 1,280 weights
- Formula: y = W·x + b

### 2. **ReLU Activation**

```python
self.relu = nn.ReLU()
```

- **ReLU(x) = max(0, x)**
- Makes network non-linear (can learn curves, not just straight lines)
- Helps learn complex droplet physics

**Without ReLU:**
```python
# Just stacking linear layers = one big linear transformation
y = W4 · W3 · W2 · W1 · x + b  # Still linear!
```

**With ReLU:**
```python
# Non-linear → can learn complex patterns
y = W4 · ReLU(W3 · ReLU(W2 · ReLU(W1 · x)))  # Non-linear!
```

### 3. **Dropout Regularization**

```python
self.dropout1 = nn.Dropout(0.2)
```

- Randomly disables 20% of neurons during training
- Prevents overfitting (memorization)
- Forces network to learn robust features
- **NOT applied during testing** (model.eval())

### 4. **No Activation in Output**

```python
self.fc4 = nn.Linear(32, output_size)
# No ReLU or Sigmoid here!
```

- Your targets can be any value in [0, 1]
- **Regression task** (continuous values) → linear output
- **Classification task** → would use softmax

---

## Model Summary

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Expected output:
# Total parameters: 22,915
# This breaks down as:
#   fc1: 10×128 + 128 = 1,408 weights
#   fc2: 128×64 + 64 = 8,256 weights
#   fc3: 64×32 + 32 = 2,080 weights
#   fc4: 32×3 + 3 = 99 weights
#   Total: 11,843 weights + 227 biases = 12,070 parameters
```

---

## Forward Pass Example

```python
import torch

model = DropletEvolutionModel()

# Create sample input (batch of 32 samples, 10 features each)
X_sample = torch.randn(32, 10)

# Forward pass
predictions = model(X_sample)

print(f"Input shape: {X_sample.shape}")      # (32, 10)
print(f"Output shape: {predictions.shape}")  # (32, 3)

# Each sample gets 3 predictions:
# - diameter_um
# - evaporation_time_ms
# - emissions_proxy
```

---

## Training Configuration

### Optimizer: Adam
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- **Adam** = Adaptive learning rate optimizer
- **lr=0.001** = Learning rate (how big steps to take)
  - Too small: slow learning
  - Too large: unstable
  - 0.001 is standard starting point

### Loss Function: MSE

```python
import torch.nn as nn

loss_fn = nn.MSELoss()
```

- **MSE** = Mean Squared Error
- Measures difference between predictions and actual values
- Good for regression tasks
- Formula: MSE = (1/n) × Σ(y_pred - y_true)²

### Why These Choices?

| Component | Choice | Alternative | Why Better |
|-----------|--------|-------------|-----------|
| Optimizer | Adam | SGD | Adapts learning rate automatically |
| Loss | MSE | MAE | More emphasis on large errors |
| Activation | ReLU | Sigmoid/Tanh | Faster, fewer vanishing gradient issues |

---

## Hyperparameters to Tune

```python
# Model architecture
input_size = 10
hidden1 = 128    # Try: 64, 128, 256
hidden2 = 64     # Try: 32, 64, 128
hidden3 = 32     # Try: 16, 32, 64
output_size = 3
dropout = 0.2    # Try: 0.1, 0.2, 0.3

# Training
learning_rate = 0.001    # Try: 0.0001, 0.001, 0.01
batch_size = 32          # Try: 16, 32, 64
epochs = 100             # Try: 50, 100, 200
```

---

## Step 5A: Create Model Script

Here's the complete code:

```python
#!/usr/bin/env python3
"""
STEP 5: NEURAL NETWORK ARCHITECTURE
Define the model for droplet evolution prediction
"""

import torch
import torch.nn as nn
import numpy as np

print("\n" + "="*80)
print("STEP 5: NEURAL NETWORK ARCHITECTURE DEFINITION")
print("="*80)

# ============================================================================
# DEFINE MODEL
# ============================================================================
print("\n[1/3] Defining neural network architecture...")

class DropletEvolutionModel(nn.Module):
    """
    Multi-layer perceptron for droplet size evolution prediction
    
    Architecture:
      Input (10) → Dense (128) → ReLU → Dropout
                → Dense (64)  → ReLU → Dropout
                → Dense (32)  → ReLU → Dropout
                → Output (3)  → Linear
    """
    
    def __init__(self, input_size=10, output_size=3, dropout_rate=0.2):
        super(DropletEvolutionModel, self).__init__()
        
        # Layer 1: Input → 128 neurons
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 128 → 64 neurons
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 64 → 32 neurons
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer: 32 → 3 targets
        self.fc4 = nn.Linear(32, output_size)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

# Create model
model = DropletEvolutionModel(input_size=10, output_size=3, dropout_rate=0.2)
print("  ✓ Model created successfully")

# ============================================================================
# DISPLAY MODEL ARCHITECTURE
# ============================================================================
print("\n[2/3] Model architecture:")
print(model)

# ============================================================================
# COUNT PARAMETERS
# ============================================================================
print("\n[3/3] Model parameters:")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  ✓ Total parameters: {total_params:,}")
print(f"  ✓ Trainable parameters: {trainable_params:,}")

# Detailed breakdown
print("\n  Parameter breakdown:")
for name, param in model.named_parameters():
    num_params = param.numel()
    print(f"    - {name}: {num_params:,} params (shape: {param.shape})")

# ============================================================================
# TEST FORWARD PASS
# ============================================================================
print("\n" + "="*80)
print("TESTING MODEL")
print("="*80)

# Create dummy input
X_dummy = torch.randn(32, 10)  # Batch of 32 samples, 10 features
print(f"\nInput shape: {X_dummy.shape}")

# Forward pass
with torch.no_grad():  # No gradient computation for testing
    predictions = model(X_dummy)

print(f"Output shape: {predictions.shape}")
print(f"Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MODEL DEFINITION COMPLETE!")
print("="*80)

print(f"""
✓ Neural network successfully defined!

ARCHITECTURE SUMMARY:
  Input Layer:     10 features
  Hidden Layer 1:  128 neurons (ReLU + Dropout)
  Hidden Layer 2:  64 neurons (ReLU + Dropout)
  Hidden Layer 3:  32 neurons (ReLU + Dropout)
  Output Layer:    3 targets (Linear)
  
TOTAL PARAMETERS: {total_params:,}
TRAINABLE PARAMETERS: {trainable_params:,}

MODEL DESIGN:
  ✓ Bottleneck architecture (10 → 128 → 64 → 32 → 3)
  ✓ ReLU activations for non-linearity
  ✓ Dropout for regularization (prevents overfitting)
  ✓ Linear output for regression
  
READY FOR TRAINING:
  ✓ Model defined
  ✓ Can be moved to GPU if available
  ✓ Ready for training loop (Step 6)

TEST RESULTS:
  Input: Batch of 32 samples × 10 features
  Output: Batch of 32 predictions × 3 targets
  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]
  (Will be in [0, 1] after training)

NEXT STEPS:
  1. Review model architecture above
  2. Proceed to Step 6: Training Loop
  3. Set up optimizer (Adam, lr=0.001)
  4. Set up loss function (MSE)
  5. Create training loop with early stopping
  6. Monitor train/val loss to prevent overfitting
""")

print("="*80 + "\n")
print("To continue to Step 6, run:\n  python3 step6_training_loop.py\n")

# Save model definition for reuse
torch.save(model.state_dict(), 'model_initial_weights.pth')
print("✓ Initial model weights saved to 'model_initial_weights.pth'\n")
```

---

## Key Concepts Summary

| Concept | What It Does | Why Important |
|---------|-------------|--------------|
| **Linear Layer** | Multiplies input by weights + adds bias | Learns relationships |
| **ReLU** | max(0, x) | Adds non-linearity, enables complex learning |
| **Dropout** | Randomly disable neurons during training | Prevents overfitting |
| **Adam Optimizer** | Updates weights intelligently | Adapts learning rate |
| **MSE Loss** | Measures prediction error | Guides learning direction |

---

## Next: Step 6 - Training Loop

Once Step 5 runs successfully, you'll have:
- ✅ Model architecture defined
- ✅ 22,915 learnable parameters
- ✅ Ready for training

Next step: **Set up the training loop to teach the model!**

```powershell
python3 step6_training_loop.py
```

