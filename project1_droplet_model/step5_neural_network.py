#!/usr/bin/env python3
"""
STEP 5: NEURAL NETWORK ARCHITECTURE
Define the model for droplet evolution prediction
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
X_dummy = torch.randn(32, 10)
print(f"\nInput shape: {X_dummy.shape}")

# Forward pass
with torch.no_grad():
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
  Hidden Layer 1:  128 neurons (ReLU + Dropout 0.2)
  Hidden Layer 2:  64 neurons (ReLU + Dropout 0.2)
  Hidden Layer 3:  32 neurons (ReLU + Dropout 0.2)
  Output Layer:    3 targets (Linear, no activation)
  
TOTAL PARAMETERS: {total_params:,}
TRAINABLE PARAMETERS: {trainable_params:,}

MODEL DESIGN:
  ✓ Bottleneck architecture (10 → 128 → 64 → 32 → 3)
  ✓ ReLU activations for non-linearity
  ✓ Dropout (20%) for regularization
  ✓ Linear output for regression

TEST FORWARD PASS:
  Input: Batch of 32 samples × 10 features
  Output: Batch of 32 predictions × 3 targets
  Output range: [{predictions.min():.4f}, {predictions.max():.4f}]

WHAT'S NEXT:
  1. Set up training loop (Step 6)
  2. Configure optimizer: Adam (lr=0.001)
  3. Configure loss: MSE
  4. Train for 100 epochs
  5. Monitor train/val loss
  6. Use early stopping to prevent overfitting

YOUR PROGRESS:
  ✓ Step 1: Understand Data
  ✓ Step 2: Extract Features
  ✓ Step 3: Normalize
  ✓ Step 4: Split Data
  ✓ Step 5: Define Model
  → Step 6: Training Loop (NEXT)
""")

print("="*80 + "\n")

# Save model definition
torch.save(model.state_dict(), 'model_initial_weights.pth')
print("✓ Model saved to: model_initial_weights.pth\n")

print("To continue to Step 6, run:\n  python3 step6_training_loop.py\n")
