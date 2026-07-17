#!/usr/bin/env python3
"""
Step 2: Visualize Your Eulerian CFD Data
Creates 3D and 2D plots to understand spatial patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("\nLoading data for visualization...")
data = pd.read_csv('Eulerian_1_-_eulerian1_1.csv')
print(f"✓ Loaded {len(data):,} points\n")

# Compute velocity magnitude
data['vel_mag'] = np.sqrt(data['U:0']**2 + data['U:1']**2 + data['U:2']**2)

# ============================================================================
# PLOT 1: 3D Temperature Distribution
# ============================================================================
print("Creating Plot 1: 3D Temperature Distribution...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['T'], cmap='hot', s=5, alpha=0.6)

ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('Temperature Distribution in 3D Domain', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Temperature (K)')
plt.tight_layout()
plt.savefig('01_Temperature_3D.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 01_Temperature_3D.png")
plt.close()

# ============================================================================
# PLOT 2: 3D Fuel (C7H16) Distribution
# ============================================================================
print("Creating Plot 2: 3D Fuel Distribution...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['C7H16'], cmap='Blues', s=5, alpha=0.6)

ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('Fuel (C7H16) Concentration in 3D Domain', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='C7H16 (mass fraction)')
plt.tight_layout()
plt.savefig('02_Fuel_3D.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 02_Fuel_3D.png")
plt.close()

# ============================================================================
# PLOT 3: 3D Oxygen Distribution
# ============================================================================
print("Creating Plot 3: 3D Oxygen Distribution...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['O2'], cmap='Greens', s=5, alpha=0.6)

ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('Oxygen (O2) Distribution in 3D Domain', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='O2 (mass fraction)')
plt.tight_layout()
plt.savefig('03_Oxygen_3D.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 03_Oxygen_3D.png")
plt.close()

# ============================================================================
# PLOT 4: 3D Density Distribution
# ============================================================================
print("Creating Plot 4: 3D Density Distribution...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['rho'], cmap='viridis', s=5, alpha=0.6)

ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('Density Distribution in 3D Domain', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Density (kg/m³)')
plt.tight_layout()
plt.savefig('04_Density_3D.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 04_Density_3D.png")
plt.close()

# ============================================================================
# PLOT 5: 3D Velocity Magnitude
# ============================================================================
print("Creating Plot 5: 3D Velocity Magnitude...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['vel_mag'], cmap='plasma', s=5, alpha=0.6)

ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('Velocity Magnitude in 3D Domain', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Velocity Magnitude (m/s)')
plt.tight_layout()
plt.savefig('05_Velocity_3D.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 05_Velocity_3D.png")
plt.close()

# ============================================================================
# PLOT 6: 2D Slices at Y ≈ 0.05 m (Middle of domain)
# ============================================================================
print("Creating Plot 6: 2D Field Slices at Y ≈ 0.05 m...")

y_slice = 0.05
tolerance = 0.003
slice_data = data[(data['Points:1'] > y_slice - tolerance) & 
                  (data['Points:1'] < y_slice + tolerance)].copy()

if len(slice_data) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Field Slices at Y ≈ {y_slice} m (XZ Plane)', 
                 fontsize=14, fontweight='bold')
    
    # Temperature
    scatter = axes[0, 0].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['T'], cmap='hot', s=30, alpha=0.7)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Z (m)')
    axes[0, 0].set_title('Temperature (K)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Fuel
    scatter = axes[0, 1].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['C7H16'], cmap='Blues', s=30, alpha=0.7)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].set_title('Fuel (C7H16)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Oxygen
    scatter = axes[0, 2].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['O2'], cmap='Greens', s=30, alpha=0.7)
    axes[0, 2].set_xlabel('X (m)')
    axes[0, 2].set_ylabel('Z (m)')
    axes[0, 2].set_title('Oxygen (O2)')
    plt.colorbar(scatter, ax=axes[0, 2])
    
    # Density
    scatter = axes[1, 0].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['rho'], cmap='viridis', s=30, alpha=0.7)
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_title('Density (kg/m³)')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Velocity magnitude
    scatter = axes[1, 1].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['vel_mag'], cmap='plasma', s=30, alpha=0.7)
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Z (m)')
    axes[1, 1].set_title('Velocity Magnitude (m/s)')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # Pressure
    scatter = axes[1, 2].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                                  c=slice_data['p']/1e6, cmap='coolwarm', s=30, alpha=0.7)
    axes[1, 2].set_xlabel('X (m)')
    axes[1, 2].set_ylabel('Z (m)')
    axes[1, 2].set_title('Pressure (MPa)')
    plt.colorbar(scatter, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('06_Field_Slices_XZ.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: 06_Field_Slices_XZ.png")
    plt.close()
else:
    print("  ⚠️  No data at that Y slice")

# ============================================================================
# PLOT 7: Distribution Histograms
# ============================================================================
print("Creating Plot 7: Distribution Histograms...")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Field Variable Distributions', fontsize=14, fontweight='bold')

# Row 1: T, p, rho
axes[0, 0].hist(data['T'], bins=60, edgecolor='black', alpha=0.7, color='red')
axes[0, 0].set_xlabel('Temperature (K)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Temperature Distribution')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(data['p']/1e6, bins=60, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Pressure (MPa)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Pressure Distribution')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].hist(data['rho'], bins=60, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].set_xlabel('Density (kg/m³)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Density Distribution')
axes[0, 2].grid(alpha=0.3)

# Row 2: Species
axes[1, 0].hist(data['C7H16'], bins=60, edgecolor='black', alpha=0.7, color='blue')
axes[1, 0].set_xlabel('C7H16 (Fuel)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Fuel Distribution')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(data['O2'], bins=60, edgecolor='black', alpha=0.7, color='cyan')
axes[1, 1].set_xlabel('O2 (Oxygen)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Oxygen Distribution')
axes[1, 1].grid(alpha=0.3)

axes[1, 2].hist(data['N2'], bins=60, edgecolor='black', alpha=0.7, color='purple')
axes[1, 2].set_xlabel('N2 (Nitrogen)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Nitrogen Distribution')
axes[1, 2].grid(alpha=0.3)

# Row 3: Velocity components and magnitude
axes[2, 0].hist(data['vel_mag'], bins=60, edgecolor='black', alpha=0.7, color='magenta')
axes[2, 0].set_xlabel('Velocity Magnitude (m/s)')
axes[2, 0].set_ylabel('Frequency')
axes[2, 0].set_title('Velocity Magnitude Distribution')
axes[2, 0].grid(alpha=0.3)

axes[2, 1].hist(data['U:0'], bins=60, edgecolor='black', alpha=0.7, color='brown')
axes[2, 1].set_xlabel('U:0 (X-velocity, m/s)')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].set_title('X-Velocity Distribution')
axes[2, 1].grid(alpha=0.3)

axes[2, 2].hist(data['U:2'], bins=60, edgecolor='black', alpha=0.7, color='gray')
axes[2, 2].set_xlabel('U:2 (Z-velocity, m/s)')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].set_title('Z-Velocity Distribution')
axes[2, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('07_Distribution_Histograms.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 07_Distribution_Histograms.png")
plt.close()

# ============================================================================
# PLOT 8: Correlation Matrix (Heatmap)
# ============================================================================
print("Creating Plot 8: Variable Correlations...")

numeric_cols = ['T', 'p', 'rho', 'C7H16', 'O2', 'N2', 'vel_mag']
correlation = data[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax.set_yticklabels(numeric_cols)

# Add correlation values to cells
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Correlation Matrix of Field Variables', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig('08_Correlation_Matrix.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 08_Correlation_Matrix.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated plots:")
print("  01_Temperature_3D.png         - 3D temperature field")
print("  02_Fuel_3D.png               - 3D fuel concentration")
print("  03_Oxygen_3D.png             - 3D oxygen distribution")
print("  04_Density_3D.png            - 3D density field")
print("  05_Velocity_3D.png           - 3D velocity magnitude")
print("  06_Field_Slices_XZ.png       - 2D slices at Y≈0.05m (6 variables)")
print("  07_Distribution_Histograms.png - Histograms of all variables")
print("  08_Correlation_Matrix.png    - Variable correlations")
print("\nAll plots saved to current directory")
print("="*80 + "\n")
