# Step 2: Data Exploration & Understanding (Your CFD Data)

## Your Data Overview

**File:** `Eulerian_1_-_eulerian1_1.csv`
- **Size:** 178,164 rows × 15 columns
- **Source:** Eulerian CFD simulation (single time step, multi-spatial grid points)
- **No missing values:** Clean data ✓

---

## What Your Data Represents

This is **Eulerian grid data** from a CFD simulation. Rather than tracking individual fuel droplets (Lagrangian), the Eulerian approach discretizes space into a grid and measures properties at each grid point.

### Columns Explained

**Flow Properties (Physical State):**
- `T` — Temperature (K). Range: 777–800 K
- `p` — Pressure (Pa). Range: ~5 MPa (nearly constant)
- `rho` — Density (kg/m³). Range: varies by location
- `U:0, U:1, U:2` — Velocity components (m/s). X, Y, Z directions

**Species Composition (Chemical Species):**
- `C7H16` — Heptane fuel (mass fraction)
- `CO2` — Carbon dioxide
- `H2O` — Water vapor
- `N2` — Nitrogen
- `O2` — Oxygen (oxidizer)

**Spatial Location (Grid Points):**
- `Points:0, Points:1, Points:2` — X, Y, Z coordinates (meters)
  - Range: -0.01 to 0.1 m (100 mm × 100 mm × 100 mm domain)

**Metadata:**
- `TimeStep` — All rows are from TimeStep 1 (single snapshot)

---

## Data Quality Assessment

✓ **No missing values**
✓ **No NaN or Inf values**
✓ **Reasonable physical ranges**

❌ **Single time step only** — For droplet dynamics, ideally you'd have multiple time snapshots
❌ **Single simulation case** — Only one pressure/temperature condition (5 MPa, ~800 K)

---

## Key Insight: What Can You Do With This Data?

This data shows the **Eulerian field** at one moment in time. To build a droplet model, you need to:

### Option A: Use This as **One Training Sample**
If you have multiple similar files from different conditions (different P, T, fuel type), you can:
- Treat each file as one "case"
- Extract statistics per case (e.g., mean T, max velocity, fuel distribution)
- Use case statistics as inputs to your droplet model

### Option B: Extract **Lagrangian Information**
Post-process this Eulerian data to track:
- Droplet trajectories through the domain
- Evaporation rates (change in fuel mass fraction along path)
- Temperature experienced by droplets

### Option C: Use for **CFD Surrogate Training**
Train a neural network to predict field variables (T, rho, species) given boundary conditions
- Inputs: boundary conditions, domain parameters
- Outputs: field values at grid points
- This becomes a surrogate for expensive CFD

---

## Step-by-Step Data Exploration Code

Here's exactly how to understand your data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv('Eulerian_1_-_eulerian1_1.csv')

# ===== BASIC EXPLORATION =====
print("Data shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nFirst few rows:")
print(data.head())

# ===== STATISTICS BY VARIABLE =====
print("\n" + "="*80)
print("STATISTICS FOR KEY VARIABLES")
print("="*80)

# Temperature
print("\nTemperature (T):")
print(f"  Min: {data['T'].min():.2f} K")
print(f"  Max: {data['T'].max():.2f} K")
print(f"  Mean: {data['T'].mean():.2f} K")
print(f"  Std: {data['T'].std():.2f} K")

# Pressure
print("\nPressure (p):")
print(f"  Min: {data['p'].min():.2e} Pa ({data['p'].min()/1e6:.2f} MPa)")
print(f"  Max: {data['p'].max():.2e} Pa ({data['p'].max()/1e6:.2f} MPa)")
print(f"  Variation: {(data['p'].max() - data['p'].min())/data['p'].mean() * 100:.2f}%")

# Density
print("\nDensity (rho):")
print(f"  Min: {data['rho'].min():.4f} kg/m³")
print(f"  Max: {data['rho'].max():.4f} kg/m³")
print(f"  Mean: {data['rho'].mean():.4f} kg/m³")

# Fuel concentration
print("\nC7H16 (Heptane fuel):")
print(f"  Min: {data['C7H16'].min()}")
print(f"  Max: {data['C7H16'].max()}")
print(f"  Non-zero rows: {(data['C7H16'] > 0).sum()}")

# Oxygen
print("\nO2 (Oxygen):")
print(f"  Min: {data['O2'].min():.4f}")
print(f"  Max: {data['O2'].max():.4f}")
print(f"  Mean: {data['O2'].mean():.4f}")

# Velocity
print("\nVelocity Components:")
print(f"  U:0 (X) range: [{data['U:0'].min():.4f}, {data['U:0'].max():.4f}] m/s")
print(f"  U:1 (Y) range: [{data['U:1'].min():.4f}, {data['U:1'].max():.4f}] m/s")
print(f"  U:2 (Z) range: [{data['U:2'].min():.4f}, {data['U:2'].max():.4f}] m/s")

velocity_magnitude = np.sqrt(data['U:0']**2 + data['U:1']**2 + data['U:2']**2)
print(f"  Velocity magnitude range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}] m/s")
print(f"  Mean velocity: {velocity_magnitude.mean():.4f} m/s")

# ===== SPATIAL DISTRIBUTION =====
print("\n" + "="*80)
print("SPATIAL DOMAIN")
print("="*80)
print(f"X (Points:0): [{data['Points:0'].min():.4f}, {data['Points:0'].max():.4f}] m")
print(f"Y (Points:1): [{data['Points:1'].min():.4f}, {data['Points:1'].max():.4f}] m")
print(f"Z (Points:2): [{data['Points:2'].min():.4f}, {data['Points:2'].max():.4f}] m")
print(f"Number of grid points: {len(data)}")

# ===== SPECIES CONSERVATION CHECK =====
print("\n" + "="*80)
print("SPECIES MASS FRACTIONS (Should sum ~1.0)")
print("="*80)
species_sum = data['C7H16'] + data['CO2'] + data['H2O'] + data['N2'] + data['O2']
print(f"Sum of species: min={species_sum.min():.4f}, max={species_sum.max():.4f}, mean={species_sum.mean():.4f}")
print("Note: If sum ≠ 1.0, other species might exist in the data")
```

---

## Visualization: Understanding Spatial Patterns

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== 3D SCATTER: Temperature Distribution =====
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['T'], cmap='hot', s=10, alpha=0.6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Temperature Distribution in 3D Space')
plt.colorbar(scatter, ax=ax, label='T (K)')
plt.savefig('temp_3d_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== 3D SCATTER: Fuel Concentration =====
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Points:0'], data['Points:1'], data['Points:2'], 
                     c=data['C7H16'], cmap='Blues', s=10, alpha=0.6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Fuel (C7H16) Distribution')
plt.colorbar(scatter, ax=ax, label='C7H16')
plt.savefig('fuel_3d_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== 2D SLICES =====
# Take a slice at Y ≈ 0.05 m
y_slice = 0.05
tolerance = 0.002
slice_data = data[(data['Points:1'] > y_slice - tolerance) & 
                  (data['Points:1'] < y_slice + tolerance)]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Temperature
axes[0, 0].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                   c=slice_data['T'], cmap='hot', s=20, alpha=0.7)
axes[0, 0].set_xlabel('X (m)')
axes[0, 0].set_ylabel('Z (m)')
axes[0, 0].set_title(f'Temperature at Y≈{y_slice} m')

# Fuel
axes[0, 1].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                   c=slice_data['C7H16'], cmap='Blues', s=20, alpha=0.7)
axes[0, 1].set_xlabel('X (m)')
axes[0, 1].set_ylabel('Z (m)')
axes[0, 1].set_title(f'Fuel at Y≈{y_slice} m')

# Density
axes[1, 0].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                   c=slice_data['rho'], cmap='viridis', s=20, alpha=0.7)
axes[1, 0].set_xlabel('X (m)')
axes[1, 0].set_ylabel('Z (m)')
axes[1, 0].set_title(f'Density at Y≈{y_slice} m')

# Velocity magnitude
vel_mag = np.sqrt(slice_data['U:0']**2 + slice_data['U:1']**2 + slice_data['U:2']**2)
axes[1, 1].scatter(slice_data['Points:0'], slice_data['Points:2'], 
                   c=vel_mag, cmap='plasma', s=20, alpha=0.7)
axes[1, 1].set_xlabel('X (m)')
axes[1, 1].set_ylabel('Z (m)')
axes[1, 1].set_title(f'Velocity Magnitude at Y≈{y_slice} m')

plt.tight_layout()
plt.savefig('field_slices.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== HISTOGRAMS =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(data['T'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Temperature (K)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Temperature Distribution')

axes[0, 1].hist(data['rho'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Density (kg/m³)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Density Distribution')

axes[0, 2].hist(data['C7H16'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('C7H16')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Fuel Concentration')

axes[1, 0].hist(data['O2'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('O2')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Oxygen Distribution')

vel_mag = np.sqrt(data['U:0']**2 + data['U:1']**2 + data['U:2']**2)
axes[1, 1].hist(vel_mag, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Velocity Magnitude (m/s)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Velocity Distribution')

axes[1, 2].hist(data['p'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Pressure (Pa)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Pressure Distribution')

plt.tight_layout()
plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualizations saved!")
```

---

## Next Steps: Preparing for ML Training

### Challenge: Single File, Single Condition

Your current data is **one snapshot** of **one case**. For ML training, you typically need:
- **Multiple cases** (different initial conditions)
- **Time evolution** (droplet evaporation over time)

### Solutions:

**Option 1: Generate Multiple Cases**
If you have access to the CFD solver (ANSYS Fluent, OpenFOAM, etc.):
- Run simulations with different inputs:
  - Varying droplet diameter: 10, 20, 50, 100 μm
  - Varying temperature: 800, 1200, 1600, 2000 K
  - Varying pressure: 1, 5, 10 MPa
  - Different fuel types: heptane, methanol, kerosene
- Extract outputs: evaporation time, final size, emissions

**Option 2: Use This as a Test Case**
If you have other Eulerian files from Dr. Mishra's prior work:
- Compile all available files into a dataset
- Normalize and aggregate into training examples

**Option 3: Extract Features from This Field**
Even from one file, you can create training examples:
- Divide the domain into sub-regions (droplet parcels)
- For each region: compute local field properties (T, P, ρ, species)
- Correlate with nearby evaporation/combustion behavior

---

## Summary Table: Your Data

| Aspect | Details |
|--------|---------|
| **Format** | Eulerian CFD field data (grid-based) |
| **Sample size** | 178,164 grid points (spatial locations) |
| **Inputs available** | T, p, ρ, composition (C7H16, O2, N2, CO2, H2O), velocity |
| **Outputs available** | None (this is raw field data, not aggregated droplet properties) |
| **Time dimension** | Single time step (snapshot) |
| **Condition coverage** | Single case (5 MPa, ~800 K, heptane) |
| **Data quality** | Excellent (no missing values, physically reasonable) |
| **Next action** | Obtain/generate multiple cases with different initial conditions |

---

## Key Questions for Dr. Mishra

Before proceeding to Steps 3–10, clarify:

1. **Do you have multiple CFD files?** (Different P, T, fuel type, droplet size)
2. **Do you have time-series data?** (Evolution of a single case over time)
3. **What are your target outputs?**
   - Evaporation time?
   - Temperature profiles?
   - Emissions (NOx, soot)?
   - Droplet size evolution?
4. **Do you have Sandia Flame D experimental data?** (For final validation)

Once you answer these, I can show you exactly how to format the data for Steps 3–10.

