#!/usr/bin/env python3
"""
PATH 2: Extract Features from Eulerian CFD Field
For Droplet Size Evolution Prediction

This script:
1. Loads your Eulerian CFD field data
2. Divides domain into "droplet parcels" (spatial regions)
3. Extracts field properties (T, P, density, species, velocity) as FEATURES
4. Creates synthetic droplet size evolution as TARGET outputs
   (Using D² law + physics-based model)
5. Saves training data ready for ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PATH 2: DROPLET FEATURE EXTRACTION FROM EULERIAN FIELD")
print("="*80)

# ============================================================================
# SECTION 1: LOAD EULERIAN DATA
# ============================================================================
print("\n[1/6] Loading Eulerian CFD data...")

data = pd.read_csv('Eulerian_1_-_eulerian1_1.csv')
print(f"  ✓ Loaded {len(data):,} grid points")

# Compute derived quantities
data['vel_mag'] = np.sqrt(data['U:0']**2 + data['U:1']**2 + data['U:2']**2)
data['H2O_total'] = data['H2O']  # Or CO2 as proxy for combustion products
data['species_sum'] = data['C7H16'] + data['CO2'] + data['H2O'] + data['N2'] + data['O2']

print(f"  Domain: X=[{data['Points:0'].min():.3f}, {data['Points:0'].max():.3f}] m")
print(f"          Y=[{data['Points:1'].min():.3f}, {data['Points:1'].max():.3f}] m")
print(f"          Z=[{data['Points:2'].min():.3f}, {data['Points:2'].max():.3f}] m")
print(f"  Pressure: {data['p'].mean()/1e6:.2f} MPa")
print(f"  Temperature: {data['T'].mean():.1f} K")

# ============================================================================
# SECTION 2: CREATE VIRTUAL DROPLET PARCELS
# ============================================================================
print("\n[2/6] Creating virtual droplet parcels...")

# Define parcel grid spacing (10 mm = 0.01 m)
# This creates ~1000 parcels from the domain
parcel_dx = 0.015  # 15 mm spacing
parcel_dy = 0.015
parcel_dz = 0.005

parcels = []
parcel_id = 0

x_min, x_max = data['Points:0'].min(), data['Points:0'].max()
y_min, y_max = data['Points:1'].min(), data['Points:1'].max()
z_min, z_max = data['Points:2'].min(), data['Points:2'].max()

print(f"  Parcel size: {parcel_dx*1000:.1f}mm × {parcel_dy*1000:.1f}mm × {parcel_dz*1000:.1f}mm")

for x in np.arange(x_min, x_max - parcel_dx, parcel_dx):
    for y in np.arange(y_min, y_max - parcel_dy, parcel_dy):
        for z in np.arange(z_min, z_max - parcel_dz, parcel_dz):
            
            # Extract grid points in this parcel
            parcel_mask = (
                (data['Points:0'] >= x) & (data['Points:0'] < x + parcel_dx) &
                (data['Points:1'] >= y) & (data['Points:1'] < y + parcel_dy) &
                (data['Points:2'] >= z) & (data['Points:2'] < z + parcel_dz)
            )
            
            parcel_data = data[parcel_mask]
            
            if len(parcel_data) > 5:  # Only if parcel has enough points
                
                # ===== EXTRACT FEATURES FROM PARCEL =====
                # These are properties experienced by a droplet parcel
                
                parcel_dict = {
                    'parcel_id': parcel_id,
                    'x_center': x + parcel_dx/2,
                    'y_center': y + parcel_dy/2,
                    'z_center': z + parcel_dz/2,
                    
                    # THERMODYNAMIC FEATURES
                    'T_local_mean': parcel_data['T'].mean(),
                    'T_local_std': parcel_data['T'].std(),
                    'p_local_mean': parcel_data['p'].mean(),
                    'rho_local_mean': parcel_data['rho'].mean(),
                    
                    # SPECIES COMPOSITION FEATURES
                    'O2_mean': parcel_data['O2'].mean(),
                    'O2_std': parcel_data['O2'].std(),
                    'C7H16_mean': parcel_data['C7H16'].mean(),
                    'C7H16_max': parcel_data['C7H16'].max(),
                    'N2_mean': parcel_data['N2'].mean(),
                    'H2O_mean': parcel_data['H2O'].mean(),
                    'CO2_mean': parcel_data['CO2'].mean(),
                    
                    # FLOW FEATURES
                    'vel_mag_mean': parcel_data['vel_mag'].mean(),
                    'vel_mag_max': parcel_data['vel_mag'].max(),
                    'vel_x_mean': parcel_data['U:0'].mean(),
                    'vel_z_mean': parcel_data['U:2'].mean(),
                    
                    # SPATIAL FEATURES
                    'distance_from_inlet': x + parcel_dx/2,  # Distance along X
                    'radial_distance': np.sqrt(y**2 + z**2),  # Radial from centerline
                }
                
                parcels.append(parcel_dict)
                parcel_id += 1

parcels_df = pd.DataFrame(parcels)
print(f"  ✓ Created {len(parcels_df)} parcels")

# ============================================================================
# SECTION 3: CREATE DROPLET SIZE EVOLUTION TARGETS
# ============================================================================
print("\n[3/6] Generating droplet size evolution targets...")
print("      (Using physics-based D² law + combustion model)")

"""
D² Law Physics:
The droplet diameter squared decreases linearly with time:
    d² = d₀² - K·t
    
Where:
    d₀ = initial droplet diameter
    K = evaporation constant = (λ·(T∞-Ts)) / (ρ_liquid·L_v)
    
For combustion, we add:
    - Combustion heating (increases evaporation)
    - Fuel consumption (decreases available evaporation)
    
Simplified model:
    K ∝ (T - T_ref) × O2_availability × fuel_availability
"""

# Simulation parameters
initial_diameters = [10, 20, 50, 100]  # μm
time_steps = np.linspace(0, 10, 11)  # ms

all_trajectories = []

for d0_um in initial_diameters:
    d0_m = d0_um * 1e-6  # Convert to meters
    
    for idx, parcel in parcels_df.iterrows():
        
        # Physics-based evaporation model
        # Evaporation rate constant
        T_local = parcel['T_local_mean']
        T_ref = 300  # Reference temperature (K)
        
        # Heat driving evaporation
        heat_factor = max(0, (T_local - T_ref) / 1000)  # Normalized to 1 at 1300K
        
        # Fuel availability (decreases as we move downstream)
        fuel_factor = parcel['C7H16_mean']
        
        # Oxygen availability
        O2_factor = parcel['O2_mean']
        
        # Combined evaporation constant (K in D² law)
        # Higher when hot, has fuel, has oxygen
        K_base = 1e-9  # m²/s (typical for heptane)
        K_local = K_base * (1 + 2 * heat_factor) * (1 + O2_factor) * (1 + fuel_factor)
        
        # Velocity effect: faster evaporation with higher relative velocity
        velocity_enhance = 1 + 0.5 * min(parcel['vel_mag_mean'], 1.0)
        K_local *= velocity_enhance
        
        # Time to complete evaporation
        t_evap = (d0_m**2) / K_local if K_local > 0 else 100  # seconds
        t_evap_ms = t_evap * 1000  # Convert to ms
        
        for t_ms in time_steps:
            t_s = t_ms / 1000  # Convert to seconds
            
            # D² law
            if t_s < t_evap:
                d_squared = d0_m**2 - K_local * t_s
                if d_squared > 0:
                    d_current_m = np.sqrt(d_squared)
                    d_current_um = d_current_m * 1e6
                else:
                    d_current_um = 0.1  # Nearly evaporated
            else:
                d_current_um = 0.1  # Fully evaporated
            
            # Temperature experienced
            T_experienced = parcel['T_local_mean']
            
            # Emissions proxy: droplet evaporation generates heat/NOx
            # Higher T, more residence time → more NOx
            emissions_proxy = (T_experienced - 300) * max(0, t_evap_ms - t_ms)
            
            trajectory = {
                'parcel_id': parcel['parcel_id'],
                'initial_diameter_um': d0_um,
                'time_ms': t_ms,
                'diameter_um': d_current_um,
                
                # Field properties at this parcel
                'T_K': T_experienced,
                'p_Pa': parcel['p_local_mean'],
                'rho_kg_m3': parcel['rho_local_mean'],
                'O2_fraction': parcel['O2_mean'],
                'C7H16_fraction': parcel['C7H16_mean'],
                'vel_mag_m_s': parcel['vel_mag_mean'],
                'distance_from_inlet_m': parcel['distance_from_inlet'],
                'radial_distance_m': parcel['radial_distance'],
                
                # Computed outputs
                'evaporation_time_ms': t_evap_ms,
                'emissions_proxy': emissions_proxy,
                'diameter_fraction': d_current_um / d0_um,  # Fraction remaining
            }
            
            all_trajectories.append(trajectory)

trajectories_df = pd.DataFrame(all_trajectories)
print(f"  ✓ Generated {len(trajectories_df)} droplet evolution samples")
print(f"    - {len(initial_diameters)} initial sizes")
print(f"    - {len(parcels_df)} spatial locations")
print(f"    - {len(time_steps)} time steps")

# ============================================================================
# SECTION 4: PREPARE DATA FOR ML TRAINING
# ============================================================================
print("\n[4/6] Preparing ML training data...")

# Create feature matrix
feature_cols = [
    'initial_diameter_um', 'time_ms',
    'T_K', 'p_Pa', 'rho_kg_m3',
    'O2_fraction', 'C7H16_fraction',
    'vel_mag_m_s', 'distance_from_inlet_m', 'radial_distance_m'
]

target_cols = [
    'diameter_um', 'evaporation_time_ms', 'emissions_proxy'
]

X = trajectories_df[feature_cols].copy()
y = trajectories_df[target_cols].copy()

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Target matrix shape: {y.shape}")
print(f"\n  Features: {feature_cols}")
print(f"\n  Targets: {target_cols}")

# Show statistics
print("\n  Feature Statistics:")
print(X.describe().round(4))

print("\n  Target Statistics:")
print(y.describe().round(4))

# ============================================================================
# SECTION 5: SAVE DATA FILES
# ============================================================================
print("\n[5/6] Saving data files...")

# Save full trajectory data
trajectories_df.to_csv('droplet_trajectories_full.csv', index=False)
print("  ✓ droplet_trajectories_full.csv")

# Save feature matrix
X.to_csv('droplet_features.csv', index=False)
print("  ✓ droplet_features.csv")

# Save target matrix
y.to_csv('droplet_targets.csv', index=False)
print("  ✓ droplet_targets.csv")

# Save parcel metadata
parcels_df.to_csv('parcel_metadata.csv', index=False)
print("  ✓ parcel_metadata.csv")

# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

# Plot 1: Droplet size evolution for different initial sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for d0 in initial_diameters:
    subset = trajectories_df[trajectories_df['initial_diameter_um'] == d0]
    time_data = subset.groupby('time_ms')['diameter_um'].mean()
    axes[0].plot(time_data.index, time_data.values, marker='o', label=f'd0={d0}um')

axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Droplet Diameter (um)')
axes[0].set_title('Average Droplet Size Evolution by Initial Diameter')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Evaporation time vs initial diameter
evap_times = trajectories_df.drop_duplicates(['initial_diameter_um'])[
    ['initial_diameter_um', 'evaporation_time_ms']
].sort_values('initial_diameter_um')

axes[1].plot(evap_times['initial_diameter_um'], evap_times['evaporation_time_ms'], 
             marker='s', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('Initial Droplet Diameter (um)')
axes[1].set_ylabel('Evaporation Time (ms)')
axes[1].set_title('Evaporation Time vs Initial Diameter (D2 Law)')
axes[1].grid(alpha=0.3)

# Add D² law line (d² ∝ t)
fit_coeffs = np.polyfit(evap_times['initial_diameter_um']**2, 
                        evap_times['evaporation_time_ms'], 1)
fit_line = fit_coeffs[0] * evap_times['initial_diameter_um']**2 + fit_coeffs[1]
axes[1].plot(evap_times['initial_diameter_um'], fit_line, '--', alpha=0.5, label='D2 fit')
axes[1].legend()

plt.tight_layout()
plt.savefig('01_Droplet_Evolution.png', dpi=150, bbox_inches='tight')
print("  ✓ 01_Droplet_Evolution.png")
plt.close()

# Plot 3: Temperature influence on evaporation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Temperature vs evaporation time
evap_by_location = trajectories_df.drop_duplicates(['parcel_id', 'initial_diameter_um'])[
    ['T_K', 'evaporation_time_ms', 'initial_diameter_um']
]

for d0 in initial_diameters:
    subset = evap_by_location[evap_by_location['initial_diameter_um'] == d0]
    axes[0].scatter(subset['T_K'], subset['evaporation_time_ms'], 
                   alpha=0.6, label=f'd0={d0}um', s=30)

axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Evaporation Time (ms)')
axes[0].set_title('Temperature Effect on Evaporation Time')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Spatial distribution of evaporation times (average)
spatial_evap = trajectories_df.groupby('parcel_id').agg({
    'distance_from_inlet_m': 'first',
    'evaporation_time_ms': 'mean',
    'T_K': 'first'
}).reset_index()

scatter = axes[1].scatter(spatial_evap['distance_from_inlet_m'], 
                         spatial_evap['evaporation_time_ms'],
                         c=spatial_evap['T_K'], cmap='hot', s=50, alpha=0.7)
axes[1].set_xlabel('Distance from Inlet (m)')
axes[1].set_ylabel('Average Evaporation Time (ms)')
axes[1].set_title('Spatial Variation of Evaporation Time')
cbar = plt.colorbar(scatter, ax=axes[1], label='Temperature (K)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_Temperature_Effects.png', dpi=150, bbox_inches='tight')
print("  ✓ 02_Temperature_Effects.png")
plt.close()

# Plot 4: Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Sample to avoid overplotting
sample_trajectories = trajectories_df.sample(min(5000, len(trajectories_df)))

# Plot 1: Temperature
axes[0, 0].hist(sample_trajectories['T_K'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Temperature (K)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Temperature Distribution')

# Plot 2: O2
axes[0, 1].hist(sample_trajectories['O2_fraction'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('O2 Mass Fraction')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Oxygen Distribution')

# Plot 3: C7H16
axes[0, 2].hist(sample_trajectories['C7H16_fraction'], bins=50, edgecolor='black', alpha=0.7, color='blue')
axes[0, 2].set_xlabel('C7H16 Mass Fraction')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Fuel Distribution')

# Plot 4: Velocity
axes[1, 0].hist(sample_trajectories['vel_mag_m_s'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Velocity Magnitude (m/s)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Velocity Distribution')

# Plot 5: Initial diameter
axes[1, 1].hist(sample_trajectories['initial_diameter_um'], bins=50, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Initial Diameter (um)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Droplet Size Distribution')

# Plot 6: Current diameter
axes[1, 2].hist(sample_trajectories['diameter_um'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 2].set_xlabel('Current Diameter (um)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Current Droplet Size Distribution')

plt.tight_layout()
plt.savefig('03_Feature_Distributions.png', dpi=150, bbox_inches='tight')
print("  ✓ 03_Feature_Distributions.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXTRACTION COMPLETE!")
print("="*80)

print(f"""
✓ Created {len(trajectories_df):,} training samples from 1 Eulerian snapshot

DATA FILES SAVED:
  1. droplet_trajectories_full.csv  - All parcel locations × diameters × time steps
  2. droplet_features.csv           - Input features (10 columns)
  3. droplet_targets.csv            - Target outputs (3 columns)
  4. parcel_metadata.csv            - Spatial and field properties of each parcel

DATA STRUCTURE:
  - Samples: {len(trajectories_df):,}
  - Features: {len(feature_cols)} (T, p, rho, O2, C7H16, velocity, position, time, diameter)
  - Targets: {len(target_cols)} (diameter, evaporation time, emissions proxy)
  
FEATURES (Inputs):
  {feature_cols}
  
TARGETS (Outputs):
  {target_cols}

PHYSICS MODEL USED:
  - D2 Law: d² = d0² - K·t
  - Evaporation constant K depends on:
    * Local temperature
    * Fuel availability
    * Oxygen availability
    * Local velocity

NEXT STEPS:
  1. Review the generated CSV files
  2. Proceed to Step 3: Normalization
  3. Use standard ML workflow (train/val/test split)
  4. Validate against Sandia Flame D data (see separate guide)
""")

print("\nGeneration stats:")
print(f"  Temperature range: {trajectories_df['T_K'].min():.1f} - {trajectories_df['T_K'].max():.1f} K")
print(f"  Diameter range: {trajectories_df['diameter_um'].min():.2f} - {trajectories_df['diameter_um'].max():.2f} um")
print(f"  Evaporation time range: {trajectories_df['evaporation_time_ms'].min():.3f} - {trajectories_df['evaporation_time_ms'].max():.3f} ms")

print("\n" + "="*80)
