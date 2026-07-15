# Your CFD Data Summary: Key Insights & Next Steps

## Quick Summary

| Aspect | Your Data |
|--------|-----------|
| **Type** | Eulerian grid-based CFD field snapshot |
| **Size** | 178,164 grid points in a 100mm × 100mm × 100mm domain |
| **Case** | Single simulation at 5 MPa, ~800 K with heptane fuel |
| **Time** | Single time step (no temporal evolution) |
| **Quality** | Excellent (no missing values, physically reasonable) |

---

## What Your Data Contains

### ✓ **Spatial Field Data**
- **Temperature (T):** 777–800 K (mostly constant ~800 K)
- **Pressure (p):** ~5 MPa (nearly constant)
- **Density (ρ):** 21–23 kg/m³ (reflects gas-phase combustion products)
- **Fuel (C7H16):** Zero in most locations (fuel consumed in flame)
- **Oxygen (O2):** 0–0.23 (decreases where fuel burns)
- **Nitrogen (N2):** 0.76 (inert diluent)
- **Velocity:** Small magnitudes (< 1 m/s in most of domain)

### ✓ **Spatial Coverage**
- Regular grid from (-0.01, 0, -0.01) to (0.1, 0.1, 0.01) meters
- Represents a combustion domain with inlet, flame region, and exit

---

## **Critical Issue: Data Structure vs. ML Requirements**

### What You Currently Have
```
Grid Point #1: T=800K, p=5MPa, ρ=21.7 kg/m³, C7H16=0, O2=0.234
Grid Point #2: T=799K, p=5MPa, ρ=21.7 kg/m³, C7H16=0, O2=0.234
Grid Point #3: T=778K, p=5MPa, ρ=21.7 kg/m³, C7H16=0, O2=0.232
...
```

### What ML Training Needs
```
Case 1: Diameter=10μm, T_ambient=800K, P=5MPa → Evap_time=1.2ms, Emissions=45ppm
Case 2: Diameter=20μm, T_ambient=800K, P=5MPa → Evap_time=2.5ms, Emissions=42ppm
Case 3: Diameter=50μm, T_ambient=1200K, P=5MPa → Evap_time=5.1ms, Emissions=38ppm
...
```

### The Gap
Your current file is **one Eulerian field snapshot**. For droplet model training, you need:
1. **Multiple simulation cases** (different conditions)
2. **Aggregated outputs** (droplet-level predictions, not grid-point data)
3. **Time evolution** (how droplet properties change over time)

---

## Three Paths Forward

### **Path 1: Generate Multiple CFD Cases (Recommended)**

**If you have access to a CFD solver:**

Run simulations with varying inputs:
```
Simulation A: Droplet diameter=10μm, T=800K, P=5MPa, Fuel=C7H16
              → Extract: evaporation_time, peak_temp, NO_x_emissions
              
Simulation B: Droplet diameter=20μm, T=800K, P=5MPa, Fuel=C7H16
              → Extract: evaporation_time, peak_temp, NO_x_emissions
              
Simulation C: Droplet diameter=50μm, T=1200K, P=5MPa, Fuel=C7H16
              → Extract: evaporation_time, peak_temp, NO_x_emissions
...
```

**Result:** Compile into CSV:
```
diameter,T,p,fuel,evap_time,peak_temp,emissions
10,800,5e6,c7h16,1.2,2050,45
20,800,5e6,c7h16,2.5,2200,42
50,1200,5e6,c7h16,5.1,2300,38
```

### **Path 2: Extract Features from This Eulerian Field**

**If generating multiple cases is too expensive:**

Treat this single Eulerian snapshot as containing multiple "virtual droplets":

```python
import pandas as pd

data = pd.read_csv('Eulerian_1_-_eulerian1_1.csv')

# Create "training examples" by grouping spatial regions
# Assume each region represents a droplet parcel

def create_parcel_examples(data, parcel_size=0.01):
    """
    Divide domain into small parcels.
    Each parcel = one training example.
    """
    examples = []
    
    for x_min in np.arange(-0.01, 0.09, parcel_size):
        for y_min in np.arange(0, 0.09, parcel_size):
            for z_min in np.arange(-0.01, 0.009, parcel_size):
                
                # Filter points in this parcel
                parcel = data[
                    (data['Points:0'] >= x_min) & (data['Points:0'] < x_min + parcel_size) &
                    (data['Points:1'] >= y_min) & (data['Points:1'] < y_min + parcel_size) &
                    (data['Points:2'] >= z_min) & (data['Points:2'] < z_min + parcel_size)
                ]
                
                if len(parcel) > 0:
                    # Extract local field properties
                    example = {
                        'x': parcel['Points:0'].mean(),
                        'y': parcel['Points:1'].mean(),
                        'z': parcel['Points:2'].mean(),
                        'T_local': parcel['T'].mean(),
                        'rho_local': parcel['rho'].mean(),
                        'O2_local': parcel['O2'].mean(),
                        'vel_mag': np.sqrt(parcel['U:0'].mean()**2 + 
                                          parcel['U:1'].mean()**2 + 
                                          parcel['U:2'].mean()**2),
                        # You would need experimental/CFD outputs for these:
                        # 'evap_time': ?,
                        # 'emissions': ?,
                    }
                    examples.append(example)
    
    return pd.DataFrame(examples)

# Create parcel data
parcels_df = create_parcel_examples(data)
parcels_df.to_csv('parcel_features.csv', index=False)
```

**Limitation:** Without actual droplet outputs (evaporation time, emissions), this is just feature extraction. You still need the target labels.

### **Path 3: Use Published Sandia Data**

**If Dr. Mishra's CFD runs are proprietary:**

Use publicly available Sandia Flame D data as your training set:
- Download from Sandia Labs online database
- Contains measured droplet properties, temperatures, species profiles
- Standard benchmark for model validation anyway

---

## Recommended Next Steps (Your Action Items)

### Week 1: Data Availability
1. **Contact Dr. Mishra:**
   - Do you have other CFD simulations from previous work?
   - What parameter ranges were explored? (P, T, fuel type, droplet size)
   - Can you run additional simulations with different conditions?

2. **Define Output Targets:**
   - What do we want to predict?
     - Evaporation time? ✓
     - Temperature profiles? ✓
     - NO_x / soot emissions? ✓
     - Droplet size evolution? ✓

3. **Obtain Sandia Flame D:**
   - Download reference data from: https://icsem.sandia.gov/TNF/
   - This will be your final test set anyway

### Week 2: Data Preparation
1. **Compile Dataset:**
   - If multiple CFD files exist: merge them
   - Extract inputs (initial conditions) and outputs (results)
   - Create single CSV with all cases

2. **Data Format Example:**
   ```
   case_id,droplet_diameter_um,T_ambient_K,p_mpa,fuel_type,evap_time_ms,peak_temp_K,emissions_ppm
   1,10,800,5,c7h16,1.2,2050,45
   2,20,800,5,c7h16,2.5,2200,42
   3,50,1200,5,c7h16,5.1,2300,38
   ...
   ```

### Week 3: ML Training Ready
- Then proceed to Step 3 (Normalization)
- All subsequent steps assume you have this tabular format

---

## Current File Analysis

Your current file (`Eulerian_1_-_eulerian1_1.csv`) is:
- ✓ **Useful for:** Understanding CFD output structure, learning Eulerian methods
- ✓ **Good for:** Post-processing examples, feature extraction demonstrations
- ✗ **Not sufficient for:** Direct ML droplet training (need aggregated outputs)

**Bottom line:** This is one piece of a larger dataset. You need to either:
1. Generate/collect more CFD cases, OR
2. Extract aggregated statistics from this field, OR
3. Use external datasets (Sandia)

---

## Quick Reference: Data Summary

```
INPUTS AVAILABLE (from this Eulerian field):
  - Temperature (T):     777.28 - 800.04 K
  - Pressure (p):        4.99 - 5.00 MPa
  - Density (ρ):         21.69 - 23.00 kg/m³
  - Heptane (C7H16):     0 - 0.234
  - Oxygen (O2):         0 - 0.234
  - Nitrogen (N2):       0.766 - 0.766
  - Velocity mag:        0 - 0.08 m/s
  
SPATIAL COVERAGE:
  - X: -0.01 to 0.10 m (110 mm)
  - Y: 0.00 to 0.10 m (100 mm)
  - Z: -0.01 to 0.01 m (20 mm)
  - Grid points: 178,164
  
MISSING (for ML training):
  - Multiple simulation cases
  - Time evolution
  - Droplet-specific outputs
  - Target variables
```

---

## Contact Points

**For Dr. Mishra:**
- "Do we have multiple CFD simulations from different conditions (P, T, fuel, droplet size)?"
- "What are the target outputs from each simulation? (evap time, emissions, etc.)"
- "Can we extract aggregated droplet properties from the Eulerian fields?"

**For Your Research:**
- Start with Step 3 (Normalization) **only after** you have:
  - ✓ Multiple cases (≥50 simulations)
  - ✓ Clear inputs (boundary/initial conditions)
  - ✓ Clear outputs (droplet properties to predict)
  - ✓ Organized in a single CSV file

---

## File Size Note

Your current file is 17 MB and contains ~178K grid points. When you compile multiple CFD cases:
- Each case might add 10-100 MB (depending on resolution)
- With 100 cases: ~1-10 GB total
- Solution: Sample grid points or aggregate regions

---

**Next: Get back to me once you have the dataset format clarified with Dr. Mishra!**
