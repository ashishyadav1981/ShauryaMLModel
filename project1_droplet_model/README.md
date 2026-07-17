# Droplet Evaporation Surrogate Model

A neural network surrogate model that predicts liquid droplet evaporation behavior — final diameter, evaporation time, and emissions proxy — from CFD-simulated spray combustion data.

## Overview

High-fidelity CFD simulation of droplet evaporation in combustion environments is computationally expensive. This project trains a neural network **surrogate model**: once trained on a set of CFD-generated examples, it predicts droplet outcomes in milliseconds instead of requiring a full simulation re-run.

The model was trained on **792 CFD-generated samples** representing single-droplet evaporation under a fixed high-pressure, high-temperature combustion condition (800 K, 5,000,000 Pa), with initial droplet diameter, position, and velocity as the varying inputs.

## Results

- **Test set R² = 0.9974** across all three predicted outputs (diameter, evaporation time, emissions proxy)
- Model architecture: 10 → 128 → 64 → 32 → 3 fully connected network, 22,915 trainable parameters

## Scope and Limitations

This model is trained and validated entirely on CFD-simulated data under a **single fixed combustion condition** (constant temperature and pressure). It has not been validated against experimental combustion data, and should not be assumed to generalize to combustion regimes outside the conditions it was trained on — for example, atmospheric-pressure open flames, varying temperature fields, or gas-phase-only combustion (no liquid droplets), all of which involve substantially different physics.

An earlier attempt to validate this model against the Sandia Flame D experimental dataset was abandoned after review: Flame D is a purely gaseous methane-air jet flame with no liquid droplet injection, so it has no droplet-related ground truth to validate against, and its operating conditions (atmospheric pressure, wide temperature range, gas velocities up to 25 m/s) fall well outside this model's training domain. That comparison is not included in these results. A separate model, trained specifically on Sandia Flame D's gas-phase measurements, is a distinct project.

## Approach

- **Framework:** PyTorch
- **Supporting libraries:** scikit-learn, pandas, numpy, matplotlib, scipy
- **Pipeline:**
  1. `path2_feature_extraction.py` — extract droplet trajectory features from raw Eulerian CFD data
  2. `step3_normalization.py` — normalize features and targets
  3. `step4_data_split.py` — train/validation/test split
  4. `step5_neural_network.py` — model architecture definition
  5. `step6_training_loop.py` — training
  6. `step7_evaluate.py` — test set evaluation (R² = 0.9974)

## Inputs and Outputs

**Inputs (10 features):** initial droplet diameter (μm), time (ms), temperature (K), pressure (Pa), fuel species fraction, velocity magnitude (m/s), distance from inlet (m), radial distance (m), and related derived features.

**Outputs (3 targets):**
- Final diameter (μm)
- Evaporation time (ms)
- Emissions proxy

## Repository Structure

```
project1_droplet_model/
├── path2_feature_extraction.py
├── step3_normalization.py
├── step4_data_split.py
├── step5_neural_network.py
├── step6_training_loop.py
├── step7_evaluate.py
├── explore_your_data.py
├── visualize_your_data.py
├── Eulerian_1.csv              # raw CFD input data
├── droplet_features.csv        # extracted features (X)
├── droplet_targets.csv         # extracted targets (y)
├── X_train.csv / X_val.csv / X_test.csv
├── y_train.csv / y_val.csv / y_test.csv
├── best_model.pth              # trained model weights
├── scaler_X.pkl / scaler_y.pkl # fitted normalization scalers
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Usage

```bash
python path2_feature_extraction.py
python step3_normalization.py
python step4_data_split.py
python step5_neural_network.py
python step6_training_loop.py
python step7_evaluate.py
```

## Author

Shaurya — [add contact/LinkedIn/GitHub profile link here]

## License

MIT
