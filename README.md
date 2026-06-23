# Machine Learning Framework for Predicting Organic Photovoltaic (OPV) Power Conversion Efficiency Using SMILES, Frontier Molecular Orbital Descriptors, and Charge-Transfer Features

## Overview

This repository contains a complete machine-learning workflow for predicting the **Power Conversion Efficiency (PCE)** of donor–acceptor organic photovoltaic (OPV) systems using:

* Molecular fingerprints derived from donor and acceptor SMILES strings
* Frontier Molecular Orbital (FMO) descriptors
* Charge-transfer (CT) descriptors obtained from quantum chemical calculations
* Nested feature selection
* Leakage-safe model evaluation
* Y-randomization validation
* Charge-transfer state identification from ORCA outputs

The repository includes both **ElasticNet** and **XGBoost** implementations together with rigorous validation protocols suitable for publication-quality studies.

---

# Repository Structure

```text
.
├── elasticnet_model.py
├── xgboost_model.py
├── elasticnet_Y_randomization.py
├── xgboost_Y_randomization.py
├── CT_detection_from_fragments.py
├── data/
│   └── opv_table_modified.csv
└── outputs/
```

---

# Scientific Objective

The objective is to investigate whether physically meaningful descriptors derived from:

* Donor and acceptor molecular structures
* Frontier molecular orbitals
* Charge-transfer characteristics

can improve machine-learning prediction of OPV device efficiency compared with molecular fingerprints alone.

---

# Features Used

## 1. SMILES Fingerprints

Donor and acceptor structures are converted into:

* Morgan fingerprints
* Radius = 2
* Configurable fingerprint length

Typical values:

```python
N_BITS = 64
N_BITS = 512
N_BITS = 1024
N_BITS = 2048
```

Additional binary indicators:

```text
donor_invalid_smiles
acceptor_invalid_smiles
```

---

## 2. Frontier Molecular Orbital (FMO) Features

```text
Donor_HOMO(ev)
Donor_LUMO(ev)
Donor_bandgap(ev)

Acceptor_HOMO(ev)
Acceptor_LUMO(ev)
Acceptor_bandgap(ev)
```

---

## 3. Engineered FMO Features

Derived descriptors:

```text
EffectiveEg_abs_DHOMO_minus_abs_ALUMO
HOMO_offset_DminusA
LUMO_offset_DminusA
Bandgap_difference_DminusA
```

These descriptors encode donor–acceptor energetic alignment.

---

## 4. Charge Transfer (CT) Descriptors

Examples include:

```text
Excitation energy (ECT)
OSc
ECT - E1S
ECT - ET1
CT
POS
PR
PRNTO
Z_HE
RMSeh
```

Some workflows exclude:

```text
COH
```

to investigate its influence on prediction performance.

---

# ElasticNet Workflow

## Model

```text
ElasticNetCV
```

with:

* Automatic α optimization
* Automatic L1/L2 ratio optimization
* StandardScaler
* Nested feature selection

---

## Cross Validation

Repeated out-of-fold evaluation:

```text
5-fold CV
×
20 repeats
=
100 validation folds
```

Every sample is predicted multiple times and predictions are averaged.

This substantially reduces variance compared with a single CV split.

---

## Leakage Prevention

Feature selection is performed only on:

```text
Training Fold
```

Validation samples are never used for:

* Feature selection
* Scaling
* Hyperparameter tuning
* Model fitting

Workflow:

```text
Outer Fold
│
├── Training Set
│   ├── Feature Selection
│   ├── StandardScaler Fit
│   ├── ElasticNetCV
│   └── Model Training
│
└── Validation Set
    └── Prediction Only
```

This prevents information leakage.

---

## ElasticNet Scenarios

### BEST_CASE

Features:

```text
SMILES
+
FMO
+
Engineered FMO
+
Strong CT descriptors
```

---

### SMILES_CT_ONLY

Features:

```text
SMILES
+
Strong CT descriptors
```

---

### NO_CT

Features:

```text
SMILES
+
FMO
+
Engineered FMO
```

---

### SMILES_ONLY

Features:

```text
SMILES fingerprints only
```

---

# XGBoost Workflow

## Model

```text
XGBRegressor
```

Objective:

```text
reg:squarederror
```

Training:

```text
tree_method = "hist"
```

for computational efficiency.

---

## Hyperparameter Optimization

Grid search is performed inside each training fold.

Example parameters:

```text
n_estimators
max_depth
learning_rate
gamma
subsample
colsample_bytree
```

Only training data are used for optimization.

---

## Ensemble Prediction

For each optimized parameter set:

```text
50 independent XGBoost models
```

are trained using different random seeds.

Predictions are averaged:

```text
Final Prediction
=
Mean of Ensemble Predictions
```

Benefits:

* Improved stability
* Reduced variance
* More robust generalization

---

## Nested Feature Selection

Feature importance is computed on the training fold only.

Top-ranked features are retained:

```text
TOP_K = 20
```

before model fitting.

---

# Y-Randomization Validation

## Purpose

Demonstrates that model performance is not obtained by chance.

---

## Procedure

### Step 1

Train model using true labels.

```text
X → y
```

---

### Step 2

Randomly shuffle labels.

```text
X → random(y)
```

---

### Step 3

Repeat complete workflow:

* Feature selection
* Scaling
* Hyperparameter optimization
* Model fitting
* Cross-validation

---

### Step 4

Repeat many times.

Typical:

```text
50–100 permutations
```

---

## Statistical Validation

Empirical p-values are computed:

```text
p(r)
p(R²)
p(RMSE)
p(MAE)
```

Comparing:

```text
True Model
vs
Randomized Models
```

---

## Reviewer Interpretation

A publishable model typically exhibits:

```text
True r  >> Randomized r

True R² >> Randomized R²

True RMSE << Randomized RMSE

True MAE << Randomized MAE
```

indicating genuine structure–property relationships.

---

# Charge Transfer State Detection

## Purpose

Identify excited states possessing charge-transfer character from ORCA output files.

---

## Input

ORCA TD-DFT output:

```text
*.out
```

containing:

```text
MULLIKEN ATOMIC CHARGES
```

for ground and excited states.

---

## Methodology

For each excited state:

### Ground State

Fragment charges:

```text
Qground(fragment)
```

---

### Excited State

Fragment charges:

```text
Qexcited(fragment)
```

---

### Charge Transfer Magnitude

```text
ΔQ = Qexcited − Qground
```

---

### CT Strength

```text
CT_strength
=
0.5 × (|ΔQ1| + |ΔQ2|)
```

---

### CT Criterion

A state is labeled CT when:

```text
|ΔQ| ≥ threshold
```

and charge transfer occurs between fragments.

---

## Output

For every state:

```text
State
Fragment Charges
Charge Difference
CT Strength
CT Flag
```

CSV export is supported.

---

# Output Files

## ElasticNet

```text
elasticnet_*_results.csv
elasticnet_*_feature_selection.csv
elasticnet_*_oof_predictions.csv
```

---

## XGBoost

```text
*_summary_metrics.csv
*_foldwise_metrics.csv
*_selected_features_by_fold.csv
*_oof_predictions.csv
```

---

## Y-Randomization

```text
*_y_randomization_results.csv
*_y_randomization_summary.csv
*_true_vs_y_randomization_comparison.csv
*_y_randomization_report.txt
```

---

# Evaluation Metrics

Reported metrics:

```text
r
R²
RMSE
MAE
MAPE
```

where:

* r = Pearson correlation coefficient
* R² = coefficient of determination
* RMSE = root mean square error
* MAE = mean absolute error
* MAPE = mean absolute percentage error

---

# Reproducibility

All workflows use fixed random seeds:

```python
RANDOM_STATE = 42
```

ensuring reproducibility.

---

# Dependencies

```bash
pip install numpy pandas scikit-learn xgboost rdkit-pypi
```

Additional packages:

```bash
pip install scipy
```

---

# Recommended Citation

If you use this workflow in academic research, please cite:

* Elastic Net:
  Zou, H.; Hastie, T. *Journal of the Royal Statistical Society B* **2005**, 67, 301–320.

* XGBoost:
  Chen, T.; Guestrin, C. *KDD* **2016**, 785–794.

* RDKit:
  Landrum, G. RDKit: Open-source cheminformatics.

---

# Key Methodological Strengths

✅ Leakage-safe nested feature selection

✅ Repeated out-of-fold validation

✅ Ensemble learning

✅ Hyperparameter optimization inside training folds

✅ Y-randomization statistical validation

✅ Integration of quantum-chemical CT descriptors

✅ Physically interpretable molecular features

✅ Publication-oriented workflow suitable for small OPV datasets

---

# Author Notes

This repository was developed for machine-learning-assisted prediction of OPV power conversion efficiency and for evaluating the contribution of charge-transfer descriptors to photovoltaic performance prediction.

The workflow combines cheminformatics, quantum chemistry, and interpretable machine learning in a fully reproducible framework.
