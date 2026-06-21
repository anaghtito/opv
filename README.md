# OPV Machine Learning and Charge-Transfer Analysis Toolkit

This repository contains three Python scripts for analyzing organic photovoltaic (OPV) donor–acceptor systems:

1. **`xgboost_model.py`** — XGBoost-based PCE prediction across multiple feature scenarios.
2. **`elasticnet_model.py`** — ElasticNet-based PCE prediction across multiple feature scenarios.
3. **`CT_detection_from_fragments.py`** — Charge-transfer state detection from ORCA Mulliken population outputs.

The workflow is designed for small OPV datasets where molecular structure, frontier molecular orbital descriptors, and charge-transfer descriptors are used to predict power conversion efficiency (PCE).

---

## Repository Structure

```text
.
├── xgboost_model.py
├── elasticnet_model.py
├── CT_detection_from_fragments.py
├── opv_table_CT_features_new.csv          # Required for ML scripts
└── orca.out                          # Default ORCA output file for CT detection
```

---

## Code 1: `xgboost_model.py`

### Purpose

Runs XGBoost regression models to predict OPV **PCE(%)** using different combinations of SMILES fingerprints, FMO descriptors, and CT descriptors.

### Main Features

- Converts donor and acceptor SMILES into Morgan fingerprints using RDKit.
- Uses XGBoost regression with inner hyperparameter tuning.
- Performs leakage-safe nested feature selection.
- Evaluates multiple feature scenarios.
- Saves out-of-fold predictions, foldwise metrics, selected features, and summary metrics.

### Feature Scenarios

| Scenario | Features Used |
|---|---|
| `CT_No_COH` | SMILES fingerprints + CT descriptors without COH |
| `SMILES_FMO_CT` | SMILES fingerprints + FMO + CT descriptors |
| `SMILES_FMO` | SMILES fingerprints + FMO descriptors |
| `SMILES_ONLY` | Donor/acceptor Morgan fingerprints only |

### Important Settings

```python
CSV_FILE = "opv_table_modified.csv"
TARGET_COL = "PCE(%)"
FP_RADIUS = 2
FP_NBITS = 2048
N_SPLITS = 8
N_ENSEMBLE_MODELS = 50
TOP_K = 20
```

### Outputs

For each scenario, the script writes files such as:

```text
<scenario>_summary_metrics.csv
<scenario>_oof_predictions.csv
<scenario>_foldwise_metrics.csv
<scenario>_selected_features_by_fold.csv
<scenario>_numeric_feature_selection_frequency.csv
```

It also writes a combined comparison file:

```text
all_xgboost_scenarios_comparison_metrics.csv
```

---

## Code 2: `elasticnet_model.py`

### Purpose

Runs ElasticNet regression models to predict OPV **PCE(%)** using four feature-set scenarios. This script is especially useful for small datasets because ElasticNet combines L1 and L2 regularization.

### Main Features

- Converts donor and acceptor SMILES into Morgan fingerprints.
- Includes invalid SMILES flags as model features.
- Supports FMO, engineered FMO, and strong CT descriptor scenarios.
- Performs feature selection inside each training fold using Random Forest importance.
- Uses `ElasticNetCV` with `StandardScaler` inside a scikit-learn pipeline.
- Uses repeated cross-validation for robust out-of-fold prediction.

### Feature Scenarios

| Scenario | Features Used |
|---|---|
| `BEST_CASE` | SMILES fingerprints + FMO + engineered FMO + strong CT descriptors |
| `SMILES_CT_ONLY` | SMILES fingerprints + strong CT descriptors only |
| `NO_CT` | SMILES fingerprints + FMO + engineered FMO only |
| `SMILES_ONLY` | SMILES fingerprints only |

### Important Settings

```python
CSV_FILE = "opv_table_modified.csv"
N_BITS = 64
RADIUS = 2
TOP_K_FEATURES = 20
N_SPLITS = 5
N_REPEATS = 20
RUN_SCENARIOS = ["ALL"]
```

### Strong CT Descriptors Used

```text
Excitation energy (ECT)
RMSeh
POS
ECT - E1S
```

### Outputs

For each scenario, the script writes:

```text
<scenario>_results.csv
<scenario>_feature_selection.csv
<scenario>_oof_predictions.csv
```

When all scenarios are run, it also writes:

```text
elasticnet_all_scenarios_comparison_results.csv
```

---

## Code 3: `CT_detection_from_fragments.py`

### Purpose

Detects charge-transfer excited states from ORCA output files by comparing ground-state and excited-state Mulliken fragment charges.

The script supports both singlet and triplet excited-state Mulliken population sections.

### Main Features

- Parses ground-state Mulliken atomic charges.
- Parses excited-state unrelaxed CIS/TDA Mulliken charges.
- Computes fragment charges for two user-defined molecular fragments.
- Calculates change in charge relative to the ground state.
- Flags a state as CT if fragment charge changes exceed a threshold and have opposite signs.
- Optionally saves results to CSV.

### Default Settings

```python
DEFAULT_INPUT_FILE = "45.out"
DEFAULT_FRAG1_START = 0
DEFAULT_FRAG1_END_INCL = 179
DEFAULT_FRAG2_START = 180
DEFAULT_FRAG2_END_INCL = 481
DEFAULT_CT_THRESHOLD = 0.1
```

### Example Usage

Run with default settings:

```bash
python CT_detection_from_fragments.py
```

Run with a custom ORCA output file:

```bash
python CT_detection_from_fragments.py -i my_orca_output.out
```

Run with custom fragment ranges and save CSV:

```bash
python CT_detection_from_fragments.py \
  -i my_orca_output.out \
  --frag1-start 0 \
  --frag1-end 179 \
  --frag2-start 180 \
  --frag2-end 481 \
  -t 0.10 \
  --csv ct_results.csv
```

### Output Columns

If CSV output is requested, the result contains:

```text
state
frag1_q
frag2_q
delta_q_frag1
delta_q_frag2
ct_strength
ct_flag
threshold
ground_frag1_q
ground_frag2_q
frag1_start
frag1_end_incl
frag2_start
frag2_end_incl
```

---

## Required Input Data

### For ML Scripts

Both `xgboost_model.py` and `elasticnet_model.py` expect a CSV file named:

```text
opv_table_modified.csv
```

The dataset should contain:

```text
SMILES(donor)
SMILES(acceptor)
PCE(%)
```

Additional columns are required depending on the scenario.

### FMO Descriptor Columns

```text
Donor_HOMO(ev)
Donor_LUMO(ev)
Donor_bandgap(ev)
Acceptor_HOMO(ev)
Acceptor_LUMO(ev)
Acceptor_bandgap(ev)
```

### CT Descriptor Columns

Depending on the script/scenario, the following CT descriptors may be used:

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

The ElasticNet best-case pipeline uses only the strongest selected CT descriptors:

```text
Excitation energy (ECT)
RMSeh
POS
ECT - E1S
```

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy pandas scikit-learn xgboost rdkit
```

If `rdkit` cannot be installed through `pip` on your system, install it with conda:

```bash
conda install -c conda-forge rdkit
```

---

## Running the Scripts

### Run XGBoost Models

```bash
python xgboost_model.py
```

### Run ElasticNet Models

```bash
python elasticnet_model.py
```

To run only selected ElasticNet scenarios, edit:

```python
RUN_SCENARIOS = ["BEST_CASE"]
```

or for multiple scenarios:

```python
RUN_SCENARIOS = ["BEST_CASE", "NO_CT"]
```

### Run CT Detection

```bash
python CT_detection_from_fragments.py -i 45.out --csv ct_results.csv
```

---

## Methodological Notes

### Leakage Control

Both ML scripts are designed to avoid data leakage. Feature selection is performed only inside each outer training fold, so the validation fold is not used for feature selection, scaling, or model fitting.

### Out-of-Fold Predictions

The ML scripts generate out-of-fold predictions. This means each prediction is made for a sample when that sample is in the validation fold, not when it is part of the training fold.

### Model Comparison

The scripts allow comparison of models trained with and without CT descriptors. This is useful for testing whether CT descriptors improve PCE prediction beyond SMILES and FMO information.

---

## Typical Workflow

```text
1. Prepare OPV dataset as opv_table_modified.csv
2. Run CT_detection_from_fragments.py on ORCA output files if CT labels/features are needed
3. Add CT descriptors to the dataset
4. Run elasticnet_model.py for regularized linear modeling
5. Run xgboost_model.py for nonlinear tree-based modeling
6. Compare OOF metrics and selected features across feature scenarios
```

---

## Recommended Citation/Reporting Details

When reporting results from these scripts, include:

- Dataset size
- Target column name: `PCE(%)`
- Fingerprint type: Morgan fingerprints
- Fingerprint radius
- Fingerprint bit size
- Feature groups used
- Cross-validation scheme
- Whether feature selection was nested inside training folds
- Metrics: Pearson `r`, `R²`, RMSE, MAE
- Whether predictions are out-of-fold predictions

---

## License



```text

```

---

## Contact

For questions, issues, or improvements, open a GitHub issue or contact the repository maintainer.
