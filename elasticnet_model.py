#!/usr/bin/env python3
"""
Complete merged ElasticNet OPV PCE prediction script.

This single script combines the four original workflows while keeping the
algorithmic settings unchanged:

1. BEST_CASE
   - Donor Morgan fingerprints
   - Acceptor Morgan fingerprints
   - Invalid SMILES flags
   - FMO descriptors
   - Engineered FMO descriptors
   - Strong CT descriptors only

2. SMILES_CT_ONLY
   - Donor Morgan fingerprints
   - Acceptor Morgan fingerprints
   - Invalid SMILES flags
   - Strong CT descriptors only

3. NO_CT
   - Donor Morgan fingerprints
   - Acceptor Morgan fingerprints
   - Invalid SMILES flags
   - FMO descriptors
   - Engineered FMO descriptors
   - No CT descriptors

4. SMILES_ONLY
   - Donor Morgan fingerprints
   - Acceptor Morgan fingerprints
   - Invalid SMILES flags
   - No FMO, engineered FMO, or CT descriptors

Leakage control:
Feature selection is performed independently inside each outer training fold.
The validation fold is never used for feature selection, scaling, or model fitting.

How to use:
- Set RUN_SCENARIOS = ["ALL"] to run all four scenarios.
- Or set, for example, RUN_SCENARIOS = ["BEST_CASE"] to run one scenario.
"""

import math
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone


# ======================================================================
# User settings
# ======================================================================

RANDOM_STATE = 42
CSV_FILE = "opv_table_modified.csv"

N_BITS = 64
RADIUS = 2
TOP_K_FEATURES = 20

N_SPLITS = 5
N_REPEATS = 20

DONOR_SMILES = "SMILES(donor)"
ACCEPTOR_SMILES = "SMILES(acceptor)"

# Options: ["ALL"] or any subset of:
# ["BEST_CASE", "SMILES_CT_ONLY", "NO_CT", "SMILES_ONLY"]
RUN_SCENARIOS = ["ALL"]


# ======================================================================
# Feature definitions from the original scripts
# ======================================================================

FMO_COLS = [
    "Donor_HOMO(ev)",
    "Donor_LUMO(ev)",
    "Donor_bandgap(ev)",
    "Acceptor_HOMO(ev)",
    "Acceptor_LUMO(ev)",
    "Acceptor_bandgap(ev)",
]

STRONG_CT_COLS = [
    "Excitation energy (ECT)",
    "RMSeh",
    "POS",
    "ECT - E1S",
]

EXCLUDED_CT_COLS = [
    "Excitation energy (ECT)",
    "OSc",
    "ECT - E1S",
    "ECT - ET1",
    "CT",
    "POS",
    "PR",
    "PRNTO",
    "RMSeh",
]

EXCLUDED_ENGINEERED_FMO_COLS = [
    "EffectiveEg_abs_DHOMO_minus_abs_ALUMO",
    "HOMO_offset_DminusA",
    "LUMO_offset_DminusA",
    "Bandgap_difference_DminusA",
]


SCENARIO_CONFIGS = {
    "BEST_CASE": {
        "case_scenario": "BEST_CASE_SCENARIO",
        "title": "BEST CASE SCENARIO",
        "feature_set": "Fingerprints + FMO + engineered FMO + strong CT descriptors",
        "use_fmo": True,
        "use_engineered_fmo": True,
        "use_strong_ct": True,
        "output_prefix": "elasticnet_best_case_scenario",
    },
    "SMILES_CT_ONLY": {
        "case_scenario": "SMILES_CT_ONLY_SCENARIO",
        "title": "SMILES + CT ONLY SCENARIO",
        "feature_set": "SMILES fingerprints + strong CT descriptors only",
        "use_fmo": False,
        "use_engineered_fmo": False,
        "use_strong_ct": True,
        "output_prefix": "elasticnet_smiles_ct_only_scenario",
    },
    "NO_CT": {
        "case_scenario": "NO_CT_SCENARIO",
        "title": "NO CT SCENARIO",
        "feature_set": "Fingerprints + FMO + engineered FMO descriptors only",
        "use_fmo": True,
        "use_engineered_fmo": True,
        "use_strong_ct": False,
        "output_prefix": "elasticnet_no_ct_scenario",
    },
    "SMILES_ONLY": {
        "case_scenario": "SMILES_ONLY_SCENARIO",
        "title": "SMILES-ONLY SCENARIO",
        "feature_set": "Donor/acceptor Morgan fingerprints + invalid SMILES flags only",
        "use_fmo": False,
        "use_engineered_fmo": False,
        "use_strong_ct": False,
        "output_prefix": "elasticnet_smiles_only_scenario",
    },
}


# ======================================================================
# Common helper functions
# ======================================================================

def smiles_to_fp(smiles):
    arr = np.zeros((N_BITS,), dtype=np.float32)

    if pd.isna(smiles):
        return arr, 1

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return arr, 1

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        RADIUS,
        nBits=N_BITS,
    )

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr, 0


def safe_pearsonr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true, y_pred):
    r = safe_pearsonr(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return r, rmse, mae, r2


def check_required_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s) in CSV file:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def build_smiles_features(df):
    donor_fps = []
    acceptor_fps = []
    donor_bad = []
    acceptor_bad = []

    for _, row in df.iterrows():
        d_fp, d_flag = smiles_to_fp(row[DONOR_SMILES])
        a_fp, a_flag = smiles_to_fp(row[ACCEPTOR_SMILES])

        donor_fps.append(d_fp)
        acceptor_fps.append(a_fp)
        donor_bad.append(d_flag)
        acceptor_bad.append(a_flag)

    donor_fps = np.asarray(donor_fps)
    acceptor_fps = np.asarray(acceptor_fps)

    donor_fp_names = [f"donor_fp_{i}" for i in range(N_BITS)]
    acceptor_fp_names = [f"acceptor_fp_{i}" for i in range(N_BITS)]

    X_parts = [
        donor_fps,
        acceptor_fps,
        np.array(donor_bad).reshape(-1, 1),
        np.array(acceptor_bad).reshape(-1, 1),
    ]

    feature_names = (
        donor_fp_names
        + acceptor_fp_names
        + ["donor_invalid_smiles", "acceptor_invalid_smiles"]
    )

    return X_parts, feature_names


def build_numeric_df(df, numeric_cols):
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median())
    return numeric_df


def build_engineered_fmo_features(numeric_df):
    effective_eg = (
        np.abs(numeric_df["Donor_HOMO(ev)"])
        - np.abs(numeric_df["Acceptor_LUMO(ev)"])
    ).values.reshape(-1, 1)

    homo_offset = (
        numeric_df["Donor_HOMO(ev)"]
        - numeric_df["Acceptor_HOMO(ev)"]
    ).values.reshape(-1, 1)

    lumo_offset = (
        numeric_df["Donor_LUMO(ev)"]
        - numeric_df["Acceptor_LUMO(ev)"]
    ).values.reshape(-1, 1)

    bandgap_difference = (
        numeric_df["Donor_bandgap(ev)"]
        - numeric_df["Acceptor_bandgap(ev)"]
    ).values.reshape(-1, 1)

    X_engineered = [
        effective_eg,
        homo_offset,
        lumo_offset,
        bandgap_difference,
    ]

    return X_engineered, EXCLUDED_ENGINEERED_FMO_COLS.copy()


def prepare_data(csv_file, scenario_name):
    config = SCENARIO_CONFIGS[scenario_name]

    df = pd.read_csv(csv_file)
    target_col = df.columns[-1]

    required_cols = [DONOR_SMILES, ACCEPTOR_SMILES, target_col]

    if config["use_fmo"] or config["use_engineered_fmo"]:
        required_cols += FMO_COLS

    if config["use_strong_ct"]:
        required_cols += STRONG_CT_COLS

    check_required_columns(df, required_cols)

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = df[target_col].values.astype(float)

    X_parts, feature_names = build_smiles_features(df)

    numeric_cols = []

    if config["use_fmo"]:
        numeric_cols += FMO_COLS

    if config["use_strong_ct"]:
        numeric_cols += STRONG_CT_COLS

    numeric_df = None

    if numeric_cols:
        numeric_df = build_numeric_df(df, numeric_cols)
        X_parts.append(numeric_df.values)
        feature_names += numeric_cols

    if config["use_engineered_fmo"]:
        # Engineered FMO features must be computed from FMO columns only.
        fmo_numeric_df = build_numeric_df(df, FMO_COLS)
        engineered_parts, engineered_names = build_engineered_fmo_features(
            fmo_numeric_df
        )
        X_parts += engineered_parts
        feature_names += engineered_names

    X = np.hstack(X_parts)

    return X.astype(np.float32), y, feature_names, target_col


def select_features_inside_fold(X_train, y_train, top_k=TOP_K_FEATURES):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=3,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    k = min(top_k, X_train.shape[1])

    return np.argsort(importances)[::-1][:k]


def get_elasticnet_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=np.linspace(0.05, 0.95, 10),
            alphas=np.logspace(-5, 1, 60),
            max_iter=100000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def run_scenario(X, y, feature_names, scenario_name):
    config = SCENARIO_CONFIGS[scenario_name]

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    model = get_elasticnet_model()

    oof_preds = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    fold_rs = []

    feature_counts = {
        name: 0
        for name in feature_names
    }

    n_folds = 0

    for train_idx, test_idx in cv.split(X):
        n_folds += 1

        X_train_full = X[train_idx]
        X_test_full = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        selected_idx = select_features_inside_fold(
            X_train_full,
            y_train,
        )

        for idx in selected_idx:
            feature_counts[feature_names[idx]] += 1

        X_train = X_train_full[:, selected_idx]
        X_test = X_test_full[:, selected_idx]

        m = clone(model)
        m.fit(X_train, y_train)

        preds = m.predict(X_test)

        oof_preds[test_idx] += preds
        counts[test_idx] += 1

        fold_r = safe_pearsonr(y_test, preds)
        fold_rs.append(fold_r)

    final_preds = oof_preds / counts

    r, rmse, mae, r2 = compute_metrics(y, final_preds)

    result_row = {
        "case_scenario": config["case_scenario"],
        "feature_set": config["feature_set"],
        "model": "ElasticNet",
        "n_features_before_selection": X.shape[1],
        "top_k_features": min(TOP_K_FEATURES, X.shape[1]),
        "r": r,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "std_fold_r": float(np.nanstd(fold_rs)),
    }

    if config["use_strong_ct"]:
        result_row["ct_descriptors"] = ", ".join(STRONG_CT_COLS)
    else:
        result_row["ct_descriptors_used"] = "None"
        result_row["ct_descriptors_excluded"] = ", ".join(EXCLUDED_CT_COLS)

    if not config["use_fmo"]:
        result_row["fmo_descriptors_used"] = "None"

    if not config["use_engineered_fmo"]:
        result_row["engineered_fmo_descriptors_used"] = "None"

    results = pd.DataFrame([result_row])

    feature_summary = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "selected_in_outer_folds": list(feature_counts.values()),
        "selected_fraction": [
            v / n_folds
            for v in feature_counts.values()
        ],
    }).sort_values("selected_fraction", ascending=False)

    oof_table = pd.DataFrame({
        "actual_PCE": y,
        "predicted_ElasticNet": final_preds,
        "residual": y - final_preds,
    })

    prefix = config["output_prefix"]

    results.to_csv(
        f"{prefix}_results.csv",
        index=False,
    )

    feature_summary.to_csv(
        f"{prefix}_feature_selection.csv",
        index=False,
    )

    oof_table.to_csv(
        f"{prefix}_oof_predictions.csv",
        index=False,
    )

    return results, feature_summary, oof_table


def print_scenario_report(scenario_name, target_col, n_samples, results, feature_summary):
    config = SCENARIO_CONFIGS[scenario_name]

    print("\n" + "=" * 80)
    print(f"{config['title']} INPUT ALGORITHM")
    print("=" * 80)

    print(f"Target column: {target_col}")
    print(f"Samples: {n_samples}")

    print("\nFeature set used:")
    print("  Morgan donor fingerprints")
    print("  Morgan acceptor fingerprints")
    print("  Donor/acceptor invalid SMILES flags")

    if config["use_fmo"]:
        print("  FMO descriptors")

    if config["use_engineered_fmo"]:
        print("  Engineered FMO descriptors")

    if config["use_strong_ct"]:
        print("  Strong CT descriptors only")

    if config["use_strong_ct"]:
        print("\nStrong CT descriptors used:")
        for col in STRONG_CT_COLS:
            print(f"  {col}")
    else:
        print("\nCT descriptors excluded from training:")
        for col in EXCLUDED_CT_COLS:
            print(f"  {col}")

    if not config["use_fmo"]:
        print("\nFMO descriptors excluded from training:")
        for col in FMO_COLS:
            print(f"  {col}")

    if not config["use_engineered_fmo"]:
        print("\nEngineered FMO descriptors excluded from training:")
        for col in EXCLUDED_ENGINEERED_FMO_COLS:
            print(f"  {col}")

    print("\nAlgorithm:")
    print("  Model: ElasticNetCV")
    print(f"  Fingerprint bits: {N_BITS}")
    print(f"  Morgan radius: {RADIUS}")
    print("  Feature selection: RandomForest inside each outer training fold")
    print(f"  TOP_K_FEATURES: {TOP_K_FEATURES}")
    print(f"  Cross-validation: {N_SPLITS}-fold x {N_REPEATS} repeats")
    print("  Leakage control: feature selection, scaling, and ElasticNetCV fitting")
    print("  are performed only inside each training fold.")

    print("\n" + "=" * 80)
    print(f"{config['title']} RESULTS")
    print("=" * 80)

    print(results.to_string(index=False))

    print("\nTop selected features:")
    print(feature_summary.head(20).to_string(index=False))

    prefix = config["output_prefix"]

    print("\nSaved files:")
    print(f"  {prefix}_results.csv")
    print(f"  {prefix}_feature_selection.csv")
    print(f"  {prefix}_oof_predictions.csv")
    print("=" * 80)


def resolve_scenarios():
    if len(RUN_SCENARIOS) == 1 and RUN_SCENARIOS[0].upper() == "ALL":
        return [
            "BEST_CASE",
            "SMILES_CT_ONLY",
            "NO_CT",
            "SMILES_ONLY",
        ]

    scenarios = [s.upper() for s in RUN_SCENARIOS]

    invalid = [s for s in scenarios if s not in SCENARIO_CONFIGS]
    if invalid:
        raise ValueError(
            "Invalid scenario name(s): "
            + ", ".join(invalid)
            + "\nValid options are: ALL, "
            + ", ".join(SCENARIO_CONFIGS.keys())
        )

    return scenarios


def main():
    scenarios = resolve_scenarios()

    all_results = []

    for scenario_name in scenarios:
        X, y, feature_names, target_col = prepare_data(
            CSV_FILE,
            scenario_name,
        )

        results, feature_summary, _ = run_scenario(
            X,
            y,
            feature_names,
            scenario_name,
        )

        print_scenario_report(
            scenario_name,
            target_col,
            len(y),
            results,
            feature_summary,
        )

        all_results.append(results)

    if len(all_results) > 1:
        comparison = pd.concat(all_results, ignore_index=True)
        comparison.to_csv(
            "elasticnet_all_scenarios_comparison_results.csv",
            index=False,
        )

        print("\n" + "=" * 80)
        print("ALL SCENARIOS SUMMARY")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("\nSaved file:")
        print("  elasticnet_all_scenarios_comparison_results.csv")
        print("=" * 80)


if __name__ == "__main__":
    main()
