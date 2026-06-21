#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL SETTINGS
# ============================================================
CSV_FILE = "opv_table_modified.csv"
FALLBACK_CSV_FILE = "opv_table_modified(6).csv"

TARGET_COL = "PCE(%)"
DONOR_SMILES_COL = "SMILES(donor)"
ACCEPTOR_SMILES_COL = "SMILES(acceptor)"

FP_RADIUS = 2
FP_NBITS = 2048
RANDOM_SEED = 42
N_SPLITS = 8
N_ENSEMBLE_MODELS = 50
TOP_K = 20

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 4],
    "learning_rate": [0.03, 0.05, 0.08],
    "gamma": [0.0, 0.1],
    "colsample_bytree": [0.7],
    "min_child_weight": [1],
    "subsample": [0.7],
}

# ============================================================
# FEATURE SETS
# ============================================================
FMO_FEATURES = [
    "Donor_HOMO(ev)",
    "Donor_LUMO(ev)",
    "Donor_bandgap(ev)",
    "Acceptor_HOMO(ev)",
    "Acceptor_LUMO(ev)",
    "Acceptor_bandgap(ev)",
]

CT_FEATURES_NO_COH = [
    "Excitation energy (ECT)",
    "OSc",
    "ECT - E1S",
    "ECT - ET1",
    "CT",
    "POS",
    "PR",
    "PRNTO",
    "Z_HE",
    "RMSeh",
]

# This matches the uploaded SMILES+FMO+CT script, where COH is not included.
CT_FEATURES = CT_FEATURES_NO_COH.copy()


@dataclass(frozen=True)
class Scenario:
    model_label: str
    numeric_features: List[str]
    use_smiles: bool
    run_before_feature_selection: bool
    output_prefix: str
    description: str


SCENARIOS = [
    Scenario(
        model_label="CT_No_COH",
        numeric_features=CT_FEATURES_NO_COH,
        use_smiles=True,
        run_before_feature_selection=False,
        output_prefix="ct_no_coh",
        description="CT feature set with COH removed + donor/acceptor Morgan fingerprints",
    ),
    Scenario(
        model_label="SMILES_FMO_CT",
        numeric_features=FMO_FEATURES + CT_FEATURES,
        use_smiles=True,
        run_before_feature_selection=False,
        output_prefix="smiles_fmo_ct",
        description="SMILES + FMO + CT feature set",
    ),
    Scenario(
        model_label="SMILES_FMO",
        numeric_features=FMO_FEATURES,
        use_smiles=True,
        run_before_feature_selection=True,
        output_prefix="smiles_fmo",
        description="SMILES + FMO feature set",
    ),
    Scenario(
        model_label="SMILES_ONLY",
        numeric_features=[],
        use_smiles=True,
        run_before_feature_selection=True,
        output_prefix="smiles_only",
        description="SMILES-only donor/acceptor Morgan fingerprints",
    ),
]

# ============================================================
# METRICS
# ============================================================
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred, eps=1e-8):
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def collect_metrics(model_label, stage, y_true, y_pred):
    return {
        "model": model_label,
        "stage": stage,
        "n_samples": int(len(y_true)),
        "n_splits": int(N_SPLITS),
        "top_k": int(TOP_K),
        "fp_nbits": int(FP_NBITS),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "r": pearson_r(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def print_metrics(title, y_true, y_pred):
    metrics = collect_metrics("", title, y_true, y_pred)
    print(f"\n{title}")
    print(f"r    = {metrics['r']:.4f}")
    print(f"R²   = {metrics['R2']:.4f}")
    print(f"RMSE = {metrics['RMSE']:.4f}")
    print(f"MAE  = {metrics['MAE']:.4f}")

# ============================================================
# SMILES TO MORGAN FINGERPRINT
# ============================================================
def smiles_to_morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_NBITS):
    arr = np.zeros((n_bits,), dtype=np.float32)

    if pd.isna(smiles):
        return arr

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return arr

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ============================================================
# BUILD FEATURE MATRIX
# ============================================================
def build_feature_matrix(df: pd.DataFrame, scenario: Scenario):
    required_cols = list(scenario.numeric_features) + [TARGET_COL]

    if scenario.use_smiles:
        required_cols.extend([DONOR_SMILES_COL, ACCEPTOR_SMILES_COL])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{scenario.model_label}] Missing required columns: {missing}")

    blocks = []
    feature_names = []

    if scenario.numeric_features:
        numeric_df = df[scenario.numeric_features].copy()
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
        blocks.append(numeric_df.to_numpy(dtype=np.float32))
        feature_names.extend(scenario.numeric_features)

    if scenario.use_smiles:
        donor_fps = []
        acceptor_fps = []
        donor_valid = []
        acceptor_valid = []

        for _, row in df.iterrows():
            d = row[DONOR_SMILES_COL]
            a = row[ACCEPTOR_SMILES_COL]

            dmol = Chem.MolFromSmiles(str(d).strip()) if pd.notna(d) else None
            amol = Chem.MolFromSmiles(str(a).strip()) if pd.notna(a) else None

            donor_valid.append(0.0 if dmol is None else 1.0)
            acceptor_valid.append(0.0 if amol is None else 1.0)

            donor_fps.append(smiles_to_morgan_fp(d))
            acceptor_fps.append(smiles_to_morgan_fp(a))

        blocks.append(np.vstack(donor_fps))
        blocks.append(np.vstack(acceptor_fps))
        blocks.append(np.column_stack([donor_valid, acceptor_valid]).astype(np.float32))

        feature_names.extend([f"Donor_FP_{i}" for i in range(FP_NBITS)])
        feature_names.extend([f"Acceptor_FP_{i}" for i in range(FP_NBITS)])
        feature_names.extend(["Donor_Valid", "Acceptor_Valid"])

    X = np.hstack(blocks)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)

    return X[mask], y[mask], feature_names

# ============================================================
# MODEL
# ============================================================
def make_xgb_model(params, seed):
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        **params,
    )

# ============================================================
# INNER HYPERPARAMETER TUNING
# ============================================================
def fit_best_params(X, y, seed):
    Xtr, Xval, ytr, yval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    best_params = None
    best_rmse = float("inf")
    keys = list(PARAM_GRID.keys())

    for values in product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))
        model = make_xgb_model(params, seed)
        model.fit(Xtr, ytr)

        pred = model.predict(Xval)
        score = rmse(yval, pred)

        if score < best_rmse:
            best_rmse = score
            best_params = params

    return best_params

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================
def ensemble_predict(Xtr, ytr, Xte, params, seed):
    preds = []

    for i in range(N_ENSEMBLE_MODELS):
        model = make_xgb_model(params, seed + i)
        model.fit(Xtr, ytr)
        preds.append(model.predict(Xte))

    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def get_feature_importance(X, y, seed):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model.feature_importances_

# ============================================================
# CV WITHOUT FEATURE SELECTION
# ============================================================
def run_cv_no_feature_selection(X, y, scenario: Scenario):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    preds = np.zeros_like(y)
    stds = np.zeros_like(y)
    fold_rows = []

    for i, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        params = fit_best_params(Xtr, ytr, RANDOM_SEED + i)
        pred, std = ensemble_predict(
            Xtr,
            ytr,
            Xte,
            params,
            RANDOM_SEED + 1000 * i,
        )

        preds[te] = pred
        stds[te] = std

        fold_rows.append({
            "model": scenario.model_label,
            "stage": "before_feature_selection",
            "fold": i,
            "n_validation_samples": len(te),
            "RMSE": rmse(yte, pred),
            "MAE": mae(yte, pred),
            "R2": r2_score(yte, pred),
            "r": pearson_r(yte, pred),
            "best_params": str(params),
        })

        print(
            f"[{scenario.model_label} | Fold {i} | all features] "
            f"r={pearson_r(yte, pred):.3f}  "
            f"R²={r2_score(yte, pred):.3f}  "
            f"RMSE={rmse(yte, pred):.3f}  "
            f"MAE={mae(yte, pred):.3f}"
        )

    print_metrics(f"{scenario.model_label} FINAL BEFORE FEATURE SELECTION:", y, preds)
    return preds, stds, pd.DataFrame(fold_rows)

# ============================================================
# LEAKAGE-SAFE NESTED FEATURE-SELECTION CV
# ============================================================
def run_cv_nested_feature_selection(X, y, feature_names, scenario: Scenario, top_k=TOP_K):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    preds = np.zeros_like(y)
    pred_stds = np.zeros_like(y)

    fold_rows = []
    selected_feature_rows = []

    k_eff = min(top_k, X.shape[1])

    for i, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # Feature selection fitted only on outer training fold.
        importances = get_feature_importance(
            Xtr,
            ytr,
            RANDOM_SEED + 100 * i,
        )

        idx = np.argsort(importances)[::-1][:k_eff]

        Xtr_sel = Xtr[:, idx]
        Xte_sel = Xte[:, idx]

        # Hyperparameter tuning uses only selected training-fold data.
        params = fit_best_params(
            Xtr_sel,
            ytr,
            RANDOM_SEED + i,
        )

        pred, std = ensemble_predict(
            Xtr_sel,
            ytr,
            Xte_sel,
            params,
            RANDOM_SEED + 1000 * i,
        )

        preds[te] = pred
        pred_stds[te] = std

        fold_rmse = rmse(yte, pred)
        fold_mae = mae(yte, pred)
        fold_r2 = r2_score(yte, pred)
        fold_r = pearson_r(yte, pred)

        fold_rows.append({
            "model": scenario.model_label,
            "stage": "nested_feature_selection",
            "fold": i,
            "n_validation_samples": len(te),
            "RMSE": fold_rmse,
            "MAE": fold_mae,
            "R2": fold_r2,
            "r": fold_r,
            "selected_features": len(idx),
            "best_params": str(params),
        })

        for rank, j in enumerate(idx, 1):
            selected_feature_rows.append({
                "model": scenario.model_label,
                "fold": i,
                "rank": rank,
                "feature_index": int(j),
                "feature_name": feature_names[j],
                "importance": float(importances[j]),
            })

        print(
            f"[{scenario.model_label} | Fold {i} | nested FS] "
            f"RMSE={fold_rmse:.3f}  "
            f"MAE={fold_mae:.3f}  "
            f"R²={fold_r2:.3f}  "
            f"r={fold_r:.3f}  "
            f"selected_features={len(idx)}"
        )

    print_metrics(f"{scenario.model_label} FINAL NESTED FEATURE-SELECTION CV:", y, preds)

    oof_df = pd.DataFrame({
        "model": scenario.model_label,
        "actual_PCE": y,
        "predicted_PCE_oof": preds,
        "prediction_std": pred_stds,
        "residual": y - preds,
    })

    fold_df = pd.DataFrame(fold_rows)
    selected_df = pd.DataFrame(selected_feature_rows)

    metrics = collect_metrics(
        scenario.model_label,
        "nested_feature_selection",
        y,
        preds,
    )

    return metrics, oof_df, fold_df, selected_df

# ============================================================
# SAVE HELPERS
# ============================================================
def save_before_outputs(scenario: Scenario, y, preds, stds, fold_df):
    before_df = pd.DataFrame({
        "model": scenario.model_label,
        "actual_PCE": y,
        "predicted_PCE_oof": preds,
        "prediction_std": stds,
        "residual": y - preds,
    })

    before_df.to_csv(f"{scenario.output_prefix}_predictions_before_feature_selection.csv", index=False)
    fold_df.to_csv(f"{scenario.output_prefix}_foldwise_metrics_before_feature_selection.csv", index=False)


def save_nested_outputs(scenario: Scenario, metrics, oof_df, fold_df, selected_df):
    metrics_df = pd.DataFrame([metrics])

    selected_numeric_df = selected_df[
        selected_df["feature_name"].isin(scenario.numeric_features)
    ].copy()

    if not selected_numeric_df.empty:
        numeric_frequency = (
            selected_numeric_df
            .groupby("feature_name")
            .size()
            .reset_index(name=f"selection_frequency_out_of_{N_SPLITS}")
            .sort_values(f"selection_frequency_out_of_{N_SPLITS}", ascending=False)
        )
    else:
        numeric_frequency = pd.DataFrame(
            columns=["feature_name", f"selection_frequency_out_of_{N_SPLITS}"]
        )

    metrics_df.to_csv(f"{scenario.output_prefix}_summary_metrics.csv", index=False)
    oof_df.to_csv(f"{scenario.output_prefix}_oof_predictions.csv", index=False)
    fold_df.to_csv(f"{scenario.output_prefix}_foldwise_metrics.csv", index=False)
    selected_df.to_csv(f"{scenario.output_prefix}_selected_features_by_fold.csv", index=False)
    numeric_frequency.to_csv(f"{scenario.output_prefix}_numeric_feature_selection_frequency.csv", index=False)

    print("\nSaved files:")
    print(f"  - {scenario.output_prefix}_summary_metrics.csv")
    print(f"  - {scenario.output_prefix}_oof_predictions.csv")
    print(f"  - {scenario.output_prefix}_foldwise_metrics.csv")
    print(f"  - {scenario.output_prefix}_selected_features_by_fold.csv")
    print(f"  - {scenario.output_prefix}_numeric_feature_selection_frequency.csv")

# ============================================================
# RUN ONE SCENARIO
# ============================================================
def run_scenario(df: pd.DataFrame, scenario: Scenario):
    print("\n" + "=" * 80)
    print(f"RUNNING SCENARIO: {scenario.model_label}")
    print(scenario.description)
    print("=" * 80)

    print("\nUsing numeric features:")
    if scenario.numeric_features:
        for f in scenario.numeric_features:
            print(f"  - {f}")
    else:
        print("  - None")

    if scenario.use_smiles:
        print(f"Using donor/acceptor Morgan fingerprints: radius={FP_RADIUS}, nBits={FP_NBITS}")

    X, y, feature_names = build_feature_matrix(df, scenario)

    print(f"\nNumber of samples: {len(y)}")
    print(f"Number of total features before selection: {X.shape[1]}")
    print(f"Number of selected features per fold: {TOP_K}")

    scenario_metrics = []

    if scenario.run_before_feature_selection:
        print("\n=== BEFORE FEATURE SELECTION ===")
        preds_before, stds_before, fold_before_df = run_cv_no_feature_selection(X, y, scenario)
        save_before_outputs(scenario, y, preds_before, stds_before, fold_before_df)
        scenario_metrics.append(
            collect_metrics(
                scenario.model_label,
                "before_feature_selection",
                y,
                preds_before,
            )
        )

    print(f"\n=== AFTER NESTED FEATURE SELECTION: TOP {TOP_K} PER TRAINING FOLD ===")
    metrics, oof_df, fold_df, selected_df = run_cv_nested_feature_selection(
        X,
        y,
        feature_names,
        scenario,
        TOP_K,
    )
    save_nested_outputs(scenario, metrics, oof_df, fold_df, selected_df)
    scenario_metrics.append(metrics)

    return scenario_metrics

# ============================================================
# MAIN
# ============================================================
def main():
    csv_path = Path(CSV_FILE)
    if not csv_path.exists() and Path(FALLBACK_CSV_FILE).exists():
        csv_path = Path(FALLBACK_CSV_FILE)

    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    all_metrics = []

    for scenario in SCENARIOS:
        all_metrics.extend(run_scenario(df, scenario))

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv("all_xgboost_scenarios_comparison_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS ALL SCENARIOS")
    print("=" * 80)
    display_cols = ["model", "stage", "n_samples", "RMSE", "MAE", "R2", "r", "MAPE"]
    print(comparison_df[display_cols].to_string(index=False))
    print("\nSaved file:")
    print("  - all_xgboost_scenarios_comparison_metrics.csv")


if __name__ == "__main__":
    main()
