#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
CSV_FILE = "opv_table_modified.csv"
FALLBACK_CSV_FILE = "opv_table_modified(6).csv"

TARGET_COL = "PCE(%)"
DONOR_SMILES_COL = "SMILES(donor)"
ACCEPTOR_SMILES_COL = "SMILES(acceptor)"

MODEL_LABEL = "CT_No_COH"

NUMERIC_FEATURES = [
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

FP_RADIUS = 2
FP_NBITS = 2048
RANDOM_SEED = 42
N_SPLITS = 8
TOP_K = 20

N_ENSEMBLE_MODELS = 50

RUN_Y_RANDOMIZATION = True
N_Y_RANDOMIZATION = 50
YRAND_ENSEMBLE_MODELS = 10

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
# METRICS
# ============================================================
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-8):
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


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

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================
def build_feature_matrix(df: pd.DataFrame):
    missing = [
        c for c in NUMERIC_FEATURES
        + [TARGET_COL, DONOR_SMILES_COL, ACCEPTOR_SMILES_COL]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_df = df[NUMERIC_FEATURES].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

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

    X = np.hstack([
        numeric_df.to_numpy(dtype=np.float32),
        np.vstack(donor_fps),
        np.vstack(acceptor_fps),
        np.column_stack([donor_valid, acceptor_valid]),
    ])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)

    feature_names = []
    feature_names.extend(NUMERIC_FEATURES)
    feature_names.extend([f"Donor_FP_{i}" for i in range(FP_NBITS)])
    feature_names.extend([f"Acceptor_FP_{i}" for i in range(FP_NBITS)])
    feature_names.extend(["Donor_Valid", "Acceptor_Valid"])

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
def ensemble_predict(
    Xtr,
    ytr,
    Xte,
    params,
    seed,
    n_models=N_ENSEMBLE_MODELS,
):
    preds = []

    for i in range(n_models):
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
# LEAKAGE-SAFE NESTED FEATURE-SELECTION CV
# ============================================================
def run_nested_cv(
    X,
    y,
    feature_names,
    top_k=TOP_K,
    model_label=MODEL_LABEL,
    n_ensemble_models=N_ENSEMBLE_MODELS,
):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    preds = np.zeros_like(y)
    pred_stds = np.zeros_like(y)

    fold_rows = []
    selected_feature_rows = []

    for i, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        importances = get_feature_importance(
            Xtr,
            ytr,
            RANDOM_SEED + 100 * i,
        )

        idx = np.argsort(importances)[::-1][:top_k]

        Xtr_sel = Xtr[:, idx]
        Xte_sel = Xte[:, idx]

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
            n_models=n_ensemble_models,
        )

        preds[te] = pred
        pred_stds[te] = std

        fold_rmse = rmse(yte, pred)
        fold_mae = mean_absolute_error(yte, pred)
        fold_r = pearson_r(yte, pred)

        fold_rows.append({
            "model": model_label,
            "fold": i,
            "n_validation_samples": len(te),
            "RMSE": fold_rmse,
            "MAE": fold_mae,
            "r": fold_r,
            "selected_features": len(idx),
            "best_params": str(params),
        })

        for rank, j in enumerate(idx, 1):
            selected_feature_rows.append({
                "model": model_label,
                "fold": i,
                "rank": rank,
                "feature_index": int(j),
                "feature_name": feature_names[j],
                "importance": float(importances[j]),
            })

        print(
            f"[{model_label} | Fold {i}] "
            f"RMSE={fold_rmse:.3f}  "
            f"MAE={fold_mae:.3f}  "
            f"r={fold_r:.3f}  "
            f"selected_features={len(idx)}"
        )

    metrics = {
        "model": model_label,
        "n_samples": int(len(y)),
        "n_splits": int(N_SPLITS),
        "top_k": int(top_k),
        "fp_nbits": int(FP_NBITS),
        "ensemble_models": int(n_ensemble_models),
        "RMSE": rmse(y, preds),
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds),
        "r": pearson_r(y, preds),
        "MAPE": mape(y, preds),
    }

    oof_df = pd.DataFrame({
        "model": model_label,
        "actual_PCE": y,
        "predicted_PCE_oof": preds,
        "prediction_std": pred_stds,
        "residual": y - preds,
    })

    fold_df = pd.DataFrame(fold_rows)
    selected_df = pd.DataFrame(selected_feature_rows)

    return metrics, oof_df, fold_df, selected_df


# ============================================================
# Y-RANDOMIZATION TEST
# ============================================================
def run_y_randomization(
    X,
    y,
    feature_names,
    original_metrics,
    n_iterations=N_Y_RANDOMIZATION,
):
    print("\n" + "=" * 70)
    print("RUNNING FAST Y-RANDOMIZATION TEST")
    print("=" * 70)
    print(f"Y-randomization iterations: {n_iterations}")
    print(f"Ensemble models per Y-randomization run: {YRAND_ENSEMBLE_MODELS}")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for i in range(n_iterations):
        y_random = rng.permutation(y)

        print("\n" + "-" * 70)
        print(f"Y-randomization iteration {i + 1}/{n_iterations}")
        print("-" * 70)

        metrics, _, _, _ = run_nested_cv(
            X,
            y_random,
            feature_names,
            TOP_K,
            model_label=f"{MODEL_LABEL}_Y_RANDOMIZED",
            n_ensemble_models=YRAND_ENSEMBLE_MODELS,
        )

        rows.append({
            "iteration": i + 1,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "r": metrics["r"],
            "MAPE": metrics["MAPE"],
            "ensemble_models": YRAND_ENSEMBLE_MODELS,
        })

        print(
            f"Y-randomization {i + 1}/{n_iterations}: "
            f"r={metrics['r']:.3f}, "
            f"R2={metrics['R2']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"MAE={metrics['MAE']:.3f}"
        )

    yrand_df = pd.DataFrame(rows)

    p_value_r = (
        np.sum(yrand_df["r"] >= original_metrics["r"]) + 1
    ) / (len(yrand_df) + 1)

    p_value_r2 = (
        np.sum(yrand_df["R2"] >= original_metrics["R2"]) + 1
    ) / (len(yrand_df) + 1)

    summary = {
        "original_r": original_metrics["r"],
        "original_R2": original_metrics["R2"],
        "original_RMSE": original_metrics["RMSE"],
        "original_MAE": original_metrics["MAE"],
        "original_ensemble_models": N_ENSEMBLE_MODELS,
        "y_randomization_ensemble_models": YRAND_ENSEMBLE_MODELS,
        "n_y_randomization": n_iterations,
        "mean_random_r": yrand_df["r"].mean(),
        "std_random_r": yrand_df["r"].std(),
        "max_random_r": yrand_df["r"].max(),
        "min_random_r": yrand_df["r"].min(),
        "mean_random_R2": yrand_df["R2"].mean(),
        "std_random_R2": yrand_df["R2"].std(),
        "max_random_R2": yrand_df["R2"].max(),
        "min_random_R2": yrand_df["R2"].min(),
        "mean_random_RMSE": yrand_df["RMSE"].mean(),
        "std_random_RMSE": yrand_df["RMSE"].std(),
        "mean_random_MAE": yrand_df["MAE"].mean(),
        "std_random_MAE": yrand_df["MAE"].std(),
        "empirical_p_value_r": p_value_r,
        "empirical_p_value_R2": p_value_r2,
    }

    summary_df = pd.DataFrame([summary])

    yrand_df.to_csv("ct_no_coh_y_randomization_results_fast.csv", index=False)
    summary_df.to_csv("ct_no_coh_y_randomization_summary_fast.csv", index=False)

    print("\nY-RANDOMIZATION SUMMARY:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved Y-randomization files:")
    print("  - ct_no_coh_y_randomization_results_fast.csv")
    print("  - ct_no_coh_y_randomization_summary_fast.csv")

    return yrand_df, summary_df


# ============================================================
# MAIN
# ============================================================
def main():
    csv_path = Path(CSV_FILE)

    if not csv_path.exists() and Path(FALLBACK_CSV_FILE).exists():
        csv_path = Path(FALLBACK_CSV_FILE)

    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    print("\nUsing CT feature set:")
    for f in NUMERIC_FEATURES:
        print(f"  - {f}")

    X, y, feature_names = build_feature_matrix(df)

    print(f"\nNumber of samples: {len(y)}")
    print(f"Number of total features before selection: {X.shape[1]}")
    print(f"Number of selected features per fold: {TOP_K}")
    print(f"Original ensemble models: {N_ENSEMBLE_MODELS}")

    metrics, oof_df, fold_df, selected_df = run_nested_cv(
        X,
        y,
        feature_names,
        TOP_K,
        model_label=MODEL_LABEL,
        n_ensemble_models=N_ENSEMBLE_MODELS,
    )

    print("\nFINAL METRICS:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    metrics_df = pd.DataFrame([metrics])

    selected_ct_df = selected_df[
        selected_df["feature_name"].isin(NUMERIC_FEATURES)
    ].copy()

    if not selected_ct_df.empty:
        ct_frequency = (
            selected_ct_df
            .groupby("feature_name")
            .size()
            .reset_index(name="selection_frequency_out_of_8")
            .sort_values("selection_frequency_out_of_8", ascending=False)
        )
    else:
        ct_frequency = pd.DataFrame(
            columns=["feature_name", "selection_frequency_out_of_8"]
        )

    metrics_df.to_csv("ct_no_coh_summary_metrics.csv", index=False)
    oof_df.to_csv("ct_no_coh_oof_predictions.csv", index=False)
    fold_df.to_csv("ct_no_coh_foldwise_metrics.csv", index=False)
    selected_df.to_csv("ct_no_coh_selected_features_by_fold.csv", index=False)
    ct_frequency.to_csv("ct_no_coh_ct_feature_selection_frequency.csv", index=False)

    print("\nSaved files:")
    print("  - ct_no_coh_summary_metrics.csv")
    print("  - ct_no_coh_oof_predictions.csv")
    print("  - ct_no_coh_foldwise_metrics.csv")
    print("  - ct_no_coh_selected_features_by_fold.csv")
    print("  - ct_no_coh_ct_feature_selection_frequency.csv")

    if RUN_Y_RANDOMIZATION:
        run_y_randomization(
            X,
            y,
            feature_names,
            original_metrics=metrics,
            n_iterations=N_Y_RANDOMIZATION,
        )


if __name__ == "__main__":
    main()
