#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# -------------------------
# SETTINGS
# -------------------------
CSV_FILE = "opv_table_modified.csv"  # change this if your CSV has a different name
TARGET_COL = "PCE(%)"

DONOR_SMILES_COL = "SMILES(donor)"
ACCEPTOR_SMILES_COL = "SMILES(acceptor)"

# Train only with these FMO/bandgap descriptors + donor/acceptor SMILES fingerprints.
# CT/TheoDORE/device features are intentionally excluded.
NUMERIC_FEATURES = [
    "Donor_HOMO(ev)",
    "Donor_LUMO(ev)",
    "Donor_bandgap(ev)",
    "Acceptor_HOMO(ev)",
    "Acceptor_LUMO(ev)",
    "Acceptor_bandgap(ev)",
]

FP_RADIUS = 2
FP_NBITS = 145
RANDOM_SEED = 42
N_SPLITS = 8
N_ENSEMBLE_MODELS = 50
TOP_K = 50

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 4],
    "learning_rate": [0.03, 0.05, 0.08],
    "gamma": [0.0, 0.1],
    "colsample_bytree": [0.7],
    "min_child_weight": [1],
    "subsample": [0.7],
}

# -------------------------
# METRICS
# -------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

# -------------------------
# SMILES → FP
# -------------------------
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

# -------------------------
# BUILD FEATURES
# -------------------------
def build_feature_matrix(df):
    """Build X from only FMO/bandgap columns plus donor/acceptor SMILES fingerprints."""

    required_cols = NUMERIC_FEATURES + [DONOR_SMILES_COL, ACCEPTOR_SMILES_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    numeric_df = df[NUMERIC_FEATURES].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    donor_fps, acceptor_fps = [], []
    donor_valid, acceptor_valid = [], []

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
        np.column_stack([donor_valid, acceptor_valid]).astype(np.float32),
    ])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)

    print("Using numeric features:", NUMERIC_FEATURES)
    print(f"Using donor/acceptor Morgan fingerprints: radius={FP_RADIUS}, nBits={FP_NBITS}")
    print(f"Final feature matrix shape: {X[mask].shape}")

    return X[mask], y[mask]

# -------------------------
# MODEL
# -------------------------
def make_xgb_model(params, seed):
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        **params,
    )

# -------------------------
# HYPERPARAMETER TUNING
# -------------------------
def fit_best_params(X, y, seed):
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=seed)

    best_params, best_rmse = None, float("inf")

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

# -------------------------
# ENSEMBLE
# -------------------------
def ensemble_predict(Xtr, ytr, Xte, params, seed):
    preds = []

    for i in range(N_ENSEMBLE_MODELS):
        model = make_xgb_model(params, seed + i)
        model.fit(Xtr, ytr)
        preds.append(model.predict(Xte))

    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
def get_feature_importance(X, y):
    model = XGBRegressor(n_estimators=300, max_depth=3)
    model.fit(X, y)
    return model.feature_importances_

# -------------------------
# FEATURE SELECTION
# -------------------------
def select_top_features(X, importances, k):
    idx = np.argsort(importances)[::-1][:k]
    return X[:, idx], idx

# -------------------------
# CROSS VALIDATION (UPDATED)
# -------------------------
def run_cv(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    preds = np.zeros_like(y)
    stds = np.zeros_like(y)

    for i, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        params = fit_best_params(Xtr, ytr, RANDOM_SEED + i)
        pred, std = ensemble_predict(Xtr, ytr, Xte, params, RANDOM_SEED + 1000*i)

        preds[te] = pred
        stds[te] = std

        print(f"[Fold {i}] RMSE={rmse(yte,pred):.3f}  r={pearson_r(yte,pred):.3f}")

    print("\nFINAL:")
    print("RMSE:", rmse(y, preds))
    print("r:", pearson_r(y, preds))

    return preds, stds

# -------------------------
# MAIN
# -------------------------
def main():
    df = pd.read_csv(CSV_FILE)

    X, y = build_feature_matrix(df)

    # BEFORE FEATURE SELECTION
    print("\n=== BEFORE FEATURE SELECTION ===")
    preds_before, stds_before = run_cv(X, y)

    df_before = pd.DataFrame({
        "actual_PCE": y,
        "predicted_PCE_oof": preds_before,
        "prediction_std": stds_before,
        "residual": y - preds_before
    })
    df_before.to_csv("predictions_before_feature_selection.csv", index=False)

    # FEATURE SELECTION
    importances = get_feature_importance(X, y)
    X_sel, idx = select_top_features(X, importances, TOP_K)

    print(f"\nSelected top {TOP_K} features")

    # AFTER FEATURE SELECTION
    print("\n=== AFTER FEATURE SELECTION ===")
    preds_after, stds_after = run_cv(X_sel, y)

    df_after = pd.DataFrame({
        "actual_PCE": y,
        "predicted_PCE_oof": preds_after,
        "prediction_std": stds_after,
        "residual": y - preds_after
    })
    df_after.to_csv("predictions_after_feature_selection.csv", index=False)


# -------------------------
if __name__ == "__main__":
    main()
