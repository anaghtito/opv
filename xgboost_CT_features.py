#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import warnings
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

CSV_FILE = "opv_table_modified.csv"
TARGET_COL = "PCE(%)"

DONOR_SMILES_COL = "SMILES(donor)"
ACCEPTOR_SMILES_COL = "SMILES(acceptor)"

NUMERIC_FEATURES = [
    "Excitation energy (ECT)", "OSc", "ECT - E1S", "ECT - ET1", "CT",
    "POS", "PR", "COH", "PRNTO", "Z_HE", "RMSeh",
]

FP_RADIUS = 2
FP_NBITS = 145
RANDOM_SEED = 42
N_SPLITS = 8
N_ENSEMBLE_MODELS = 50

# 🔥 UPDATED FEATURE COUNT
TOP_K = 40

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

def mape(y_true, y_pred, eps=1e-8):
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

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
# BUILD FEATURES (FIXED)
# -------------------------
def build_feature_matrix(df):

    numeric_df = df[NUMERIC_FEATURES].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    donor_fps, acceptor_fps = [], []
    donor_valid, acceptor_valid = [], []

    for _, row in df.iterrows():
        d = row[DONOR_SMILES_COL]
        a = row[ACCEPTOR_SMILES_COL]

        dmol = Chem.MolFromSmiles(str(d)) if pd.notna(d) else None
        amol = Chem.MolFromSmiles(str(a)) if pd.notna(a) else None

        donor_valid.append(0.0 if dmol is None else 1.0)
        acceptor_valid.append(0.0 if amol is None else 1.0)

        donor_fps.append(smiles_to_morgan_fp(d))
        acceptor_fps.append(smiles_to_morgan_fp(a))

    X = np.hstack([
        numeric_df.to_numpy(dtype=np.float32),
        np.vstack(donor_fps),
        np.vstack(acceptor_fps),
        np.column_stack([donor_valid, acceptor_valid])
    ])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)

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
# TUNING
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
# CROSS VALIDATION
# -------------------------
def run_cv(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    preds = np.zeros_like(y)

    for i, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        params = fit_best_params(Xtr, ytr, RANDOM_SEED + i)
        pred, std = ensemble_predict(Xtr, ytr, Xte, params, RANDOM_SEED + 1000*i)

        preds[te] = pred

        print(f"[Fold {i}] RMSE={rmse(yte,pred):.3f}  r={pearson_r(yte,pred):.3f}")

    print("\nFINAL:")
    print("RMSE:", rmse(y, preds))
    print("r:", pearson_r(y, preds))

# -------------------------
# MAIN
# -------------------------
def main():
    df = pd.read_csv(CSV_FILE)

    X, y = build_feature_matrix(df)

    print("\n=== BEFORE FEATURE SELECTION ===")
    run_cv(X, y)

    importances = get_feature_importance(X, y)

    X_sel, idx = select_top_features(X, importances, TOP_K)

    print(f"\nSelected top {TOP_K} features")

    print("\n=== AFTER FEATURE SELECTION ===")
    run_cv(X_sel, y)

# -------------------------
if __name__ == "__main__":
    main()
