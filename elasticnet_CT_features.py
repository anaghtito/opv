#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
CSV_FILE = "opv_table_modified.csv"

N_BITS = 64
RADIUS = 2
TOP_K_FEATURES = 20

N_SPLITS = 5
N_REPEATS = 20

DONOR_SMILES = "SMILES(donor)"
ACCEPTOR_SMILES = "SMILES(acceptor)"

FMO_COLS = [
    "Donor_HOMO(ev)", "Donor_LUMO(ev)", "Donor_bandgap(ev)",
    "Acceptor_HOMO(ev)", "Acceptor_LUMO(ev)", "Acceptor_bandgap(ev)",
]

CT_COLS = [
    "Excitation energy (ECT)", "OSc", "ECT - E1S", "ECT - ET1",
    "CT", "POS", "PR", "PRNTO", "RMSeh"
]

# ============================================================
# FUNCTIONS
# ============================================================

def smiles_to_fp(smiles):
    arr = np.zeros((N_BITS,), dtype=np.float32)

    if pd.isna(smiles):
        return arr, 1

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return arr, 1

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, 0


def compute_metrics(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return r, rmse, r2


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    target_col = df.columns[-1]

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    donor_fps, acceptor_fps = [], []
    donor_bad, acceptor_bad = [], []

    for _, row in df.iterrows():
        d_fp, d_flag = smiles_to_fp(row[DONOR_SMILES])
        a_fp, a_flag = smiles_to_fp(row[ACCEPTOR_SMILES])

        donor_fps.append(d_fp)
        acceptor_fps.append(a_fp)
        donor_bad.append(d_flag)
        acceptor_bad.append(a_flag)

    donor_fps = np.asarray(donor_fps)
    acceptor_fps = np.asarray(acceptor_fps)

    numeric_cols = FMO_COLS + CT_COLS
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median())

    eff_gap = (
        np.abs(numeric_df["Donor_HOMO(ev)"]) -
        np.abs(numeric_df["Acceptor_LUMO(ev)"])
    ).values.reshape(-1, 1)

    X = np.hstack([
        donor_fps,
        acceptor_fps,
        np.array(donor_bad).reshape(-1, 1),
        np.array(acceptor_bad).reshape(-1, 1),
        numeric_df.values,
        eff_gap
    ])

    y = df[target_col].values

    print("Samples:", len(y))
    print("Initial features:", X.shape[1])

    return X.astype(np.float32), y.astype(float)


# ============================================================
# GLOBAL FEATURE SELECTION
# ============================================================

def global_feature_selection(X, y):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X, y)

    perm = permutation_importance(
        rf, X, y,
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importances = perm.importances_mean
    idx = np.argsort(importances)[::-1][:TOP_K_FEATURES]

    print("\nSelected top features:", TOP_K_FEATURES)

    return X[:, idx]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_rf():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 50)))
    ])


def get_elastic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-4, 1, 50),
            max_iter=100000
        ))
    ])


# ============================================================
# CORRECT OOF EVALUATION (FIXED)
# ============================================================

def evaluate_model(name, model, X, y):

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    oof_preds = np.zeros(len(y))
    counts = np.zeros(len(y))

    fold_rs = []

    for train_idx, test_idx in cv.split(X):

        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[test_idx])

        oof_preds[test_idx] += preds
        counts[test_idx] += 1

        r, _, _ = compute_metrics(y[test_idx], preds)
        fold_rs.append(r)

    oof_preds /= counts

    r, rmse, r2 = compute_metrics(y, oof_preds)

    return {
        "model": name,
        "r": r,
        "rmse": rmse,
        "r2": r2,
        "std_r": np.std(fold_rs),
        "predictions": oof_preds
    }


# ============================================================
# MAIN
# ============================================================

def main():
    X, y = prepare_data(CSV_FILE)

    X_sel = global_feature_selection(X, y)

    models = {
        "RandomForest": get_rf(),
        "Ridge": get_ridge(),
        "ElasticNet": get_elastic()
    }

    results = []

    for name, model in models.items():
        print("\nRunning:", name)
        res = evaluate_model(name, model, X_sel, y)

        print(f"r = {res['r']:.4f}")
        print(f"RMSE = {res['rmse']:.4f}")
        print(f"R² = {res['r2']:.4f}")
        print(f"std(r) = {res['std_r']:.4f}")

        results.append(res)

    pd.DataFrame(results).to_csv("final_results.csv", index=False)
    print("\nSaved: final_results.csv")


if __name__ == "__main__":
    main()
