#!/usr/bin/env python3
"""
ElasticNet SMILES + CT model with nested feature selection and Y-randomization.

Nature Scientific Reports submission-oriented version:
1. True-label model uses repeated outer CV for robust out-of-fold evaluation.
2. Feature selection is performed only inside each training fold.
3. Scaling is fitted only on each training fold through a Pipeline.
4. ElasticNet hyperparameter tuning is nested inside the training data through ElasticNetCV.
5. Y-randomization repeats the full modeling workflow after permuting the target labels.
6. Empirical p-values compare the true model against randomized-label models.

Expected outputs:
- elasticnet_smiles_ct_only_scenario_results.csv
- elasticnet_smiles_ct_only_scenario_feature_selection.csv
- elasticnet_smiles_ct_only_scenario_oof_predictions.csv
- elasticnet_y_randomization_results.csv
- elasticnet_y_randomization_summary.csv
- elasticnet_true_vs_y_randomization_comparison.csv
- elasticnet_y_randomization_report.txt
"""

import math
import time
import warnings

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


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


RANDOM_STATE = 42
CSV_FILE = "opv_table_modified.csv"

N_BITS = 64
RADIUS = 2
TOP_K_FEATURES = 20

# True model: keep this stronger for reporting.
TRUE_N_SPLITS = 5
TRUE_N_REPEATS = 20
TRUE_RF_TREES = 500

# Y-randomization: reviewer-acceptable and computationally practical.
# For final submission, 100 permutations is preferable if runtime allows.
N_Y_RANDOMIZATIONS = 100
YRAND_N_SPLITS = 5
YRAND_N_REPEATS = 5
YRAND_RF_TREES = 100

DONOR_SMILES = "SMILES(donor)"
ACCEPTOR_SMILES = "SMILES(acceptor)"

STRONG_CT_COLS = [
    "Excitation energy (ECT)",
    "RMSeh",
    "POS",
    "ECT - E1S",
]


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
        nBits=N_BITS
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


def empirical_p_value_greater(true_value, randomized_values):
    """P-value for true_value being greater than randomized values."""
    randomized_values = np.asarray(randomized_values, dtype=float)
    randomized_values = randomized_values[~np.isnan(randomized_values)]
    return float((np.sum(randomized_values >= true_value) + 1) / (len(randomized_values) + 1))


def empirical_p_value_lower(true_value, randomized_values):
    """P-value for true_value being lower than randomized values."""
    randomized_values = np.asarray(randomized_values, dtype=float)
    randomized_values = randomized_values[~np.isnan(randomized_values)]
    return float((np.sum(randomized_values <= true_value) + 1) / (len(randomized_values) + 1))


def prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    target_col = df.columns[-1]

    required_cols = [DONOR_SMILES, ACCEPTOR_SMILES] + STRONG_CT_COLS + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    y = df[target_col].values.astype(float)

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

    numeric_df = df[STRONG_CT_COLS].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median())

    donor_fp_names = [f"donor_fp_{i}" for i in range(N_BITS)]
    acceptor_fp_names = [f"acceptor_fp_{i}" for i in range(N_BITS)]

    feature_names = (
        donor_fp_names
        + acceptor_fp_names
        + ["donor_invalid_smiles", "acceptor_invalid_smiles"]
        + STRONG_CT_COLS
    )

    X = np.hstack([
        donor_fps,
        acceptor_fps,
        np.array(donor_bad).reshape(-1, 1),
        np.array(acceptor_bad).reshape(-1, 1),
        numeric_df.values,
    ])

    return X.astype(np.float32), y, feature_names, target_col


def select_features_inside_fold(
    X_train,
    y_train,
    top_k=TOP_K_FEATURES,
    n_estimators=TRUE_RF_TREES,
    random_state=RANDOM_STATE,
):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        min_samples_leaf=3,
        max_features=0.7,
        random_state=random_state,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    k = min(top_k, X_train.shape[1])

    return np.argsort(importances)[::-1][:k]


def get_elasticnet_model(fast=False):
    if fast:
        l1_ratio_grid = np.linspace(0.1, 0.9, 5)
        alpha_grid = np.logspace(-4, 1, 30)
    else:
        l1_ratio_grid = np.linspace(0.05, 0.95, 10)
        alpha_grid = np.logspace(-5, 1, 60)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=l1_ratio_grid,
            alphas=alpha_grid,
            max_iter=100000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def run_repeated_oof_model(
    X,
    y,
    feature_names,
    n_splits,
    n_repeats,
    rf_trees,
    fast_elasticnet=False,
    scenario_label="TRUE_LABEL_MODEL",
    save_feature_summary=True,
):
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    model = get_elasticnet_model(fast=fast_elasticnet)

    oof_preds = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    fold_rs = []

    feature_counts = {name: 0 for name in feature_names}

    n_folds = 0

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        n_folds += 1

        X_train_full = X[train_idx]
        X_test_full = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        selected_idx = select_features_inside_fold(
            X_train_full,
            y_train,
            n_estimators=rf_trees,
            random_state=RANDOM_STATE + fold_id,
        )

        if save_feature_summary:
            for idx in selected_idx:
                feature_counts[feature_names[idx]] += 1

        X_train = X_train_full[:, selected_idx]
        X_test = X_test_full[:, selected_idx]

        m = clone(model)
        m.fit(X_train, y_train)

        preds = m.predict(X_test)

        oof_preds[test_idx] += preds
        counts[test_idx] += 1

        fold_rs.append(safe_pearsonr(y_test, preds))

    if np.any(counts == 0):
        raise RuntimeError("At least one sample was never assigned to a validation fold.")

    final_preds = oof_preds / counts

    r, rmse, mae, r2 = compute_metrics(y, final_preds)

    results = pd.DataFrame([{
        "case_scenario": scenario_label,
        "feature_set": "SMILES fingerprints + strong CT descriptors only",
        "model": "ElasticNetCV",
        "ct_descriptors": ", ".join(STRONG_CT_COLS),
        "n_samples": len(y),
        "n_features_before_selection": X.shape[1],
        "top_k_features": min(TOP_K_FEATURES, X.shape[1]),
        "cv_scheme": f"{n_splits}-fold x {n_repeats} repeats",
        "total_outer_folds": n_splits * n_repeats,
        "rf_trees_feature_selection": rf_trees,
        "elasticnet_grid": "fast" if fast_elasticnet else "full",
        "r": r,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "std_fold_r": float(np.nanstd(fold_rs)),
    }])

    feature_summary = pd.DataFrame()
    if save_feature_summary:
        feature_summary = pd.DataFrame({
            "feature": list(feature_counts.keys()),
            "selected_in_outer_folds": list(feature_counts.values()),
            "selected_fraction": [v / n_folds for v in feature_counts.values()],
        }).sort_values("selected_fraction", ascending=False)

    oof_table = pd.DataFrame({
        "actual_PCE": y,
        "predicted_ElasticNet": final_preds,
        "residual": y - final_preds,
        "n_validation_predictions_per_sample": counts,
    })

    return results, feature_summary, oof_table


def run_true_model(X, y, feature_names):
    results, feature_summary, oof_table = run_repeated_oof_model(
        X=X,
        y=y,
        feature_names=feature_names,
        n_splits=TRUE_N_SPLITS,
        n_repeats=TRUE_N_REPEATS,
        rf_trees=TRUE_RF_TREES,
        fast_elasticnet=False,
        scenario_label="SMILES_CT_ONLY_TRUE_LABEL_MODEL",
        save_feature_summary=True,
    )

    results.to_csv("elasticnet_smiles_ct_only_scenario_results.csv", index=False)
    feature_summary.to_csv("elasticnet_smiles_ct_only_scenario_feature_selection.csv", index=False)
    oof_table.to_csv("elasticnet_smiles_ct_only_scenario_oof_predictions.csv", index=False)

    return results, feature_summary, oof_table


def run_y_randomization(X, y, feature_names, n_randomizations=N_Y_RANDOMIZATIONS):
    print("\n" + "=" * 80)
    print("RUNNING Y-RANDOMIZATION TEST")
    print("=" * 80)

    rng = np.random.RandomState(RANDOM_STATE)
    random_results = []

    for i in range(n_randomizations):
        y_random = rng.permutation(y)

        start_i = time.time()

        results_i, _, _ = run_repeated_oof_model(
            X=X,
            y=y_random,
            feature_names=feature_names,
            n_splits=YRAND_N_SPLITS,
            n_repeats=YRAND_N_REPEATS,
            rf_trees=YRAND_RF_TREES,
            fast_elasticnet=True,
            scenario_label=f"Y_RANDOMIZATION_{i + 1}",
            save_feature_summary=False,
        )

        row = results_i.iloc[0].to_dict()
        row["iteration"] = i + 1
        row["elapsed_seconds"] = round(time.time() - start_i, 2)
        random_results.append(row)

        print(
            f"Y-randomization {i + 1}/{n_randomizations}: "
            f"r = {row['r']:.4f}, "
            f"RMSE = {row['rmse']:.4f}, "
            f"MAE = {row['mae']:.4f}, "
            f"R2 = {row['r2']:.4f}"
        )

    random_df = pd.DataFrame(random_results)
    random_df.to_csv("elasticnet_y_randomization_results.csv", index=False)

    summary = pd.DataFrame([{
        "n_randomizations": n_randomizations,
        "cv_scheme_per_randomization": f"{YRAND_N_SPLITS}-fold x {YRAND_N_REPEATS} repeats",
        "total_randomized_outer_folds": n_randomizations * YRAND_N_SPLITS * YRAND_N_REPEATS,
        "rf_trees_for_randomization": YRAND_RF_TREES,
        "elasticnet_grid_for_randomization": "5 l1_ratio values x 30 alpha values",
        "mean_random_r": random_df["r"].mean(),
        "std_random_r": random_df["r"].std(),
        "min_random_r": random_df["r"].min(),
        "max_random_r": random_df["r"].max(),
        "mean_random_rmse": random_df["rmse"].mean(),
        "std_random_rmse": random_df["rmse"].std(),
        "min_random_rmse": random_df["rmse"].min(),
        "max_random_rmse": random_df["rmse"].max(),
        "mean_random_mae": random_df["mae"].mean(),
        "std_random_mae": random_df["mae"].std(),
        "min_random_mae": random_df["mae"].min(),
        "max_random_mae": random_df["mae"].max(),
        "mean_random_r2": random_df["r2"].mean(),
        "std_random_r2": random_df["r2"].std(),
        "min_random_r2": random_df["r2"].min(),
        "max_random_r2": random_df["r2"].max(),
    }])

    summary.to_csv("elasticnet_y_randomization_summary.csv", index=False)

    return random_df, summary


def make_true_vs_randomization_comparison(true_results, random_df):
    true_row = true_results.iloc[0]

    comparison = pd.DataFrame([{
        "true_r": true_row["r"],
        "mean_random_r": random_df["r"].mean(),
        "std_random_r": random_df["r"].std(),
        "empirical_p_r_greater": empirical_p_value_greater(true_row["r"], random_df["r"]),
        "true_rmse": true_row["rmse"],
        "mean_random_rmse": random_df["rmse"].mean(),
        "std_random_rmse": random_df["rmse"].std(),
        "empirical_p_rmse_lower": empirical_p_value_lower(true_row["rmse"], random_df["rmse"]),
        "true_mae": true_row["mae"],
        "mean_random_mae": random_df["mae"].mean(),
        "std_random_mae": random_df["mae"].std(),
        "empirical_p_mae_lower": empirical_p_value_lower(true_row["mae"], random_df["mae"]),
        "true_r2": true_row["r2"],
        "mean_random_r2": random_df["r2"].mean(),
        "std_random_r2": random_df["r2"].std(),
        "empirical_p_r2_greater": empirical_p_value_greater(true_row["r2"], random_df["r2"]),
    }])

    comparison.to_csv("elasticnet_true_vs_y_randomization_comparison.csv", index=False)
    return comparison


def write_report(true_results, feature_summary, y_random_summary, comparison, target_col):
    true_row = true_results.iloc[0]
    yrand_row = y_random_summary.iloc[0]
    comp_row = comparison.iloc[0]

    lines = []
    lines.append("ElasticNet SMILES + CT Y-randomization validation report")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Target column: {target_col}")
    lines.append(f"Feature set: donor Morgan fingerprints, acceptor Morgan fingerprints, invalid-SMILES flags, and strong CT descriptors: {', '.join(STRONG_CT_COLS)}")
    lines.append(f"Fingerprint settings: Morgan radius = {RADIUS}, nBits = {N_BITS}")
    lines.append(f"Outer validation for true model: {TRUE_N_SPLITS}-fold x {TRUE_N_REPEATS} repeats")
    lines.append(f"Y-randomization: {N_Y_RANDOMIZATIONS} permutations, {YRAND_N_SPLITS}-fold x {YRAND_N_REPEATS} repeats per permutation")
    lines.append("")
    lines.append("Leakage control:")
    lines.append("Feature selection was repeated independently inside each outer training fold using only the training labels. The held-out fold was not used during feature selection, scaling, ElasticNetCV tuning, or model fitting. StandardScaler was fitted only within the training fold through the scikit-learn Pipeline.")
    lines.append("")
    lines.append("True-label repeated out-of-fold performance:")
    lines.append(f"r = {true_row['r']:.4f}, R2 = {true_row['r2']:.4f}, RMSE = {true_row['rmse']:.4f}, MAE = {true_row['mae']:.4f}")
    lines.append("")
    lines.append("Y-randomized-label performance:")
    lines.append(f"mean random r = {yrand_row['mean_random_r']:.4f} ± {yrand_row['std_random_r']:.4f}")
    lines.append(f"mean random R2 = {yrand_row['mean_random_r2']:.4f} ± {yrand_row['std_random_r2']:.4f}")
    lines.append(f"mean random RMSE = {yrand_row['mean_random_rmse']:.4f} ± {yrand_row['std_random_rmse']:.4f}")
    lines.append(f"mean random MAE = {yrand_row['mean_random_mae']:.4f} ± {yrand_row['std_random_mae']:.4f}")
    lines.append("")
    lines.append("Empirical one-sided p-values:")
    lines.append(f"p(r_random >= r_true) = {comp_row['empirical_p_r_greater']:.4f}")
    lines.append(f"p(R2_random >= R2_true) = {comp_row['empirical_p_r2_greater']:.4f}")
    lines.append(f"p(RMSE_random <= RMSE_true) = {comp_row['empirical_p_rmse_lower']:.4f}")
    lines.append(f"p(MAE_random <= MAE_true) = {comp_row['empirical_p_mae_lower']:.4f}")
    lines.append("")
    lines.append("Most frequently selected features in the true-label model:")
    for _, row in feature_summary.head(20).iterrows():
        lines.append(f"- {row['feature']}: selected fraction = {row['selected_fraction']:.3f}")
    lines.append("")
    lines.append("Suggested manuscript wording:")
    lines.append("A Y-randomization test was performed by randomly permuting the PCE labels and repeating the complete nested modeling workflow, including fold-wise feature selection, scaling, hyperparameter optimization, model fitting, and out-of-fold prediction. The true-label model was compared with the randomized-label distribution using empirical one-sided p-values. A true model that clearly outperforms the randomized-label distribution supports that the observed predictive performance is not due to chance correlation.")

    report = "\n".join(lines)
    with open("elasticnet_y_randomization_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    start_total = time.time()

    X, y, feature_names, target_col = prepare_data(CSV_FILE)

    print("\n" + "=" * 80)
    print("RUNNING TRUE-LABEL MODEL")
    print("=" * 80)

    true_results, feature_summary, _ = run_true_model(X, y, feature_names)

    y_random_df, y_random_summary = run_y_randomization(
        X,
        y,
        feature_names,
        n_randomizations=N_Y_RANDOMIZATIONS,
    )

    comparison = make_true_vs_randomization_comparison(
        true_results,
        y_random_df,
    )

    report = write_report(
        true_results,
        feature_summary,
        y_random_summary,
        comparison,
        target_col,
    )

    elapsed_total = time.time() - start_total

    print("\n" + "=" * 80)
    print("SMILES + CT ONLY INPUT ALGORITHM")
    print("=" * 80)

    print("Feature set:")
    print("  Morgan donor fingerprints")
    print("  Morgan acceptor fingerprints")
    print("  Donor/acceptor invalid SMILES flags")
    print("  Strong CT descriptors only")

    print("\nStrong CT descriptors used:")
    for col in STRONG_CT_COLS:
        print(f"  {col}")

    print("\nAlgorithm:")
    print("  Model: ElasticNetCV")
    print(f"  Fingerprint bits: {N_BITS}")
    print(f"  Morgan radius: {RADIUS}")
    print(f"  Feature selection: RandomForest inside each outer training fold")
    print(f"  TOP_K_FEATURES: {TOP_K_FEATURES}")
    print(f"  True-model validation: {TRUE_N_SPLITS}-fold x {TRUE_N_REPEATS} repeats")
    print(f"  Y-randomization permutations: {N_Y_RANDOMIZATIONS}")
    print(f"  Y-randomization validation: {YRAND_N_SPLITS}-fold x {YRAND_N_REPEATS} repeats")
    print("  Leakage control: feature selection, scaling, and ElasticNetCV fitting")
    print("  are performed only inside each training fold.")

    print("\n" + "=" * 80)
    print("TRUE-LABEL RESULTS")
    print("=" * 80)
    print(true_results.to_string(index=False))

    print("\nTop selected features:")
    print(feature_summary.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("Y-RANDOMIZATION SUMMARY")
    print("=" * 80)
    print(y_random_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("TRUE VS Y-RANDOMIZATION COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))

    print("\nSaved files:")
    print("  elasticnet_smiles_ct_only_scenario_results.csv")
    print("  elasticnet_smiles_ct_only_scenario_feature_selection.csv")
    print("  elasticnet_smiles_ct_only_scenario_oof_predictions.csv")
    print("  elasticnet_y_randomization_results.csv")
    print("  elasticnet_y_randomization_summary.csv")
    print("  elasticnet_true_vs_y_randomization_comparison.csv")
    print("  elasticnet_y_randomization_report.txt")

    print(f"\nTotal elapsed time: {elapsed_total / 60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
