# -*- coding: utf-8 -*-
"""
This script is now responsible for evaluating the full AMC pipeline, including:
1. Loading the trained Scaler, QDA, and XGBoost models.
2. Selecting an optimal abstention threshold on the validation set.
3. Performing a comprehensive evaluation on the test set, including:
    - Overall accuracy.
    - Selective accuracy and coverage based on the abstention threshold.
    - Confusion matrix plotting.
    - Per-SNR accuracy analysis.
"""
import os
import pickle
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from evaluation_utils import (
    MOD_FAMILIES,
    FAMILY_TARGET_COVERAGE,
    select_family_thresholds,
    apply_family_thresholds,
    print_evaluation_summary,
    plot_confusion_matrix_with_abstention,
    plot_evaluation_curves
)

# --- Configuration ---
PER_KEY = 2000
DATA_DIR = "data_cache"
MODEL_DIR = "models"
# Epsilon for numerical stability
EPS = 1e-9

# --- Input Data Paths ---
X_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_val.npy")
X_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_test.npy")
SNR_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_snr_test.npy")

# --- Input Model Paths ---
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_perkey{PER_KEY}.joblib")
GEN_MODEL_PATH = os.path.join(MODEL_DIR, f"gen_model_qda_perkey{PER_KEY}.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, f"xgb_model_perkey{PER_KEY}.joblib")
LE_PATH = os.path.join(MODEL_DIR, f"label_encoder_perkey{PER_KEY}.joblib")


def create_meta_features(gen_model, X_scaled):
    """Helper to create meta-features. Must be identical to the one in training."""
    log_post = gen_model.predict_log_proba(X_scaled)
    sorted_log_post = np.sort(log_post, axis=1)
    margin_gen = (sorted_log_post[:, -1] - sorted_log_post[:, -2]).reshape(-1, 1)
    return np.hstack([X_scaled, log_post, margin_gen])

def main():
    """Main evaluation pipeline."""
    print("--- Starting AMC Model Evaluation Pipeline ---")

    # --- 1. Load Models and Data ---
    print("Loading models and data...")
    scaler = joblib.load(SCALER_PATH)
    gen_model = joblib.load(GEN_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    le = joblib.load(LE_PATH)

    X_val = np.load(X_VAL_PATH)
    y_val_str = np.load(Y_VAL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test_str = np.load(Y_TEST_PATH)
    snr_test = np.load(SNR_TEST_PATH)

    y_val_int = le.transform(y_val_str)
    y_test_int = le.transform(y_test_str)

    # --- 2. Prepare Meta-Features for Validation and Test Sets ---
    print("Preparing meta-features...")
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_val_meta = create_meta_features(gen_model, X_val_scaled)
    X_test_meta = create_meta_features(gen_model, X_test_scaled)

    # --- 3. Select Abstention Thresholds for each Family on Validation Set ---
    print("\n--- Selecting Abstention Thresholds by Family ---")
    y_prob_val = xgb_model.predict_proba(X_val_meta)
    
    family_thresholds = select_family_thresholds(
        y_prob_val,
        y_val_str, # Use string labels to map to families
        MOD_FAMILIES,
        FAMILY_TARGET_COVERAGE
    )
    print("Selected Abstention Thresholds (tau) by Family:")
    for family, tau in family_thresholds.items():
        print(f"  - {family:<8}: {tau:.4f}")

    # --- 4. Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    y_prob_test = xgb_model.predict_proba(X_test_meta)
    y_pred_int_full = np.argmax(y_prob_test, axis=1)
    y_pred_abstained = apply_family_thresholds(y_prob_test, family_thresholds, le, MOD_FAMILIES)

    # Print a summary of metrics
    print_evaluation_summary(y_test_int, y_pred_int_full, y_pred_abstained)

    # --- Plotting and Saving Results ---
    print("\n--- Plotting and Saving Results ---")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to '{output_dir}/'")

    # Plot confusion matrix for non-abstained predictions
    cm_save_path = os.path.join(output_dir, f"confusion_matrix_perkey{PER_KEY}.png")
    plot_confusion_matrix_with_abstention(y_test_int, y_pred_abstained, le, save_path=cm_save_path)

    # Plot evaluation curves (SNR performance and Coverage-Accuracy trade-off)
    if os.path.exists(SNR_TEST_PATH):
        eval_curves_save_path = os.path.join(output_dir, f"evaluation_curves_perkey{PER_KEY}.png")
        plot_evaluation_curves(
            y_prob_test=y_prob_test,
            y_true_test=y_test_int,
            snrs_test=snr_test,
            y_pred_abstained_test=y_pred_abstained,
            y_prob_val=y_prob_val,
            y_true_val=y_val_int,
            save_path=eval_curves_save_path
        )
    else:
        print("SNR data not found, skipping evaluation curves plot.")

    print("\n--- Evaluation Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    main()
