
# -*- coding: utf-8 -*-
"""
This script analyzes the model's prediction confidence on the samples
that were *accepted* (i.e., not rejected) by the abstention mechanism.

It groups the accepted samples by their true modulation type and calculates
statistics on their prediction margins to gauge per-class confidence.
"""
import os
import numpy as np
import joblib
from evaluation_utils import (
    calculate_log_margins,
    get_abstention_results
)

# --- Configuration ---
PER_KEY = 2000
DATA_DIR = "data_cache"
MODEL_DIR = "models"

# --- Input Data Paths ---
X_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_val.npy")
X_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_test.npy")

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
    """Main confidence analysis pipeline."""
    print("--- Starting Confidence Analysis Pipeline ---")

    # --- 1. Load Models and Data ---
    print("Loading models and data...")
    if not all(os.path.exists(p) for p in [SCALER_PATH, GEN_MODEL_PATH, XGB_MODEL_PATH, LE_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH]):
        print("Error: Not all required model or data files were found. Please run the training and feature generation scripts first.")
        return

    scaler = joblib.load(SCALER_PATH)
    gen_model = joblib.load(GEN_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    le = joblib.load(LE_PATH)

    X_val = np.load(X_VAL_PATH)
    y_val_str = np.load(Y_VAL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test_str = np.load(Y_TEST_PATH)

    y_val_int = le.transform(y_val_str)
    y_test_int = le.transform(y_test_str)

    # --- 2. Prepare Meta-Features ---
    print("Preparing meta-features for validation and test sets...")
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_val_meta = create_meta_features(gen_model, X_val_scaled)
    X_test_meta = create_meta_features(gen_model, X_test_scaled)

    # --- 3. Determine Abstention Thresholds and Apply to Test Set (Centralized) ---
    family_thresholds, y_pred_abstained = get_abstention_results(
        xgb_model, X_val_meta, y_val_str, X_test_meta, le
    )

    # --- 4. Get Predictions and Margins for the Test Set ---
    y_prob_test = xgb_model.predict_proba(X_test_meta)
    margins_test = calculate_log_margins(y_prob_test)

    # --- 5. Filter for Accepted Samples ---
    accepted_mask = (y_pred_abstained != -1)
    num_total = len(y_test_int)
    num_accepted = np.sum(accepted_mask)
    print(f"\nTotal samples in test set: {num_total}")
    print(f"Accepted samples: {num_accepted} ({num_accepted/num_total:.2%})")

    accepted_margins = margins_test[accepted_mask]
    accepted_true_labels_int = y_test_int[accepted_mask]

    # --- 6. Analyze and Display Confidence Statistics ---
    print("\n--- Confidence Analysis on Accepted Samples (Grouped by TRUE modulation type) ---")
    print(f"{'Modulation':<12} | {'Count':>10} | {'Mean Margin':>15} | {'Std Dev':>15} | {'Min Margin':>15} | {'Max Margin':>15}")
    print("-" * 90)

    all_class_margins = []
    for mod_int, mod_str in enumerate(le.classes_):
        # Find all accepted samples where the TRUE label matches the current modulation
        class_mask = (accepted_true_labels_int == mod_int)

        if np.sum(class_mask) == 0:
            print(f"{mod_str:<12} | {'0':>10} | {'N/A':>15} | {'N/A':>15} | {'N/A':>15} | {'N/A':>15}")
            continue

        class_margins = accepted_margins[class_mask]
        all_class_margins.append(class_margins)

        count = len(class_margins)
        mean_margin = np.mean(class_margins)
        std_margin = np.std(class_margins)
        min_margin = np.min(class_margins)
        max_margin = np.max(class_margins)

        print(f"{mod_str:<12} | {count:>10} | {mean_margin:>15.4f} | {std_margin:>15.4f} | {min_margin:>15.4f} | {max_margin:>15.4f}")

    print("-" * 90)
    print("\n--- Analysis Finished ---")


if __name__ == "__main__":
    main()
