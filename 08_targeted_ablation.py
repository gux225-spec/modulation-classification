
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm

# Add project root to path to import the newly refactored feature_extraction
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features

# --- Configuration ---
DATA_DIR = "data_cache"

def run_ablation_variant(
    variant_name: str,
    feature_flags: dict,
    X_train_raw, y_train_str, X_test_raw, y_test_str
):
    """Runs a full train-and-test cycle for a single feature variant."""
    print(f"--- Running Ablation Variant: {variant_name} ---")

    # 1. Feature Extraction with specific flags
    print("Extracting features...")
    X_train = np.array([extract_features(x, **feature_flags) for x in tqdm(X_train_raw)])
    X_test = np.array([extract_features(x, **feature_flags) for x in tqdm(X_test_raw)])

    # 2. Label Encoding, Scaling, and Training
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train_str)
    y_test_int = le.transform(y_test_str)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training model...")
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train_int, eval_set=[(X_test_scaled, y_test_int)], early_stopping_rounds=10, verbose=False)

    # 4. Evaluation
    y_pred_int = xgb_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test_int, y_pred_int, labels=le.transform(le.classes_), normalize='true')
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # 5. Report Key Metrics
    overall_acc = np.trace(cm_df.values)
    wbfm_amdsb = cm_df.loc['WBFM', 'AM-DSB']
    amdsb_wbfm = cm_df.loc['AM-DSB', 'WBFM']
    qam16_qam64 = cm_df.loc['QAM16', 'QAM64']
    qam64_qam16 = cm_df.loc['QAM64', 'QAM16']

    return {
        'variant': variant_name,
        'accuracy': overall_acc,
        'WBFM->AM-DSB': wbfm_amdsb,
        'AM-DSB->WBFM': amdsb_wbfm,
        'QAM16->QAM64': qam16_qam64,
        'QAM64->QAM16': qam64_qam16,
    }

def main():
    """Main ablation study pipeline."""
    # This script requires the raw, unprocessed data split.
    # We assume a file 'data_split_raw.pkl' is created by 03_build_feature_dataset.py
    # before feature extraction begins.
    try:
        with open(os.path.join(DATA_DIR, 'data_split_raw.pkl'), 'rb') as f:
            data = pickle.load(f)
        X_train_raw, y_train_str = data['train']
        X_test_raw, y_test_str = data['test']
    except FileNotFoundError:
        print("Error: Raw data split 'data_split_raw.pkl' not found.")
        print("Please ensure 03_build_feature_dataset.py saves the raw split before feature extraction.")
        return

    # Define the feature combinations to test
    variants = [
        {"name": "Baseline", "flags": {"add_analog_features": False, "add_qam_features": False}},
        {"name": "Baseline + Analog", "flags": {"add_analog_features": True, "add_qam_features": False}},
        {"name": "Baseline + QAM", "flags": {"add_analog_features": False, "add_qam_features": True}},
        {"name": "Baseline + All New", "flags": {"add_analog_features": True, "add_qam_features": True}},
    ]
    
    results = []
    for v in variants:
        result = run_ablation_variant(
            v['name'], v['flags'],
            X_train_raw, y_train_str, X_test_raw, y_test_str
        )
        results.append(result)

    print("\n\n--- Ablation Study Summary ---")
    results_df = pd.DataFrame(results).set_index('variant')
    print(results_df.to_string(float_format="%.4f"))

if __name__ == "__main__":
    main()
