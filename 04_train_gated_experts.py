# -*- coding: utf-8 -*-
"""
This script is now responsible for training the full AMC pipeline, including:
1. Training a generative model (QDA) on baseline features.
2. Creating meta-features by combining original features and generative log-posteriors.
3. Training a powerful discriminative model (XGBoost) on the meta-features.
"""
import os
import pickle
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
# This should match the configuration in 03_build_feature_dataset.py
PER_KEY = 2000
DATA_DIR = "data_cache"
MODEL_DIR = "models"

# --- File Paths ---
X_TRAIN_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_val.npy")
META_PATH = os.path.join(DATA_DIR, f"meta_perkey{PER_KEY}.pkl")

# --- Output Model Paths ---
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_perkey{PER_KEY}.joblib")
GEN_MODEL_PATH = os.path.join(MODEL_DIR, f"gen_model_qda_perkey{PER_KEY}.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, f"xgb_model_perkey{PER_KEY}.joblib")
LE_PATH = os.path.join(MODEL_DIR, f"label_encoder_perkey{PER_KEY}.joblib")


def train_generative_model(X_train_scaled, y_train_int):
    """Trains QDA, with GaussianNB as a fallback."""
    print("Training generative model (QDA)...")
    try:
        # QDA with regularization is a good starting point
        model = QuadraticDiscriminantAnalysis(reg_param=0.1)
        model.fit(X_train_scaled, y_train_int)
        print("QDA training successful.")
    except Exception as e:
        print(f"Warning: QDA failed: {e}. Falling back to GaussianNB.")
        model = GaussianNB()
        model.fit(X_train_scaled, y_train_int)
        print("GaussianNB training successful.")
    return model

def create_meta_features(gen_model, X_scaled):
    """Creates meta-features using the generative model."""
    # Get log-posteriors from the generative model
    log_post = gen_model.predict_log_proba(X_scaled)
    
    # Calculate the margin (top1 - top2 log-posterior)
    sorted_log_post = np.sort(log_post, axis=1)
    margin_gen = (sorted_log_post[:, -1] - sorted_log_post[:, -2]).reshape(-1, 1)
    
    # Combine original features with generative meta-features
    return np.hstack([X_scaled, log_post, margin_gen])

def main():
    """Main training pipeline."""
    print("--- Starting AMC Model Training Pipeline ---")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 1. Load Data ---
    print("Loading training and validation data...")
    X_train = np.load(X_TRAIN_PATH)
    y_train_str = np.load(Y_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)
    y_val_str = np.load(Y_VAL_PATH)
    
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    
    # --- 2. Label Encoding ---
    print("Encoding labels...")
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train_str)
    y_val_int = le.transform(y_val_str)
    num_classes = len(le.classes_)
    print(f"Found {num_classes} classes: {le.classes_}")
    joblib.dump(le, LE_PATH) # Save the encoder for later use
    print(f"Label encoder saved to {LE_PATH}")

    # --- 3. Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # --- 4. Train Generative Model & Create Meta-Features ---
    gen_model = train_generative_model(X_train_scaled, y_train_int)
    joblib.dump(gen_model, GEN_MODEL_PATH)
    print(f"Generative model saved to {GEN_MODEL_PATH}")
    
    print("Creating meta-features for train and validation sets...")
    X_train_meta = create_meta_features(gen_model, X_train_scaled)
    X_val_meta = create_meta_features(gen_model, X_val_scaled)
    print(f"Original feature dim: {X_train_scaled.shape[1]}")
    print(f"Meta-feature dim: {X_train_meta.shape[1]}")

    # --- 5. Train XGBoost Classifier ---
    print("Training final XGBoost classifier...")
    
    # Calculate class weights for imbalance
    class_weights_val = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    sample_weights_train = np.array([class_weights_val[i] for i in y_train_int])

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train_meta, y_train_int,
        sample_weight=sample_weights_train
    )
    
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    print(f"XGBoost model saved to {XGB_MODEL_PATH}")

    print("\n--- Training Pipeline Finished Successfully! ---")
    print(f"All models saved in '{MODEL_DIR}' directory.")


if __name__ == "__main__":
    main()
