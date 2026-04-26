import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from feature_extraction import extract_features
from feature_enhancer import extract_disambiguation_features


PER_KEY = 2000
MODEL_DIR = "models"
DATA_PATH = os.path.join("data_cache", "custommod_overlap_eval.npz")
OUTPUT_DIR = "plots_generalize_snr_analysis"

SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_perkey{PER_KEY}.joblib")
GEN_MODEL_PATH = os.path.join(MODEL_DIR, f"gen_model_qda_perkey{PER_KEY}.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, f"xgb_model_perkey{PER_KEY}.joblib")
LE_PATH = os.path.join(MODEL_DIR, f"label_encoder_perkey{PER_KEY}.joblib")


def create_meta_features(gen_model, X_scaled):
    log_post = gen_model.predict_log_proba(X_scaled)
    sorted_log_post = np.sort(log_post, axis=1)
    margin_gen = (sorted_log_post[:, -1] - sorted_log_post[:, -2]).reshape(-1, 1)
    return np.hstack([X_scaled, log_post, margin_gen])


def build_feature_matrix(X_iq):
    feature_rows = []
    for sample in X_iq:
        base_features = extract_features(sample)
        extra_features = extract_disambiguation_features(sample)
        feature_rows.append(np.concatenate([base_features, extra_features]))
    return np.vstack(feature_rows).astype(np.float32)


def plot_custom_confusion_matrix(y_true, y_pred, model_classes, save_path):
    labels = list(model_classes)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("CustomMOD Generalization Confusion Matrix (10x10)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    print("Loading overlap dataset:", DATA_PATH)
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    snr = data["snr"]

    print("Loading model artifacts...")
    scaler = joblib.load(SCALER_PATH)
    gen_model = joblib.load(GEN_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    le = joblib.load(LE_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Extracting handcrafted features...")
    X_feat = build_feature_matrix(X)
    print("Feature matrix shape:", X_feat.shape)

    print("Running inference...")
    X_scaled = scaler.transform(X_feat)
    X_meta = create_meta_features(gen_model, X_scaled)
    y_prob = xgb_model.predict_proba(X_meta)
    y_pred = le.inverse_transform(np.argmax(y_prob, axis=1))

    overall_acc = float(np.mean(y_pred == y))
    print("\n=== Generalization Result ===")
    print("Overall accuracy on overlapping classes: {:.4f}".format(overall_acc))

    print("\nPer-class accuracy:")
    for mod in le.classes_:
        class_mask = (y == mod)
        if np.any(class_mask):
            class_acc = float(np.mean(y_pred[class_mask] == y[class_mask]))
            print("  {}: {:.4f} ({} samples)".format(mod, class_acc, int(np.sum(class_mask))))

    print("\nPer-SNR accuracy:")
    for snr_value in np.unique(snr):
        snr_mask = (snr == snr_value)
        snr_acc = float(np.mean(y_pred[snr_mask] == y[snr_mask]))
        print("  SNR {}: {:.4f} ({} samples)".format(snr_value, snr_acc, int(np.sum(snr_mask))))

    print("\nPrediction distribution:")
    pred_names, pred_counts = np.unique(y_pred, return_counts=True)
    for name, count in zip(pred_names, pred_counts):
        print("  {}: {}".format(name, int(count)))

    cm_path = os.path.join(OUTPUT_DIR, "custommod_generalization_confusion_matrix.png")
    plot_custom_confusion_matrix(y, y_pred, le.classes_, cm_path)
    print("\nSaved confusion matrix to:", cm_path)


if __name__ == "__main__":
    main()
