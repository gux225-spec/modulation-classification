
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from evaluation_utils import calculate_log_margins, MOD_FAMILIES

# --- Configuration ---
PER_KEY = 2000
DATA_DIR = "data_cache"
MODEL_DIR = "models"
OUTPUT_DIR = "plots_snr_analysis"
SNR_THRESHOLD = 8  # Threshold to distinguish low vs high SNR

# --- Input Paths ---
X_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_y_test.npy")
SNR_TEST_PATH = os.path.join(DATA_DIR, f"perkey{PER_KEY}_snr_test.npy")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_perkey{PER_KEY}.joblib")
GEN_MODEL_PATH = os.path.join(MODEL_DIR, f"gen_model_qda_perkey{PER_KEY}.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, f"xgb_model_perkey{PER_KEY}.joblib")
LE_PATH = os.path.join(MODEL_DIR, f"label_encoder_perkey{PER_KEY}.joblib")

def create_meta_features(gen_model, X_scaled):
    log_post = gen_model.predict_log_proba(X_scaled)
    sorted_log_post = np.sort(log_post, axis=1)
    margin_gen = (sorted_log_post[:, -1] - sorted_log_post[:, -2]).reshape(-1, 1)
    return np.hstack([X_scaled, log_post, margin_gen])

def plot_confusion_matrix_custom(cm, labels, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("--- Starting SNR-Stratified Classification Analysis ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved to '{OUTPUT_DIR}/'")

    # --- 1. Load Models and Data ---
    print("Loading models and data...")
    scaler = joblib.load(SCALER_PATH)
    gen_model = joblib.load(GEN_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    le = joblib.load(LE_PATH)
    
    X_test = np.load(X_TEST_PATH)
    y_test_str = np.load(Y_TEST_PATH)
    snr_test = np.load(SNR_TEST_PATH)
    unique_snrs = np.unique(snr_test)

    # --- 2. Get Full Test Set Predictions ---
    print("Generating predictions for the full test set...")
    X_test_scaled = scaler.transform(X_test)
    X_test_meta = create_meta_features(gen_model, X_test_scaled)
    y_prob_test = xgb_model.predict_proba(X_test_meta)
    y_pred_int = np.argmax(y_prob_test, axis=1)
    y_pred_str = le.inverse_transform(y_pred_int)
    margins_test = calculate_log_margins(y_prob_test)

    # --- 3. Per-SNR Analysis ---
    print("Performing analysis for each SNR level...")
    acc_vs_snr = {}
    margin_stats_vs_snr = {}
    all_misclassifications = []

    for snr in unique_snrs:
        snr_mask = (snr_test == snr)
        y_true_snr = y_test_str[snr_mask]
        y_pred_snr = y_pred_str[snr_mask]
        
        # 1. Compute accuracy
        acc = accuracy_score(y_true_snr, y_pred_snr)
        acc_vs_snr[snr] = acc
        
        # 2/3. Compute confusion matrix and misclassification probabilities
        cm = confusion_matrix(y_true_snr, y_pred_snr, labels=le.classes_)
        cm_prob = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        for i, true_mod in enumerate(le.classes_):
            for j, pred_mod in enumerate(le.classes_):
                if i != j and cm_prob[i, j] > 0:
                    all_misclassifications.append((snr, true_mod, pred_mod, cm_prob[i, j]))

        # 6. Analyze confidence margins
        margins_snr = margins_test[snr_mask]
        margin_stats_vs_snr[snr] = {
            'mean': np.mean(margins_snr),
            'std': np.std(margins_snr)
        }

    # --- 4. Aggregated Analysis ---
    # 4. Identify most confused pairs
    misclass_df = pd.DataFrame(all_misclassifications, columns=['SNR', 'True', 'Predicted', 'Probability'])
    top_confused = misclass_df.groupby(['True', 'Predicted'])['Probability'].mean().sort_values(ascending=False)
    print("\n--- Top 15 Most Confused Modulation Pairs (Averaged across SNRs) ---")
    print(top_confused.head(15))

    # --- 5. Generate and Save Plots ---
    print("\n--- Generating and Saving Plots ---")
    
    # Plot 1: Accuracy vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(list(acc_vs_snr.keys()), list(acc_vs_snr.values()), marker='o')
    plt.title("Accuracy vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_snr.png"))
    plt.close()
    print("Saved: accuracy_vs_snr.png")

    # Plot 2/3: Confusion Matrix for High and Low SNR
    low_snr_mask = snr_test < SNR_THRESHOLD
    high_snr_mask = snr_test >= SNR_THRESHOLD

    cm_low = confusion_matrix(y_test_str[low_snr_mask], y_pred_str[low_snr_mask], labels=le.classes_, normalize='true')
    plot_confusion_matrix_custom(cm_low, le.classes_, f"Confusion Matrix (SNR < {SNR_THRESHOLD} dB)", os.path.join(OUTPUT_DIR, "cm_low_snr.png"))
    print("Saved: cm_low_snr.png")

    cm_high = confusion_matrix(y_test_str[high_snr_mask], y_pred_str[high_snr_mask], labels=le.classes_, normalize='true')
    plot_confusion_matrix_custom(cm_high, le.classes_, f"Confusion Matrix (SNR >= {SNR_THRESHOLD} dB)", os.path.join(OUTPUT_DIR, "cm_high_snr.png"))
    print("Saved: cm_high_snr.png")

    # Plot 4: Top confused modulation pairs
    plt.figure(figsize=(12, 8))
    top_confused.head(15).sort_values().plot(kind='barh')
    plt.title("Top 15 Most Confused Pairs (Avg. Misclassification Probability)")
    plt.xlabel("Average Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_confused_pairs.png"))
    plt.close()
    print("Saved: top_confused_pairs.png")

    # Plot 5: Margin vs SNR
    mean_margins = [margin_stats_vs_snr[snr]['mean'] for snr in unique_snrs]
    std_margins = [margin_stats_vs_snr[snr]['std'] for snr in unique_snrs]
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, mean_margins, marker='o', label='Mean Margin')
    plt.fill_between(unique_snrs, np.array(mean_margins) - np.array(std_margins), np.array(mean_margins) + np.array(std_margins), alpha=0.2, label='+/- 1 Std Dev')
    plt.title("Confidence Margin vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Log-Probability Margin")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "margin_vs_snr.png"))
    plt.close()
    print("Saved: margin_vs_snr.png")

    # 5. Family-level confusion
    family_order = ['PSK', 'QAM', 'PAM', 'FSK', 'Analog']
    y_true_family = pd.Series(y_test_str).map(MOD_FAMILIES).values
    y_pred_family = pd.Series(y_pred_str).map(MOD_FAMILIES).values
    
    cm_family_high = confusion_matrix(y_true_family[high_snr_mask], y_pred_family[high_snr_mask], labels=family_order, normalize='true')
    plot_confusion_matrix_custom(cm_family_high, family_order, f"Family Confusion Matrix (SNR >= {SNR_THRESHOLD} dB)", os.path.join(OUTPUT_DIR, "cm_family_high_snr.png"))
    print("Saved: cm_family_high_snr.png")

if __name__ == "__main__":
    main()
