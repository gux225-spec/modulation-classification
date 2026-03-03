import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Family-based Abstention Definitions ---

MOD_FAMILIES = {
    'BPSK': 'PSK', 'QPSK': 'PSK', '8PSK': 'PSK',
    'QAM16': 'QAM', 'QAM64': 'QAM',
    'GFSK': 'FSK', 'CPFSK': 'FSK',
    'AM-DSB': 'Analog', 'WBFM': 'Analog',
    'PAM4': 'PAM'
}

# Recommended target coverages by family
FAMILY_TARGET_COVERAGE = {
    'PSK': 0.7,
    'QAM': 0.4,
    'FSK': 0.7,
    'Analog': 0.5,
    'PAM': 0.7
}


# --- 核心计算函数 ---

EPS = 1e-12

def calculate_log_margins(y_prob: np.ndarray) -> np.ndarray:
    """
    在对数空间中计算 top-1 和 top-2 概率之间的差值（margin）。

    Args:
        y_prob (np.ndarray): 模型输出的概率矩阵 (N, K)。

    Returns:
        np.ndarray: 每个样本的对数 margin (N,)。
    """
    # 使用 log(proba) 增加数值稳定性
    log_prob = np.log(y_prob + EPS)
    
    # 高效找到 top-1 和 top-2 的值
    # np.partition 将第 k-1 大的元素放在它最终排序后的位置
    # 对于 top-2，我们需要第 K-2 大的元素
    part = np.partition(log_prob, -2, axis=1)
    top1_log_prob = part[:, -1]
    top2_log_prob = part[:, -2]
    
    return top1_log_prob - top2_log_prob

def select_family_thresholds(
    y_prob_val: np.ndarray,
    y_true_val_str: np.ndarray,
    mod_families: dict,
    target_coverages: dict
) -> dict:
    """
    Calculates a dictionary of abstention thresholds, one for each modulation family.

    Args:
        y_prob_val (np.ndarray): Predicted probabilities for the validation set.
        y_true_val_str (np.ndarray): True string labels for the validation set.
        mod_families (dict): Mapping from modulation type string to family string.
        target_coverages (dict): Mapping from family string to target coverage float.

    Returns:
        dict: A dictionary mapping family name to its calculated threshold `tau`.
    """
    margins_val = calculate_log_margins(y_prob_val)
    family_thresholds = {}

    all_families = set(mod_families.values())
    for family in all_families:
        # Create a mask for samples belonging to the current family
        family_mask = np.array([mod_families.get(mod, 'Unknown') == family for mod in y_true_val_str])

        if np.sum(family_mask) == 0:
            print(f"Warning: No validation samples found for family '{family}'. Using global threshold as fallback.")
            continue

        family_margins = margins_val[family_mask]
        target_coverage = target_coverages.get(family, 0.5) # Default coverage if not specified
        
        tau = np.quantile(family_margins, 1.0 - target_coverage)
        family_thresholds[family] = float(tau)

    # Fallback for any family that had no samples
    if len(family_thresholds) < len(all_families):
        global_tau = np.quantile(margins_val, 1.0 - np.mean(list(target_coverages.values())))
        for family in all_families:
            if family not in family_thresholds:
                family_thresholds[family] = float(global_tau)

    return family_thresholds


def apply_family_thresholds(
    y_prob: np.ndarray,
    family_thresholds: dict,
    le: LabelEncoder,
    mod_families: dict,
    unknown_label: int = -1
) -> np.ndarray:
    """
    Applies a conservative abstention rule using family-specific thresholds based on
    the top-1 and top-2 predicted classes.

    The effective threshold for a sample is max(tau_family[top1], tau_family[top2]).

    Args:
        y_prob (np.ndarray): Model output probabilities (N, K).
        family_thresholds (dict): Dictionary mapping family name to its threshold.
        le (LabelEncoder): The label encoder to map class indices to names.
        mod_families (dict): Mapping from modulation type string to family string.
        unknown_label (int, optional): Label for abstention. Defaults to -1.

    Returns:
        np.ndarray: Prediction labels with abstentions (N,).
    """
    margins = calculate_log_margins(y_prob)
    # Get indices of top 2 predictions for each sample
    top_indices = np.argsort(y_prob, axis=1)[:, -2:]
    top1_indices = top_indices[:, 1]
    top2_indices = top_indices[:, 0]

    y_pred_abstained = top1_indices.copy()

    for i in range(len(y_prob)):
        # Get families of top-1 and top-2 predictions
        top1_mod_str = le.classes_[top1_indices[i]]
        top2_mod_str = le.classes_[top2_indices[i]]
        
        top1_family = mod_families.get(top1_mod_str, 'Unknown')
        top2_family = mod_families.get(top2_mod_str, 'Unknown')

        # Get thresholds for both families
        tau1 = family_thresholds.get(top1_family, float('inf'))
        tau2 = family_thresholds.get(top2_family, float('inf'))

        # Use the more conservative (higher) threshold
        tau_sample = max(tau1, tau2)

        if margins[i] < tau_sample:
            y_pred_abstained[i] = unknown_label
            
    return y_pred_abstained

# --- 绘图与评估函数 ---

def plot_confusion_matrix_with_abstention(
    y_true: np.ndarray, 
    y_pred_abstained: np.ndarray, 
    le: LabelEncoder,
    unknown_label: int = -1,
    save_path: str = None
):
    """
    绘制拒识后的混淆矩阵，只统计非拒识样本。

    Args:
        y_true (np.ndarray): 真实标签 (N,)。
        y_pred_abstained (np.ndarray): 应用拒识后的预测标签 (N,)。
        le (LabelEncoder): 标签编码器，用于获取类别名称。
        unknown_label (int, optional): 代表拒识的标签。默认为 -1。
    """
    mask = y_pred_abstained != unknown_label
    coverage = np.mean(mask)
    
    cm = confusion_matrix(y_true[mask], y_pred_abstained[mask])
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + EPS)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    class_names = le.classes_
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Normalized Confusion Matrix (Coverage: {coverage:.2%})',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 循环遍历数据并添加文本
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_evaluation_curves(
    # For SNR plot (left)
    y_prob_test: np.ndarray, 
    y_true_test: np.ndarray, 
    snrs_test: np.ndarray, 
    y_pred_abstained_test: np.ndarray,
    # For Trade-off curve (right)
    y_prob_val: np.ndarray, 
    y_true_val: np.ndarray,
    # General
    unknown_label: int = -1,
    save_path: str = None
):
    """
    Plots two key evaluation curves in a single figure:
    1. Left Subplot: Performance (Overall Acc, Selective Acc, Coverage) vs. SNR.
    2. Right Subplot: Coverage-Accuracy Trade-off curve.

    Args:
        y_prob_test (np.ndarray): Predicted probabilities for the test set.
        y_true_test (np.ndarray): True labels for the test set.
        snrs_test (np.ndarray): SNR for each sample in the test set.
        y_pred_abstained_test (np.ndarray): Abstention results for the test set.
        y_prob_val (np.ndarray): Predicted probabilities for the validation set.
        y_true_val (np.ndarray): True labels for the validation set.
        unknown_label (int, optional): Label for abstention. Defaults to -1.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Left Subplot: Performance vs. SNR ---
    unique_snrs = np.unique(snrs_test)
    overall_acc_vs_snr = []
    selective_acc_vs_snr = []
    coverage_vs_snr = []

    y_pred_test_full = np.argmax(y_prob_test, axis=1)
    accepted_mask_test = (y_pred_abstained_test != unknown_label)

    for snr in unique_snrs:
        snr_mask = (snrs_test == snr)
        
        # Overall Accuracy
        overall_acc_vs_snr.append(accuracy_score(y_true_test[snr_mask], y_pred_test_full[snr_mask]))
        
        # Selective Accuracy & Coverage using the pre-calculated abstention results
        accepted_in_snr_mask = accepted_mask_test[snr_mask]
        coverage = np.mean(accepted_in_snr_mask)
        coverage_vs_snr.append(coverage)
        
        if np.sum(accepted_in_snr_mask) > 0:
            selective_acc_vs_snr.append(
                accuracy_score(y_true_test[snr_mask][accepted_in_snr_mask], y_pred_test_full[snr_mask][accepted_in_snr_mask])
            )
        else:
            selective_acc_vs_snr.append(0.0)

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Accuracy')
    ax1.plot(unique_snrs, overall_acc_vs_snr, '^-', label='Overall Accuracy', color='tab:blue')
    ax1.plot(unique_snrs, selective_acc_vs_snr, 'o-', label='Selective Accuracy', color='tab:orange')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--')

    # Use a secondary y-axis for coverage
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Coverage', color='tab:green')
    ax1_twin.plot(unique_snrs, coverage_vs_snr, 'x:', label='Coverage', color='tab:green')
    ax1_twin.tick_params(axis='y', labelcolor='tab:green')
    ax1_twin.set_ylim(0, 1.05)
    ax1.set_title('Performance vs. SNR (on Test Set)')

    # --- Right Subplot: Coverage-Accuracy Trade-off ---
    margins_val = calculate_log_margins(y_prob_val)
    y_pred_val_full = np.argmax(y_prob_val, axis=1)
    
    coverage_points = []
    accuracy_points = []
    
    # Sweep tau over the quantiles of the validation margins
    taus_sweep = np.quantile(margins_val, np.linspace(0, 1, 101))
    
    for t in taus_sweep:
        mask = margins_val >= t
        coverage = np.mean(mask)
        
        if coverage > 0:
            accuracy = accuracy_score(y_true_val[mask], y_pred_val_full[mask])
            coverage_points.append(coverage)
            accuracy_points.append(accuracy)
            
    ax2.plot(coverage_points, accuracy_points, marker='.', linestyle='-')
    ax2.set_xlabel('Coverage')
    ax2.set_ylabel('Selective Accuracy')
    ax2.set_title('Coverage-Accuracy Trade-off (on Validation Set)')
    ax2.grid(True, linestyle='--')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(min(accuracy_points) - 0.05 if accuracy_points else 0, 1.05)
    ax2.invert_xaxis()

    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Evaluation curves saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def print_evaluation_summary(y_true: np.ndarray, y_pred_full: np.ndarray, y_pred_abstained: np.ndarray, unknown_label: int = -1):
    """
    打印一个完整的评估指标摘要。

    Args:
        y_true (np.ndarray): 真实标签。
        y_pred_full (np.ndarray): 未经拒识的预测。
        y_pred_abstained (np.ndarray): 经过拒识的预测。
        unknown_label (int, optional): 拒识标签。
    """
    mask = y_pred_abstained != unknown_label
    
    overall_accuracy = accuracy_score(y_true, y_pred_full)
    coverage = np.mean(mask)
    selective_accuracy = accuracy_score(y_true[mask], y_pred_abstained[mask]) if coverage > 0 else 0.0
    
    # 将拒识样本视为错误，计算整体准确率
    y_pred_abstained_as_wrong = y_pred_abstained.copy()
    overall_with_unknown_as_wrong = accuracy_score(y_true, y_pred_abstained_as_wrong)

    print("--- Evaluation Summary ---")
    print(f"Without Abstention:")
    print(f"  - Overall Accuracy: {overall_accuracy:.4f}")
    print("\nWith Abstention:")
    print(f"  - Coverage: {coverage:.4f} ({np.sum(mask)} / {len(y_true)} samples)")
    print(f"  - Selective Accuracy (on non-abstained): {selective_accuracy:.4f}")
    print(f"  - Overall Accuracy (unknowns as wrong): {overall_with_unknown_as_wrong:.4f}")
    print("--------------------------")
