import numpy as np

# feature indices (based on current feature_extraction.py)
# base feats (0..23)
IDX = {
    "am_depth": 7,        # AM depth
    "dph_std": 11,        # std(diff(unwrap(angle)))
    "peak_num_full": 20,  # full-res peak count
}

# multi-scale feats start at 24:
# for each scale: [ent, peak_num, top1, top2]
MS_START = 24
def ms_idx(scale_pos: int, field_pos: int) -> int:
    """
    scale_pos: 0 for k=2, 1 for k=4, 2 for k=8
    field_pos: 0 ent, 1 peak_num, 2 top1, 3 top2
    """
    return MS_START + scale_pos * 4 + field_pos

IDX["peak_num_k8"] = ms_idx(2, 1)  # k=8 peak_num (most robust)

def gate_by_mod_type(
    X_feat: np.ndarray,
    thr_am_depth: float = 0.55,
    thr_dph_std: float = 2.0,
    thr_peak_k8: float = 1.0,
):
    """
    Output group_id:
      0: AM-like      (AM-DSB, PAM4 mostly)
      1: Freq-like    (GFSK, CPFSK, WBFM mostly)
      2: Phase/QAM-like (BPSK, QPSK, 8PSK, QAM16/64 mostly)
    """
    X = np.asarray(X_feat, dtype=np.float32)

    am_depth = X[:, IDX["am_depth"]]
    dph_std  = X[:, IDX["dph_std"]]
    peak_k8  = X[:, IDX["peak_num_k8"]]

    group = np.full(X.shape[0], 2, dtype=np.int8)  # default: phase/qam-like

    # AM-like
    group[am_depth > thr_am_depth] = 0

    # Freq-like (robust): large dph std OR coarse peak count
    freq_mask = (dph_std > thr_dph_std) | (peak_k8 > thr_peak_k8)
    group[(group != 0) & freq_mask] = 1

    return group

def soft_gate_by_snr(
    snr: np.ndarray,
    tau1: float = -2.0,
    tau2: float = 10.0,
) -> np.ndarray:
    """
    Calculates a soft gating weight based on SNR.
    The weight `w` transitions linearly from 0 to 1 as SNR goes from tau1 to tau2.

    Args:
        snr: A numpy array of SNR values for each sample.
        tau1: The lower SNR threshold. Below this, weight is 0.
        tau2: The upper SNR threshold. Above this, weight is 1.

    Returns:
        A numpy array of weights `w` for each sample, shape (n_samples,).
    """
    snr = np.asarray(snr, dtype=np.float32)
    
    # Ensure tau2 is greater than tau1 to avoid division by zero
    if tau2 <= tau1:
        raise ValueError("tau2 must be greater than tau1")

    # Calculate the weight using the formula
    w = (snr - tau1) / (tau2 - tau1)

    # Clip the weights to be within the [0, 1] range
    w_clipped = np.clip(w, 0.0, 1.0)

    return w_clipped.astype(np.float32)
