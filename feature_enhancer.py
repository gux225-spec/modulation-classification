import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans

# --- 辅助函数 ---
EPS = 1e-9

def _calculate_stats(data: np.ndarray, prefix: str) -> dict:
    """计算一维数组的基本统计量"""
    if data.size == 0:
        return {
            f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0,
            f'{prefix}_skew': 0.0, f'{prefix}_kurt': 0.0
        }
    return {
        f'{prefix}_mean': np.mean(data),
        f'{prefix}_std': np.std(data),
        f'{prefix}_skew': skew(data),
        f'{prefix}_kurt': kurtosis(data, fisher=False) # Pearson's kurtosis
    }

def _histogram_features(data: np.ndarray, bins: int, prefix: str) -> dict:
    """从直方图中提取熵、峰值度、峰数等特征"""
    if data.size == 0:
        return {f'{prefix}_hist_entropy': 0.0, f'{prefix}_hist_peakiness': 0.0, f'{prefix}_hist_num_peaks': 0.0}
        
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist_norm = hist * np.diff(_)[0] # 归一化使和为1
    
    # 熵
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
    
    # 峰值度
    peakiness = np.max(hist_norm) if hist_norm.size > 0 else 0.0
    
    # 峰数 (简单的局部最大值)
    # 忽略边缘，寻找 hist[i-1] < hist[i] > hist[i+1]
    peaks = 0
    if hist_norm.size > 2:
        for i in range(1, len(hist_norm) - 1):
            if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]:
                peaks += 1
    
    return {
        f'{prefix}_hist_entropy': entropy,
        f'{prefix}_hist_peakiness': peakiness,
        f'{prefix}_hist_num_peaks': float(peaks)
    }

# --- 主特征提取函数 ---

def extract_disambiguation_features(x_2x128: np.ndarray) -> np.ndarray:
    """
    为单个IQ样本(2, 128)提取专门用于区分混淆对的补充特征。

    Args:
        x_2x128 (np.ndarray): 输入的单个IQ样本，形状为 (2, 128)。

    Returns:
        np.ndarray: 包含所有新特征的一维numpy向量。
    """
    s = x_2x128[0, :] + 1j * x_2x128[1, :]
    features = {}

    # A) QAM16 vs QAM64 (幅度特征)
    r = np.abs(s)
    q = np.quantile(r, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    features.update({
        'amp_q05': q[0], 'amp_q10': q[1], 'amp_q25': q[2], 'amp_q50': q[3],
        'amp_q75': q[4], 'amp_q90': q[5], 'amp_q95': q[6]
    })
    features['amp_iqr_90_10'] = q[5] - q[1]
    features['amp_iqr_75_25'] = q[4] - q[2]
    features.update(_histogram_features(r, bins=32, prefix='amp'))

    # B) QPSK vs 8PSK (相位特征)
    phi = np.unwrap(np.angle(s))
    dphi = np.diff(phi)
    dphi_wrapped = np.angle(np.exp(1j * dphi)) # wrap to [-pi, pi]
    
    # Circular stats
    R = np.abs(np.mean(np.exp(1j * phi)))
    features['phase_circ_var'] = 1 - R
    features['phase_circ_mean_res_len'] = R
    
    # M-th power features
    for M in [4, 8]:
        phi_m = np.angle(np.exp(1j * M * phi))
        features[f'phase_M{M}_abs_mean'] = np.abs(np.mean(np.exp(1j * phi_m)))
        
        s_m_fft = np.fft.fftshift(np.fft.fft(s**M))
        s_m_power = np.abs(s_m_fft)**2
        total_power = np.sum(s_m_power)
        peak_power = np.max(s_m_power) if total_power > EPS else 0.0
        features[f'phase_M{M}_fft_peak_ratio'] = peak_power / (total_power + EPS)

    # C) WBFM vs AM-DSB (瞬时频率 vs 包络)
    inst_freq = dphi_wrapped / (2 * np.pi) # 归一化
    features.update(_calculate_stats(inst_freq, 'inst_freq'))
    features.update(_histogram_features(inst_freq, bins=16, prefix='inst_freq'))
    
    envelope = r
    features.update(_calculate_stats(envelope, 'envelope'))
    features['envelope_mod_depth'] = np.std(envelope) / (np.mean(envelope) + EPS)
    
    # Occupied bandwidth
    fft_power = np.abs(np.fft.fftshift(np.fft.fft(s)))**2
    cum_power = np.cumsum(fft_power)
    total_power = cum_power[-1]
    try:
        idx_95 = np.where(cum_power >= 0.95 * total_power)[0][0]
        idx_05 = np.where(cum_power >= 0.05 * total_power)[0][0]
        occupied_bw_90 = (idx_95 - idx_05) / len(fft_power)
    except IndexError:
        occupied_bw_90 = 1.0
    features['occupied_bw_90'] = occupied_bw_90

    # D) BPSK vs PAM4 (I分量幅度特征)
    I = x_2x128[0, :]
    features.update(_histogram_features(I, bins=16, prefix='I'))
    features['I_kurt'] = kurtosis(I, fisher=False)

    # 按key排序以保证特征顺序恒定
    sorted_features = sorted(features.items())
    return np.array([v for k, v in sorted_features])

def augment_features(Z_base: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    在基础特征矩阵上增强补充特征。

    Args:
        Z_base (np.ndarray): 基础特征矩阵，形状为 (N, D_base)。
        X (np.ndarray): 原始IQ数据，形状为 (N, 2, 128)。

    Returns:
        np.ndarray: 拼接了新特征的矩阵，形状为 (N, D_base + D_new)。
    """
    num_samples = X.shape[0]
    
    # 使用列表推导式并行化（或简单循环）提取新特征
    # 注意：如果样本量巨大，可以考虑使用 joblib.Parallel
    Z_new_list = [extract_disambiguation_features(X[i]) for i in range(num_samples)]
    
    Z_new = np.array(Z_new_list)
    
    # 确保 Z_new 是二维的
    if Z_new.ndim == 1:
        Z_new = Z_new.reshape(-1, 1)
        
    if Z_new.shape[0] != num_samples:
        raise ValueError("The number of samples in new features does not match original.")

    return np.hstack([Z_base, Z_new])
