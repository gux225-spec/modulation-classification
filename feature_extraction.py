
import numpy as np
import pywt

# Epsilon for numerical stability
EPS = 1e-9

def _histogram_entropy(x: np.ndarray, bins: int = 10) -> float:
    """Calculate the Shannon entropy of a signal's histogram."""
    counts, _ = np.histogram(x, bins=bins, density=True)
    pk = counts / (np.sum(counts) + EPS)
    return -np.sum(pk * np.log2(pk + EPS))

def _analog_disambiguation_features(amp: np.ndarray, dphase: np.ndarray) -> list[float]:
    """Features to distinguish WBFM (frequency varies) vs AM-DSB (amplitude varies)."""
    # Envelope/Amplitude Features (high for AM, low for FM)
    amp_cov = np.std(amp) / (np.mean(amp) + EPS)
    papr = np.max(amp**2) / (np.mean(amp**2) + EPS)
    amp_iqr = np.quantile(amp, 0.75) - np.quantile(amp, 0.25)
    amp_entropy = _histogram_entropy(amp)

    # Instantaneous Frequency/dphase Features (high for FM, low for AM)
    dphase_abs = np.abs(dphase)
    dphase_abs_mean = np.mean(dphase_abs)
    dphase_abs_std = np.std(dphase_abs)
    dphase_abs_max = np.max(dphase_abs)
    dphase_sign_change_rate = np.mean(np.diff(np.sign(dphase)) != 0)
    dphase_entropy = _histogram_entropy(dphase)

    # Coupling Features (measures if envelope and frequency are related)
    power = amp**2
    corr_amp_dphase = np.corrcoef(amp, dphase_abs)[0, 1]
    corr_power_dphase = np.corrcoef(power, dphase_abs)[0, 1]

    return [
        amp_cov, papr, amp_iqr, amp_entropy,
        dphase_abs_mean, dphase_abs_std, dphase_abs_max,
        dphase_sign_change_rate, dphase_entropy,
        corr_amp_dphase, corr_power_dphase
    ]

def _qam_radius_features(c_norm: np.ndarray, amp: np.ndarray) -> list[float]:
    """Features to distinguish QAM16 vs QAM64 based on constellation geometry."""
    # Amplitude/Radius Distribution Features
    q = np.quantile(amp, [0.1, 0.25, 0.5, 0.75, 0.9])
    q10, q25, q50, q75, q90 = q.tolist()
    gap_iqr = q75 - q25
    gap_outer = q90 - q10
    norm_radius_var = np.var(amp)
    amp_entropy = _histogram_entropy(amp)
    amp_hist_counts, _ = np.histogram(amp, bins=4, range=(0, 2.0))
    occupied_bins = np.sum(amp_hist_counts > 0)

    # Higher-Order Complex Moment Features
    m20 = np.mean(c_norm**2)
    m21 = np.mean(c_norm * np.conj(c_norm))
    m40 = np.mean(c_norm**4)
    m42 = np.mean(c_norm**2 * np.conj(c_norm)**2) # Correct complex definition

    feat_m20_norm = np.abs(m20) / (m21 + EPS)
    feat_m40_norm = np.abs(m40) / (m21**2 + EPS)
    
    # For the complex moment m42, use its real and imaginary parts as separate features
    m42_norm = m42 / (m21**2 + EPS)
    feat_m42_real = np.real(m42_norm)
    feat_m42_imag = np.imag(m42_norm)

    return [
        q10, q25, q50, q75, q90, gap_iqr, gap_outer,
        norm_radius_var, amp_entropy, occupied_bins,
        feat_m20_norm, feat_m40_norm, feat_m42_real, feat_m42_imag
    ]

def _iq_geometry_features(i: np.ndarray, q: np.ndarray) -> list[float]:
    """Auxiliary features based on I/Q component geometry."""
    i_energy = np.sum(i**2)
    q_energy = np.sum(q**2)
    iq_energy_ratio = i_energy / (q_energy + EPS)
    iq_energy_diff_norm = (i_energy - q_energy) / (i_energy + q_energy + EPS)
    iq_std_ratio = np.std(i) / (np.std(q) + EPS)
    iq_corr = np.corrcoef(i, q)[0, 1]
    
    cov_matrix = np.cov(i, q)
    # Use eigvalsh for real symmetric matrices, guarantees real eigenvalues
    eigvals = np.linalg.eigvalsh(cov_matrix)
    cov_eig_ratio = np.max(eigvals) / (np.min(eigvals) + EPS)
    
    return [iq_energy_ratio, iq_energy_diff_norm, iq_std_ratio, iq_corr, cov_eig_ratio]

def _wavelet_features(x: np.ndarray, wavelet: str = 'db4', max_level: int = 4) -> list[float]:
    """Calculates compact wavelet features (energy and entropy)."""
    try:
        coeffs_i = pywt.wavedec(x[0, :], wavelet, level=max_level)
        coeffs_q = pywt.wavedec(x[1, :], wavelet, level=max_level)
        features = []
        for d_i, d_q in zip(coeffs_i[1:], coeffs_q[1:]):
            d_mag = np.abs(d_i + 1j * d_q)
            features.append(np.sum(d_mag**2))
            features.append(_histogram_entropy(d_mag))
        return features
    except ImportError:
        return []

def extract_features(
    x: np.ndarray,
    add_analog_features: bool = True,
    add_qam_features: bool = True,
    add_iq_features: bool = True,
    add_wavelet: bool = True
) -> np.ndarray:
    """
    Main feature extraction function, refactored for targeted feature addition.
    """
    # --- 1. Preprocessing ---
    i, q = x[0, :], x[1, :]
    c = i + 1j * q
    c -= np.mean(c)
    c_norm = c / np.sqrt(np.mean(np.abs(c)**2) + EPS)
    
    amplitude = np.abs(c_norm)
    phase_unwrapped = np.unwrap(np.angle(c_norm))
    dphase = np.diff(phase_unwrapped)
    dphase = np.concatenate(([dphase[0]], dphase))
    
    fft_mag = np.abs(np.fft.fft(c_norm, n=128))
    fft_mag_shifted = np.fft.fftshift(fft_mag)

    # --- 2. Baseline Feature Calculation (Curated) ---
    baseline_features = [
        np.std(amplitude),
        np.std(phase_unwrapped),
        np.mean(np.abs(dphase)),
        np.std(dphase),
        np.mean(fft_mag_shifted),
        np.std(fft_mag_shifted),
        np.exp(np.mean(np.log(fft_mag_shifted + EPS))) / (np.mean(fft_mag_shifted) + EPS)
    ]

    # --- 3. Targeted & Optional Feature Addition ---
    all_features = baseline_features.copy()
    if add_analog_features:
        all_features.extend(_analog_disambiguation_features(amplitude, dphase))
    if add_qam_features:
        all_features.extend(_qam_radius_features(c_norm, amplitude))
    if add_iq_features:
        all_features.extend(_iq_geometry_features(i, q))
    if add_wavelet:
        all_features.extend(_wavelet_features(x))

    # --- 4. Final Sanitization ---
    feature_vector = np.array(all_features, dtype=np.float32)
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
