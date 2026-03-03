import numpy as np
import pywt

EPS = 1e-12

# ----------------------------
# Basic stats helpers
# ----------------------------
def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std()
    # If std is zero, skewness is not well-defined, return 0
    if sd < EPS:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))

def _kurt(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std()
    # If std is zero, kurtosis is not well-defined, return 0
    if sd < EPS:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4))

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c) or np.isinf(c):
        return 0.0
    return float(c)

# ----------------------------
# Preprocessing (zero-center + normalize)
# ----------------------------
def preprocess_iq(I: np.ndarray, Q: np.ndarray, mode: str = "zero_mean_rms") -> np.ndarray:
    """
    mode options:
      - "none": no preprocessing
      - "zero_mean_rms": remove DC and normalize by complex RMS energy
      - "zero_mean_std": remove DC and normalize I/Q separately by std
    Return: complex baseband s (length 128)
    """
    I = np.asarray(I, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if mode in ("zero_mean_rms", "zero_mean_std"):
        I = I - I.mean()
        Q = Q - Q.mean()

    if mode == "none":
        return I + 1j * Q

    if mode == "zero_mean_rms":
        s = I + 1j * Q
        rms = np.sqrt(np.mean(np.abs(s) ** 2)) + EPS
        return s / rms

    if mode == "zero_mean_std":
        I = I / (I.std() + EPS)
        Q = Q / (Q.std() + EPS)
        return I + 1j * Q

    raise ValueError(f"Unknown preprocess mode: {mode}")

# ----------------------------
# Multi-scale FFT spectrum features
# ----------------------------
def multi_scale_spectrum_features(s: np.ndarray, scales=(2, 4, 8)) -> list:
    """
    Multi-scale spectrum features by binning FFT magnitude spectrum into k groups.
    For each k in scales, compute:
      - entropy
      - peak_num (bins > mean + 2*std)
      - top1
      - top2
    Return list length = len(scales) * 4
    """
    mag = np.abs(np.fft.fft(s))
    mag = mag / (mag.sum() + EPS)

    feats = []
    for k in scales:
        bins = np.array_split(mag, k)
        coarse = np.array([b.mean() for b in bins], dtype=np.float64)
        coarse = coarse / (coarse.sum() + EPS)

        ent = -np.sum(coarse * np.log(coarse + EPS))
        thr = coarse.mean() + 2 * coarse.std()
        peak_num = int(np.sum(coarse > thr))
        top1 = float(np.max(coarse))
        top2 = float(np.partition(coarse, -2)[-2])

        feats.extend([float(ent), float(peak_num), top1, top2])

    return feats

# ----------------------------
# Wavelet features
# ----------------------------
def wavelet_features(s: np.ndarray, wavelet: str = 'db4', level: int = 4) -> list:
    """
    Extracts wavelet features from the complex signal.
    For each decomposition level, compute for both I and Q components:
      - Energy of approximation coefficients (cA)
      - Energy of detail coefficients (cD)
      - Entropy of cA
      - Entropy of cD
    Return list length = level * 4 (I) + level * 4 (Q)
    """
    # Decompose I and Q components separately
    I = np.real(s)
    Q = np.imag(s)

    def _get_feats(x: np.ndarray, wavelet: str, level: int) -> list:
        coeffs = pywt.wavedec(x, wavelet, level=level)
        feats = []
        for c in coeffs:
            # Energy
            energy = np.sum(c**2)
            # Entropy
            c_norm = c**2 / (energy + EPS)
            entropy = -np.sum(c_norm * np.log2(c_norm + EPS))
            feats.extend([energy, entropy])
        return feats

    feats_i = _get_feats(I, wavelet, level)
    feats_q = _get_feats(Q, wavelet, level)

    return feats_i + feats_q

# ----------------------------
# Main feature extraction
# ----------------------------
def extract_features(x_2x128: np.ndarray,
                     preprocess_mode: str = "zero_mean_rms",
                     ms_scales=(2, 4, 8)) -> np.ndarray:
    """
    Input: x_2x128 shape (2, 128), where x[0]=I, x[1]=Q
    Output: feature vector (float32), shape (D,)
      D = 24 (base) + 4*len(ms_scales) (multi-scale FFT)
        = 24 + 12 = 36 (default)
    """
    x_2x128 = np.asarray(x_2x128)
    assert x_2x128.shape == (2, 128), f"Expected (2,128), got {x_2x128.shape}"

    I_raw = x_2x128[0]
    Q_raw = x_2x128[1]

    # preprocessing
    s = preprocess_iq(I_raw, Q_raw, mode=preprocess_mode)

    # If you still want raw I/Q stats too, use preprocessed real/imag:
    I = np.real(s)
    Q = np.imag(s)

    amp = np.abs(s)               # amplitude
    pwr = amp ** 2                # power

    # phase / inst. phase increment (proxy for inst. freq)
    ph = np.unwrap(np.angle(s))
    dph = np.diff(ph)

    # FFT magnitude spectrum (full resolution)
    S = np.fft.fft(s)
    mag = np.abs(S)
    pmag = mag / (mag.sum() + EPS)  # normalized spectrum "probability"

    # spectral entropy (full resolution)
    spec_entropy = float(-np.sum(pmag * np.log(pmag + EPS)))

    # peak count (full resolution)
    thr = pmag.mean() + 2 * pmag.std()
    peak_num = int(np.sum(pmag > thr))

    # spectral centroid (index-weighted average)
    k = np.arange(len(pmag), dtype=np.float64)
    spec_centroid = float(np.sum(k * pmag))

    # top peaks (full resolution)
    top1 = float(np.max(pmag))
    top2 = float(np.partition(pmag, -2)[-2])

    # time-domain IQ stats (on preprocessed I/Q)
    corr_iq = _safe_corr(I, Q)

    # amplitude modulation depth (rough) -- on preprocessed amp
    am_depth = float((amp.max() - amp.min()) / (amp.mean() + EPS))

    # multi-scale FFT features (2/4/8 bins)
    ms_feats = multi_scale_spectrum_features(s, scales=ms_scales)

    # wavelet features
    wv_feats = wavelet_features(s, wavelet='db4', level=4)

    feats = [
        # amplitude stats
        float(amp.mean()), float(amp.std()), _skew(amp), _kurt(amp),
        # power stats
        float(pwr.mean()), float(pwr.std()),
        # peak-to-rms
        float(amp.max() / (np.sqrt(np.mean(amp**2)) + EPS)),
        # AM depth
        am_depth,
        # phase stats
        float(ph.mean()), float(ph.std()),
        # dphase stats
        float(dph.mean()), float(dph.std()), _skew(dph), _kurt(dph),
        # I/Q stats
        float(I.mean()), float(I.std()), float(Q.mean()), float(Q.std()),
        corr_iq,
        # spectrum stats (full-res)
        spec_entropy, float(peak_num), spec_centroid, top1, top2,
        # multi-scale spectrum stats
        *ms_feats,
        # wavelet stats
        *wv_feats
    ]

    f = np.array(feats, dtype=np.float32)

    # sanitize
    if not np.all(np.isfinite(f)):
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return f
