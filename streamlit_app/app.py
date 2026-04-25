from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="AMC Project Demo",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PLOT_DIR = PROJECT_DIR / "plots_snr_analysis"


def add_awgn(signal, snr_db, rng):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    )
    return signal + noise


def generate_demo_signal(modulation, snr_db, seed, n=256):
    rng = np.random.default_rng(seed)

    if modulation == "BPSK":
        symbols = rng.choice([-1, 1], size=n)
        signal = symbols.astype(complex)

    elif modulation == "QPSK":
        phases = rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=n)
        signal = np.exp(1j * phases)

    elif modulation == "8PSK":
        phases = rng.choice(np.arange(8) * 2 * np.pi / 8, size=n)
        signal = np.exp(1j * phases)

    elif modulation == "QAM16":
        levels = np.array([-3, -1, 1, 3])
        i = rng.choice(levels, size=n)
        q = rng.choice(levels, size=n)
        signal = i + 1j * q
        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    elif modulation == "AM-DSB":
        t = np.arange(n)
        carrier = np.exp(1j * 2 * np.pi * 0.08 * t)
        message = 1 + 0.6 * np.sin(2 * np.pi * 0.015 * t)
        signal = message * carrier

    elif modulation == "WBFM":
        t = np.arange(n)
        message = np.sin(2 * np.pi * 0.015 * t)
        phase = 2 * np.pi * 0.08 * t + 4.0 * np.cumsum(message) / n
        signal = np.exp(1j * phase)

    else:
        signal = rng.normal(size=n) + 1j * rng.normal(size=n)

    noisy_signal = add_awgn(signal, snr_db, rng)
    return noisy_signal


def get_family(modulation):
    if modulation in ["BPSK", "QPSK", "8PSK"]:
        return "PSK"
    if modulation in ["QAM16"]:
        return "QAM"
    if modulation in ["AM-DSB", "WBFM"]:
        return "Analog"
    return "Unknown"


def get_difficulty_note(modulation, snr_db):
    if snr_db <= -10:
        return (
            "This is a very low-SNR condition. The noise can dominate the signal, "
            "so classification is expected to be difficult."
        )
    if modulation in ["AM-DSB", "WBFM"]:
        return (
            "Analog modulation types such as AM-DSB and WBFM can be difficult to separate, "
            "especially when the spectrum and amplitude patterns overlap."
        )
    if modulation == "QAM16":
        return (
            "QAM16 belongs to the QAM family. In real AMC datasets, QAM16 and QAM64 are often "
            "confused because they share similar constellation structure."
        )
    return (
        "This signal is relatively easier to interpret at moderate or high SNR, "
        "but classification can still degrade as noise increases."
    )


st.title("Automatic Modulation Classification Web Demo")

st.markdown("""
This web application presents my final project on **Automatic Modulation Classification (AMC)**.
The goal is to classify wireless signal modulation types from raw I/Q signal samples using
classical machine learning methods rather than deep learning.
""")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Project Overview",
    "Methodology",
    "Results",
    "Interactive Demo"
])

with tab1:
    st.header("Project Overview")

    st.markdown("""
    ### Task

    Automatic Modulation Classification is the task of identifying the modulation type of a
    received wireless signal from its I/Q samples.

    In this project, the input signal is represented as two channels:

    - **I channel**: in-phase component
    - **Q channel**: quadrature component

    The model predicts modulation classes such as PSK, QAM, FSK, PAM, AM, and WBFM-related
    signal types.

    ### Main Challenge

    The classification problem becomes difficult under low-SNR conditions. Some modulation
    pairs also have very similar signal patterns, such as:

    - WBFM vs AM-DSB
    - QAM16 vs QAM64
    - QPSK vs 8PSK
    """)

with tab2:
    st.header("Methodology")

    st.markdown("""
    This project uses a classical machine learning pipeline.

    ### Pipeline

    1. Load RadioML-style I/Q signal data.
    2. Extract interpretable signal features.
    3. Train a baseline model for comparison.
    4. Train a QDA-based likelihood model.
    5. Use XGBoost as the final discriminative classifier.
    6. Apply a family-aware abstention mechanism to reject low-confidence predictions.

    ### Why Classical Machine Learning?

    Instead of using deep learning, this project focuses on engineered features and
    interpretable classification behavior. This makes it easier to analyze which signal
    properties are useful and where the model fails.
    """)

    st.code("""
I/Q Signal
    ↓
Feature Extraction
    ↓
QDA Likelihood Modeling
    ↓
XGBoost Classification
    ↓
Family-aware Abstention
    ↓
Final Prediction or Rejection
""")

with tab3:
    st.header("Experimental Results")

    st.markdown("""
    This section displays the result figures generated during the project.
    These figures summarize classification performance, SNR sensitivity, confidence behavior,
    and major confusion patterns.
    """)

    if PLOT_DIR.exists():
        image_files = sorted(
            list(PLOT_DIR.glob("*.png")) +
            list(PLOT_DIR.glob("*.jpg")) +
            list(PLOT_DIR.glob("*.jpeg"))
        )

        if image_files:
            st.success(f"Found {len(image_files)} result figure(s).")

            for image_path in image_files:
                st.subheader(image_path.stem.replace("_", " ").title())
                st.image(str(image_path), use_container_width=True)
        else:
            st.warning("The plots_snr_analysis folder exists, but no image files were found.")
    else:
        st.error("The plots_snr_analysis folder was not found in the repository.")

with tab4:
    st.header("Interactive I/Q Signal Demo")

    st.markdown("""
    This lightweight demo generates example I/Q signals for several modulation types.
    It is intended to help users visually understand what the model receives as input.
    """)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        modulation = st.selectbox(
            "Select modulation type",
            ["BPSK", "QPSK", "8PSK", "QAM16", "AM-DSB", "WBFM"]
        )

        snr_db = st.slider(
            "Select SNR level (dB)",
            min_value=-20,
            max_value=20,
            value=0,
            step=2
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1
        )

        family = get_family(modulation)

        st.metric("Selected Modulation", modulation)
        st.metric("Signal Family", family)
        st.metric("SNR", f"{snr_db} dB")

    signal = generate_demo_signal(modulation, snr_db, seed)
    i_signal = signal.real
    q_signal = signal.imag
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal)))

    with col_right:
        st.subheader("I/Q Waveform")

        fig, ax = plt.subplots()
        ax.plot(i_signal, label="I channel")
        ax.plot(q_signal, label="Q channel")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Constellation View")

        fig, ax = plt.subplots()
        ax.scatter(i_signal, q_signal, s=12, alpha=0.7)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True)
        ax.axis("equal")
        st.pyplot(fig)

    with col_b:
        st.subheader("Frequency Spectrum")

        fig, ax = plt.subplots()
        ax.plot(spectrum)
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Magnitude")
        st.pyplot(fig)

    st.subheader("Interpretation")

    st.info(get_difficulty_note(modulation, snr_db))

    st.markdown("""
    In the full AMC pipeline, these I/Q samples are transformed into engineered features such as
    spectral features, statistical features, and higher-order signal descriptors. The trained
    classifier then predicts the modulation type, while the abstention mechanism can reject
    uncertain samples.
    """)

st.divider()

st.markdown("""
### Submission Information

This Streamlit application is part of the final project deliverables. It provides a web-based
summary of the AMC project, including the problem definition, methodology, experimental results,
and an interactive I/Q signal demonstration.
""")
