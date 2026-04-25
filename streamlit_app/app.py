from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib


st.set_page_config(
    page_title="AMC Project Demo",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PLOT_DIR = PROJECT_DIR / "plots_snr_analysis"
MODEL_DIR = PROJECT_DIR / "models"
DEMO_PATH = PROJECT_DIR / "demo_data" / "demo_samples.npz"

# Allow Streamlit app.py to import project-level files
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import feature_extraction as fe_base
import feature_enhancer as fe_enhancer


def extract_one_feature(iq_sample):
    """
    Build the same 80-dimensional feature vector used during training.

    feature_extraction.py -> 45 features
    feature_enhancer.py   -> 35 features
    combined              -> 80 features
    """

    base_feat = fe_base.extract_features(iq_sample)
    enhancer_feat = fe_enhancer.extract_disambiguation_features(iq_sample)

    base_feat = np.asarray(base_feat, dtype=float).reshape(-1)
    enhancer_feat = np.asarray(enhancer_feat, dtype=float).reshape(-1)

    full_feat = np.concatenate([base_feat, enhancer_feat], axis=0)

    if full_feat.shape[0] != 80:
        raise RuntimeError(
            f"Feature dimension error: expected 80 features, but got {full_feat.shape[0]}."
        )

    return full_feat.reshape(1, -1)


def qda_features(qda_model, X_scaled):
    """
    Create QDA posterior features:
    log posterior probabilities + margin.
    """

    if hasattr(qda_model, "predict_log_proba"):
        log_post = qda_model.predict_log_proba(X_scaled)
    else:
        proba = qda_model.predict_proba(X_scaled)
        log_post = np.log(proba + 1e-12)

    sorted_log = np.sort(log_post, axis=1)
    margin = sorted_log[:, -1] - sorted_log[:, -2]

    return np.hstack([log_post, margin.reshape(-1, 1)])


def create_meta_features(qda_model, X_scaled):
    """
    Match the training/evaluation pipeline:
    X_meta = [X_scaled, log_post, margin]
    """

    X_qda = qda_features(qda_model, X_scaled)
    return np.hstack([X_scaled, X_qda])


def get_xgb_expected_features(xgb_model):
    if hasattr(xgb_model, "n_features_in_"):
        return xgb_model.n_features_in_

    if hasattr(xgb_model, "get_booster"):
        return xgb_model.get_booster().num_features()

    return None


@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "scaler_perkey2000.joblib")
    qda_model = joblib.load(MODEL_DIR / "gen_model_qda_perkey2000.joblib")
    xgb_model = joblib.load(MODEL_DIR / "xgb_model_perkey2000.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder_perkey2000.joblib")

    return scaler, qda_model, xgb_model, label_encoder


@st.cache_data
def load_demo_data():
    demo = np.load(DEMO_PATH, allow_pickle=True)

    X_demo = demo["X"]
    y_demo = demo["y"]
    snr_demo = demo["snr"]

    return X_demo, y_demo, snr_demo


def predict_one_sample(iq_sample):
    scaler, qda_model, xgb_model, label_encoder = load_models()

    raw_feat = extract_one_feature(iq_sample)
    X_scaled = scaler.transform(raw_feat)

    X_qda = qda_features(qda_model, X_scaled)
    X_full = create_meta_features(qda_model, X_scaled)

    expected = get_xgb_expected_features(xgb_model)

    if expected == X_scaled.shape[1]:
        X_for_xgb = X_scaled
        feature_mode = "Scaled 80 engineered features"

    elif expected == X_qda.shape[1]:
        X_for_xgb = X_qda
        feature_mode = "QDA posterior features"

    elif expected == X_full.shape[1]:
        X_for_xgb = X_full
        feature_mode = "Scaled 80 engineered features + QDA meta features"

    else:
        raise RuntimeError(
            "Feature dimension mismatch. "
            f"XGBoost expects {expected}, but available shapes are "
            f"{X_scaled.shape[1]}, {X_qda.shape[1]}, {X_full.shape[1]}."
        )

    proba = xgb_model.predict_proba(X_for_xgb)[0]

    pred_index = int(np.argmax(proba))
    pred_encoded = xgb_model.classes_[pred_index]

    pred_label = label_encoder.inverse_transform([int(pred_encoded)])[0]
    confidence = float(np.max(proba))

    class_names = label_encoder.inverse_transform(
        np.asarray(xgb_model.classes_, dtype=int)
    )

    top_idx = np.argsort(proba)[::-1][:5]

    top_df = pd.DataFrame({
        "Modulation": [class_names[i] for i in top_idx],
        "Probability": [float(proba[i]) for i in top_idx]
    })

    qda_margin = float(X_qda[0, -1])

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "top_df": top_df,
        "feature_mode": feature_mode,
        "raw_feature_shape": raw_feat.shape,
        "scaled_feature_shape": X_scaled.shape,
        "qda_feature_shape": X_qda.shape,
        "full_feature_shape": X_full.shape,
        "qda_margin": qda_margin
    }


def plot_iq_waveform(iq_sample):
    i_signal = iq_sample[0]
    q_signal = iq_sample[1]

    fig, ax = plt.subplots()
    ax.plot(i_signal, label="I channel")
    ax.plot(q_signal, label="Q channel")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    return fig


def plot_constellation(iq_sample):
    i_signal = iq_sample[0]
    q_signal = iq_sample[1]

    fig, ax = plt.subplots()
    ax.scatter(i_signal, q_signal, s=14, alpha=0.7)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.axis("equal")
    return fig


def plot_spectrum(iq_sample):
    complex_signal = iq_sample[0] + 1j * iq_sample[1]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(complex_signal)))

    fig, ax = plt.subplots()
    ax.plot(spectrum)
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Magnitude")
    return fig


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
    "Real Model Demo"
])

with tab1:
    st.header("Project Overview")

    st.markdown("""
    ### Task

    Automatic Modulation Classification is the task of identifying the modulation type of a
    received wireless signal from its I/Q samples.

    In this project, each input signal is represented as two channels:

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
    3. Add targeted disambiguation features.
    4. Standardize the 80-dimensional feature vector.
    5. Use QDA to generate likelihood-based meta features.
    6. Use XGBoost as the final discriminative classifier.
    7. Analyze performance across modulation types and SNR levels.

    ### Why Classical Machine Learning?

    Instead of using deep learning, this project focuses on engineered signal features and
    interpretable classification behavior. This makes it easier to analyze which signal
    properties are useful and where the model fails.
    """)

    st.code("""
I/Q Signal
    ↓
45 baseline signal features
    ↓
35 targeted disambiguation features
    ↓
80-dimensional engineered feature vector
    ↓
StandardScaler
    ↓
QDA log-posterior and margin features
    ↓
XGBoost classifier
    ↓
Predicted modulation type and probability
""")

with tab3:
    st.header("Experimental Results")

    st.markdown("""
    This section displays result figures generated during the project.
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
    st.header("Real Model Inference Demo")

    st.markdown("""
    This page randomly selects one I/Q sample from a small demo subset of the RadioML-style
    dataset and runs the trained classical machine learning pipeline in real time.
    """)

    required_paths = [
        MODEL_DIR / "scaler_perkey2000.joblib",
        MODEL_DIR / "gen_model_qda_perkey2000.joblib",
        MODEL_DIR / "xgb_model_perkey2000.joblib",
        MODEL_DIR / "label_encoder_perkey2000.joblib",
        DEMO_PATH,
    ]

    missing = [str(p.relative_to(PROJECT_DIR)) for p in required_paths if not p.exists()]

    if missing:
        st.error("Some required files are missing from the repository.")
        st.write(missing)
    else:
        X_demo, y_demo, snr_demo = load_demo_data()

        st.write(f"Demo dataset size: **{len(X_demo)} samples**")

        if "selected_idx" not in st.session_state:
            st.session_state.selected_idx = int(np.random.randint(0, len(X_demo)))

        col_btn1, col_btn2 = st.columns([1, 4])

        with col_btn1:
            if st.button("Random sample"):
                st.session_state.selected_idx = int(np.random.randint(0, len(X_demo)))

        idx = st.session_state.selected_idx

        iq_sample = X_demo[idx]
        true_label = y_demo[idx]
        snr = snr_demo[idx]

        result = predict_one_sample(iq_sample)

        st.subheader("Selected Sample")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sample Index", idx)
        col2.metric("True Label", str(true_label))
        col3.metric("SNR", f"{snr} dB")
        col4.metric("I/Q Shape", str(iq_sample.shape))

        st.subheader("Model Prediction")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Predicted Label", str(result["pred_label"]))
        col6.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
        col7.metric("QDA Margin", f"{result['qda_margin']:.3f}")

        if str(result["pred_label"]) == str(true_label):
            col8.success("Correct")
        else:
            col8.error("Wrong")

        st.caption(f"Feature mode used by XGBoost: {result['feature_mode']}")

        with st.expander("Feature shapes"):
            st.write("Raw feature shape:", result["raw_feature_shape"])
            st.write("Scaled feature shape:", result["scaled_feature_shape"])
            st.write("QDA feature shape:", result["qda_feature_shape"])
            st.write("Full meta feature shape:", result["full_feature_shape"])

        st.subheader("Top Predicted Probabilities")

        st.dataframe(result["top_df"], use_container_width=True)
        st.bar_chart(
            result["top_df"].set_index("Modulation")["Probability"]
        )

        st.subheader("Signal Visualization")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### I/Q Waveform")
            st.pyplot(plot_iq_waveform(iq_sample))

        with col_b:
            st.markdown("#### Constellation View")
            st.pyplot(plot_constellation(iq_sample))

        st.markdown("#### Frequency Spectrum")
        st.pyplot(plot_spectrum(iq_sample))

        st.info("""
        The probability shown here comes from the trained XGBoost classifier. The demo uses a
        small exported subset of the original dataset so that the web app remains lightweight
        and stable during deployment.
        """)

st.divider()

st.markdown("""
### Submission Information

This Streamlit application is part of the final project deliverables. It provides a web-based
summary of the AMC project, including the problem definition, methodology, experimental results,
and a real model inference demo.
""")
