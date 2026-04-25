from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="AMC Project Demo",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PLOT_DIR = PROJECT_DIR / "plots_snr_analysis"

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
    "Demo Explanation"
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
    st.header("Demo Explanation")

    st.markdown("""
    A full interactive demo would allow users to select an I/Q sample and view the model's
    predicted modulation type.

    For this final project web application, the page focuses on presenting the completed
    pipeline, model methodology, and experimental results.

    The trained model is designed to classify modulation types from engineered features
    extracted from I/Q samples. The abstention mechanism can reject uncertain predictions,
    which is especially useful for very noisy low-SNR signals.
    """)

    st.info("The current version is a lightweight project demonstration web app.")

st.divider()

st.markdown("""
### Submission Information

This Streamlit application is part of the final project deliverables. It provides a web-based
summary of the AMC project, including the problem definition, methodology, and experimental results.
""")
