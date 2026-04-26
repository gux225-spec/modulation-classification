# Automatic Modulation Classification Project

## Overview
This submission contains the core Python scripts for an Automatic Modulation Classification (AMC) project based on classical machine learning and handcrafted signal features.

The project has two main parts:

1. A complete AMC pipeline built on the RML2016.10b dataset.
2. A cross-dataset generalization study that evaluates the trained RML2016.10b model on the CustomMOD-2026.a dataset.

The original pipeline transforms raw I/Q radio samples into engineered feature vectors, trains a stacked QDA + XGBoost classifier, and evaluates performance across signal-to-noise ratio (SNR) conditions. The project is then extended to test whether the trained model generalizes to a new dataset containing partially overlapping modulation types.

## Project Functionality
The included code supports:

- inspecting and summarizing the original RML2016.10b dataset
- preparing a mini subset for quick debugging
- extracting handcrafted statistical, spectral, analog, QAM, IQ-geometry, wavelet, and disambiguation features
- building train / validation / test feature datasets
- training a stacked QDA + XGBoost AMC model
- evaluating model performance by SNR
- analyzing confidence and abstention behavior
- performing detailed SNR-stratified confusion analysis
- preparing overlap-only data from CustomMOD-2026.a
- evaluating cross-dataset generalization on CustomMOD-2026.a

## Included Code Files

### Core RML2016.10b pipeline
- `00_inspect_dataset.py`
- `01_prepare_data.py`
- `02_check_features_on_mini.py`
- `03_build_feature_dataset.py`
- `04_train_gated_experts.py`
- `05_eval_by_snr.py`
- `06_analyze_confidence.py`
- `07_analyze_by_snr.py`
- `08_targeted_ablation.py`
- `run_all.py`

### Feature and evaluation modules
- `feature_extraction.py`
- `feature_enhancer.py`
- `evaluation_utils.py`

### CustomMOD generalization workflow
- `09_inspect_custommod.py`
- `10_prepare_custommod_overlap.py`
- `11_eval_custommod_generalization.py`

## File-by-File Description

### Core RML2016.10b pipeline scripts
- `00_inspect_dataset.py`: Loads the original RML2016.10b `.dat` file and prints its dictionary structure, modulation classes, SNR values, and sample shape.
- `01_prepare_data.py`: Creates a smaller debugging subset from the original dataset and saves it for quick inspection.
- `02_check_features_on_mini.py`: Runs feature extraction on the small subset to verify feature dimensionality and numerical validity.
- `03_build_feature_dataset.py`: Extracts handcrafted features from the full RML2016.10b dataset and builds train / validation / test splits.
- `04_train_gated_experts.py`: Trains the current stacked classical pipeline consisting of StandardScaler, QDA or GaussianNB fallback, and the final XGBoost classifier.
- `05_eval_by_snr.py`: Evaluates the trained model on the RML test set and generates summary evaluation plots.
- `06_analyze_confidence.py`: Analyzes prediction margin and accepted-sample confidence after the abstention mechanism is applied.
- `07_analyze_by_snr.py`: Performs detailed SNR-level analysis, including per-SNR accuracy, confusion behavior, and margin trends.
- `08_targeted_ablation.py`: Runs targeted ablation experiments to test the contribution of selected feature groups.
- `run_all.py`: Provides a convenient runner that executes the main RML2016.10b pipeline scripts in sequence.

### Feature and utility modules
- `feature_extraction.py`: Implements the main handcrafted feature extractor, including baseline, analog, QAM, IQ-geometry, and wavelet features.
- `feature_enhancer.py`: Implements additional disambiguation features for difficult modulation pairs such as QAM, phase, envelope, and instantaneous-frequency cases.
- `evaluation_utils.py`: Contains shared utility functions for confidence margins, abstention thresholds, confusion matrix plotting, and evaluation summaries.

### CustomMOD generalization scripts
- `09_inspect_custommod.py`: Inspects the CustomMOD-2026.a HDF5 dataset and reports its keys, array shapes, class information, and SNR values.
- `10_prepare_custommod_overlap.py`: Reads CustomMOD-2026.a, maps its labels to the model-known label space, discards unseen modulation types, and saves the overlap-only evaluation subset.
- `11_eval_custommod_generalization.py`: Extracts handcrafted features for the overlap subset, runs the trained RML2016.10b model, reports overall and per-class accuracy, and saves the generalization confusion matrix.

## Method Summary

### Original RML2016.10b training pipeline
The main model is trained on the RML2016.10b dataset. Raw I/Q samples are converted into a combined handcrafted feature vector. The resulting features are standardized and passed through a generative model (QDA, with GaussianNB fallback if needed). The log-posterior outputs and their margin are then concatenated with the original scaled features and used as meta-features for a final XGBoost classifier.

### Cross-dataset generalization pipeline
The CustomMOD-2026.a dataset is stored in HDF5 format with shape `(samples, 128, 2)`, one-hot labels, and SNR values. Since the original model was trained on only 10 RML classes, the generalization scripts first filter the CustomMOD dataset to keep only overlapping modulation types. The retained subset is then evaluated by the same trained RML2016.10b model.

The overlap currently used by the scripts is:
- `BPSK`
- `QPSK`
- `8PSK`
- `16QAM -> QAM16`
- `64QAM -> QAM64`

All other CustomMOD classes are discarded during overlap preparation because the original trained model never saw them.

## Required Packages
Install the following packages before running the code:

```bash
pip install numpy scipy scikit-learn xgboost joblib pandas matplotlib seaborn PyWavelets h5py tqdm
```

## Data and Model Placement
This submission follows the requested zip structure and therefore includes a `data/` folder with instructions only.

To preserve the original working pipeline, the scripts keep the same relative paths used during development.

### 1. Original RML dataset
Dataset source:
[RML2016.10b Kaggle page](https://www.kaggle.com/datasets/marwanabudeeb/rml201610b)

Expected file placement:

```text
Project_Root/
|-- RML2016.10b.dat/
|   |-- RML2016.10b.dat
```

Expected runtime path used by the scripts:

```text
./RML2016.10b.dat/RML2016.10b.dat
```

This dataset is used for:
- inspection
- train / validation / test split preparation
- handcrafted feature extraction
- training the QDA + XGBoost AMC pipeline
- evaluation by SNR and confidence

### 2. CustomMOD dataset
Dataset source:
[CustomMOD-2026.a on Zenodo](https://zenodo.org/records/18505222)

Expected file placement:

```text
Project_Root/
|-- RML2016.10b.dat/
|   |-- CustomMOD-2026.a.h5
```

Expected runtime path used by the scripts:

```text
./RML2016.10b.dat/CustomMOD-2026.a.h5
```

This dataset is used only for external generalization testing.

### 3. Trained model files
The evaluation scripts expect the following model artifacts inside `./models/`:

- `scaler_perkey2000.joblib`
- `gen_model_qda_perkey2000.joblib`
- `xgb_model_perkey2000.joblib`
- `label_encoder_perkey2000.joblib`

These model files are included in this submission folder.

## How to Run

### 1. Original RML2016.10b pipeline
Run the full pipeline step by step:

```bash
python 00_inspect_dataset.py
python 01_prepare_data.py
python 03_build_feature_dataset.py
python 04_train_gated_experts.py
python 05_eval_by_snr.py
python 06_analyze_confidence.py
python 07_analyze_by_snr.py
```

Or run the default sequence with:

```bash
python run_all.py
```

`run_all.py` executes the main RML2016.10b workflow only.

### 2. CustomMOD generalization workflow
Inspect the HDF5 dataset:

```bash
python 09_inspect_custommod.py
```

Prepare the overlap-only evaluation set:

```bash
python 10_prepare_custommod_overlap.py
```

Evaluate cross-dataset generalization:

```bash
python 11_eval_custommod_generalization.py
```

## Output Files

### Original RML2016.10b evaluation outputs
Plots are saved to:
- `plots/`
- `plots_snr_analysis/`

### CustomMOD generalization outputs
The generalization evaluation script saves the confusion matrix to:

```text
plots_generalize_snr_analysis/custommod_generalization_confusion_matrix.png
```

## Notes
- The original code files were intentionally preserved rather than rewritten into a new unified `main.py`.
- Some scripts are preprocessing or analysis scripts rather than standalone end-user programs.
- The filename `04_train_gated_experts.py` is historical. The current implementation uses a stacked QDA + XGBoost pipeline, not the earlier rule-gated expert design.
- The CustomMOD evaluation is a true external generalization test, not an in-domain split from the original RML2016.10b dataset.

## Web App
[Project web app / submission page](https://modulation-classification-5hdddbtdybe38gjzhvht3d.streamlit.app/#submission-information)
