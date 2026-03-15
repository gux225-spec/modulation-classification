# Feature-Based Automatic Modulation Classification Using Classical Machine Learning

## Summary
This project focuses on automatic modulation classification using the RML2016.10b dataset. Instead of relying on deep learning, the project adopts a classical machine learning pipeline built on handcrafted feature engineering and stacked modeling. Raw IQ radio signal samples are transformed into interpretable feature vectors, then used to train a two-stage classification system. In addition to overall recognition performance, the project evaluates behavior across different signal-to-noise ratio (SNR) levels, analyzes confidence and abstention patterns, and investigates difficult modulation pairs through targeted error analysis and ablation experiments. The goal is not only to achieve strong classification performance, but also to better understand which signal features remain informative under noisy conditions.

## Methodology
The project follows a structured pipeline from raw signal data to final evaluation.

First, the raw RML2016.10b dataset is loaded and inspected to verify its modulation classes, SNR values, and sample structure. A smaller subset is also generated for quick testing and debugging.

Second, handcrafted features are extracted from each IQ sample. These features are implemented in a modular feature extraction module and include statistical, spectral, and modulation-specific descriptors. The feature design emphasizes physically meaningful signal characteristics and allows optional feature blocks to be turned on or off for later ablation studies.

Third, the complete feature dataset is built by traversing every modulation-type and SNR combination. The data is then split using a joint stratification strategy based on modulation label and SNR, producing training, validation, and test sets in a 70% / 15% / 15% ratio. This ensures balanced evaluation across both class type and noise condition.

Fourth, a stacked classical machine learning pipeline is trained. A base Quadratic Discriminant Analysis (QDA) model is trained first, and its outputs are used to construct meta-features. These meta-features are then passed to an XGBoost meta-model for final prediction. StandardScaler and LabelEncoder are trained and saved as part of the pipeline. Despite the historical script name `04_train_gated_experts.py`, the current implementation does not use a rule-based gate or expert-routing module.

Finally, the project performs multiple evaluation and analysis steps. These include SNR-based accuracy analysis, confidence-margin analysis, abstention with family-level thresholds, confusion-matrix analysis, detailed per-SNR error inspection, and targeted ablation experiments to validate the usefulness of newly designed feature groups.

## Project Structure
- `00_inspect_dataset.py`  
  Loads the original `RML2016.10b.dat` dataset and inspects its structure, including modulation classes, SNR levels, and sample dimensions.

- `01_prepare_data.py`  
  Prepares the initial dataset and creates a smaller subset file (`rml2016_10b_mini.pkl`) for quick testing.

- `03_build_feature_dataset.py`  
  Builds the full engineered-feature dataset. It extracts features from raw IQ samples, collects labels and SNR metadata, and performs the joint stratified split into training, validation, and test sets.

- `feature_extraction.py`  
  Core feature engineering module. It converts raw IQ samples into numerically stable `float32` feature vectors and supports modular feature blocks for targeted experiments.

- `04_train_gated_experts.py`  
  Trains the stacked classical machine learning system. It fits and saves the scaler, label encoder, base QDA model, and XGBoost meta-model. The filename is historical; the current training pipeline is a stacked QDA + XGBoost model rather than a rule-gated expert system.

- `rule_gate.py`  
  Legacy experimental utility for rule-based modulation grouping and SNR soft gating. It is not imported by the current training, evaluation, or analysis pipeline and is retained only as historical reference.

- `evaluation_utils.py`  
  Shared evaluation utilities. It provides functions for confidence-margin computation, abstention logic, threshold learning, and plotting.

- `05_eval_by_snr.py`  
  Evaluates overall model performance as a function of SNR and generates performance curves.

- `06_analyze_confidence.py`  
  Analyzes confidence statistics and accepted-sample behavior under the abstention framework.

- `07_analyze_by_snr.py`  
  Performs more detailed SNR-stratified performance and confusion analysis.

- `08_targeted_ablation.py`  
  Runs ablation experiments to test the contribution of specific newly designed feature groups.

## Data Source
This project uses the RML2016.10b dataset, a widely used benchmark dataset for automatic modulation classification research. The dataset contains complex IQ radio signal samples spanning multiple digital and analog modulation types under a range of signal-to-noise ratio (SNR) conditions. In this project, the dataset serves as the raw source for feature extraction, model training, and evaluation.

## References
1. RML2016.10a / RadioML dataset materials and related open-source benchmark resources.

2. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, 2011.

3. Chen, T. and Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
