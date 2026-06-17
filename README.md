
                       OCTOPY


A Modular Python Library for Machine Learning Automation
--------------------------------------------------------------
Author : Sahil Kewat
Version: 1.0.0
License: MIT
Language: Python 3.12+


1. OVERVIEW

OctoPy is a modular machine learning support library that automates 
data preprocessing, feature selection, model evaluation, and report 
generation. It is designed to simplify the ML workflow by providing 
a collection of plug-and-play Python modules.

The library is built for developers, data scientists, and researchers 
who want a fast, reproducible, and well-structured way to prepare, 
train, and analyze ML models.


2. MODULES INCLUDED

OctoPy contains the following main Python modules:

1. pipeline.py     - Handles the creation and execution of ML pipelines.
2. prep.py         - Cleans and preprocesses raw datasets.
3. selector.py     - Performs feature selection and ranking.
4. smart_eda.py    - Generates visual and statistical exploratory data analysis.
5. report.py       - Evaluates models, generating structured JSON and premium HTML dashboards.
6. comparison.py   - Benchmarks, profiles, and ranks multiple ML models side-by-side.
7. explain.py      - Dynamic explainability containing global SHAP/permutation and local LOFO prediction tracing.


3. MODULE DETAILS AND FUNCTIONS


--------------------------------------------------------------
A. pipeline.py
--------------------------------------------------------------
Purpose:
    Automates machine learning pipeline creation from preprocessing 
    to model training and saving.

Key Functions:
    • build_pipeline(model, preprocess_steps)
        - Combines preprocessing and model into a single pipeline.
    • train_pipeline(pipeline, X_train, y_train)
        - Fits the pipeline to training data.
    • save_pipeline(pipeline, filename)
        - Saves the pipeline to disk using pickle/joblib.
    • load_pipeline(filename)
        - Loads an existing pipeline for inference or retraining.

--------------------------------------------------------------
B. prep.py
--------------------------------------------------------------
Purpose:
    Cleans, encodes, and scales data for model training.

Key Functions:
    • handle_missing_values(df)
        - Fills or removes missing values automatically.
    • encode_categorical(df)
        - Converts categorical variables into numeric form.
    • scale_features(df)
        - Applies standard or min-max scaling to numeric features.
    • preprocess_data(df)
        - Combines all preprocessing operations into one function.

--------------------------------------------------------------
C. selector.py
--------------------------------------------------------------
Purpose:
    Selects the most important features for model training.

Key Functions:
    • select_k_best_features(X, y, k)
        - Selects top k features based on statistical tests.
    • feature_importance(model, X, y)
        - Displays or returns feature importance scores.
    • recursive_feature_elimination(model, X, y)
        - Uses RFE to iteratively eliminate less important features.

--------------------------------------------------------------
D. smart_eda.py
--------------------------------------------------------------
Purpose:
    Automates exploratory data analysis (EDA) and visualization.

Key Functions:
    • describe_data(df)
        - Provides summary statistics of the dataset.
    • plot_distributions(df)
        - Plots histograms and distribution graphs for numeric features.
    • correlation_heatmap(df)
        - Displays correlation between numeric variables.
    • detect_outliers(df)
        - Identifies outliers using z-score or IQR method.

--------------------------------------------------------------
E. report.py
--------------------------------------------------------------
Purpose:
    Evaluates trained ML models, generating structured JSON files and 
    sleek, zero-dependency HTML dashboard reports with inline vector SVGs.

Key Functions:
    • load_model(model_path)
        - Loads a trained model from .pkl, .sav, or joblib files.
    • load_test_data(x_path, y_path)
        - Loads X_test and y_test from CSV files.
    • evaluate_model(model, X_test, y_test)
        - Computes accuracy, F1, precision, recall (classification) or MAE, RMSE, R² (regression).
    • generate_report(model_path, x_test_path=None, y_test_path=None, format=None)
        - Generates structured JSON reports, HTML reports, or both, featuring 
          embedded global explainability summary charts and evaluation metrics.

--------------------------------------------------------------
F. comparison.py
--------------------------------------------------------------
Purpose:
    Compares, benchmarks, and ranks multiple ML models on mathematically identical 
    data partitions. Profiles both metric performance and fitting/prediction runtimes.

Key Functions:
    • compare(models=None, rank_by=None, test_size=0.2, random_state=42)
        - Trains multiple string shorthands or custom estimators. Interlinks 
          with ModelSelector for optimal candidates fallback if list is None.
    • print_leaderboard()
        - Prints a beautiful, high-contrast plain-text ASCII table to the terminal.
    • generate_html_report(output_path="comparison_report.html")
        - Compiles model runtimes and metric scores into a sleek standalone HTML report.

--------------------------------------------------------------
G. explain.py
--------------------------------------------------------------
Purpose:
    Demystifies model predictions, explaining global feature importance and 
    tracing individual decisions. Provides high-end visual summaries with 
    smart dependency fallbacks.

Key Functions:
    • explain_global(X_val, y_val=None, save_path=None)
        - Computes global explanations. Leverages game-theoretic SHAP values 
          if installed, else falls back to tree importances, model coefficients, 
          or calculates model-agnostic Permutation Importance on validation sets.
    • explain_prediction(sample_row)
        - Traces a single prediction row using Leave-One-Feature-Out (LOFO) deltas.
          Outputs an interactive contribution table with high-contrast console ASCII bars.


4. INSTALLATION


1. Clone or download the repository:
       git clone https://github.com/Sahilkewat80085/OctoPy.git
       cd OctoPy

2. Install dependencies and package:
       pip install -e .


5. PLANNED UPGRADES

• Terminal CLI Support     - Direct command-line utility to train, compare, and report.
• Smart Dataset Analyzer   - Auto-diagnostics for data skewness, leaks, and quality.
