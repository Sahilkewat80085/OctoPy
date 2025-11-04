
                       OCTOPY


A Modular Python Library for Machine Learning Automation
--------------------------------------------------------------
Author : Sahil Kewat
Version: 1.0.0
License: MIT
Language: Python 3.12+


1. OVERVIEW

Octopy is a modular machine learning support library that automates 
data preprocessing, feature selection, model evaluation, and report 
generation. It is designed to simplify the ML workflow by providing 
a collection of plug-and-play Python modules.

The library is built for developers, data scientists, and researchers 
who want a fast, reproducible, and well-structured way to prepare, 
train, and analyze ML models.


2. MODULES INCLUDED

Octopy contains the following main Python modules:

1. pipeline.py     - Handles the creation and execution of ML pipelines.
2. prep.py         - Cleans and preprocesses raw datasets.
3. selector.py     - Performs feature selection and ranking.
4. smart_eda.py    - Generates visual and statistical exploratory data analysis.
5. report.py       - Loads a trained model, evaluates it, and creates JSON reports.


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
    Evaluates trained ML models and generates automated reports.

Key Functions:
    • load_model(model_path)
        - Loads a trained model from .pkl, .sav, or joblib files.
    • load_test_data(x_path, y_path)
        - Loads X_test and y_test from CSV files.
    • evaluate_model(model, X_test, y_test)
        - Computes MAE, MSE, RMSE, and R² metrics.
    • extract_hyperparameters(model)
        - Extracts key hyperparameters from scikit-learn compatible models.
    • generate_report(model_path, x_test_path=None, y_test_path=None)
        - Generates a structured JSON report containing model name,
          hyperparameters, and evaluation metrics.


4. INSTALLATION


1. Clone or download the repository:
       got clone https://github.com/Sahilkewat80085/OctoPy.git
       cd Octopy

2. Install dependencies and package:
       pip install -e .

