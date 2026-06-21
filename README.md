
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


2. MODULES & CLASSES INCLUDED

OctoPy contains the following core modules and classes:

1. prep.py         - Standardized data preparation via the Preprocessor class.
2. selector.py     - Dataset analysis and model recommendations via the ModelSelector class.
3. pipeline.py     - Automated model training workflows via the PipelineBuilder class.
4. smart_eda.py    - Visual and statistical exploratory data analysis via the SmartEDA class.
5. explain.py      - Explainable AI (SHAP, Permutation, LOFO) via the ModelExplainer class.
6. comparison.py   - Runtimes and metrics benchmarking via the ModelComparer class.
7. report.py       - Automated report generation via the generate_report function.


3. API DETAILS AND CLASS USAGE

--------------------------------------------------------------
A. prep.py (Preprocessor)
--------------------------------------------------------------
Purpose:
    Cleans, encodes, and scales data for model training.

Key Methods:
    • Preprocessor(df)
        - Initializes the preprocessor with a copy of the input DataFrame.
    • handle_missing(strategy='mean', columns=None)
        - Handles missing values via 'mean', 'median', 'mode', or 'drop'.
    • encode_categorical(columns=None, method='onehot')
        - Encodes object/categorical columns using 'onehot' or 'label' encoding.
    • scale_features(columns=None, method='standard')
        - Standardizes or min-max scales numeric features.
    • get_processed_data()
        - Returns the transformed pandas DataFrame.

--------------------------------------------------------------
B. selector.py (ModelSelector)
--------------------------------------------------------------
Purpose:
    Analyzes dataset shape, feature count, and class imbalance to suggest optimal ML architectures.

Key Methods:
    • ModelSelector(df, target, problem_type=None)
        - Initializes the recommender. Target problem type is auto-inferred if not specified.
    • suggest_models()
        - Returns a list of suitable un-fitted model instances (scikit-learn and XGBoost).
    • print_summary()
        - Prints dataset statistics and recommended models to the terminal.

--------------------------------------------------------------
C. pipeline.py (PipelineBuilder)
--------------------------------------------------------------
Purpose:
    Maps model shorthand string identifiers to estimator instances and executes training sweeps.

Key Methods:
    • PipelineBuilder(df, target, problem_type=None)
        - Initializes with dataset, labels, and target column.
    • train(model_name='randomforest', test_size=0.2, random_state=42)
        - Splits data, trains the specified model (supports 30+ string shorthand estimators),
          and returns the fitted model instance along with evaluation metrics.

--------------------------------------------------------------
D. smart_eda.py (SmartEDA)
--------------------------------------------------------------
Purpose:
    Automates exploratory data analysis and headless plotting.

Key Methods:
    • SmartEDA(df)
        - Initializes with the target DataFrame.
    • basic_info()
        - Prints shape, data types, duplicates, and summary statistics.
    • value_counts()
        - Displays value counts for categorical variables.
    • distribution_plots(save_path=None)
        - Plots histograms. Saves as file if save_path is provided.
    • boxplots(save_path=None)
        - Generates outlier boxplots. Saves files per feature if save_path is provided.
    • correlation_heatmap(save_path=None)
        - Plots the numeric feature correlation matrix.
    • target_relation(target, save_path=None)
        - Visualizes feature interactions relative to the target column.
    • run_all(target=None)
        - Executes all EDA routines in sequence.

--------------------------------------------------------------
E. explain.py (ModelExplainer)
--------------------------------------------------------------
Purpose:
    Demystifies predictions, explaining global feature impact and local decision contributions.

Key Methods:
    • ModelExplainer(model, X_train, y_train, feature_names=None)
        - Initializes with a trained model and baseline training partitions.
    • explain_global(X_val, y_val=None, save_path=None)
        - Computes global explanations. Leverages game-theoretic SHAP values if installed,
          else falls back to tree importances, model coefficients, or Permutation Importance.
    • explain_prediction(sample_row)
        - Traces a single prediction row using Leave-One-Feature-Out (LOFO) counterfactual weights,
          printing a high-contrast terminal ASCII graph.

--------------------------------------------------------------
F. comparison.py (ModelComparer)
--------------------------------------------------------------
Purpose:
    Compares, benchmarks, and profiles multiple ML models on identical splits.

Key Methods:
    • ModelComparer(df, target, problem_type=None)
        - Initializes benchmark environment.
    • compare(models=None, rank_by=None, test_size=0.2, random_state=42)
        - Trains, scores, and profiles model list. Falls back to ModelSelector if models is None.
    • print_leaderboard()
        - Prints plain-text ASCII leaderboard tables.
    • generate_html_report(output_path="comparison_report.html")
        - Compiles statistics and performance plots into a premium standalone HTML dashboard.

--------------------------------------------------------------
G. report.py
--------------------------------------------------------------
Purpose:
    Evaluates models, generating structured JSON files and interactive HTML dashboards.

Key Functions:
    • load_model(model_path)
        - Safely loads a serialized model file.
    • load_test_data(x_path, y_path)
        - Loads validation CSV files.
    • evaluate_model(model, X_test, y_test, is_clf)
        - Computes task-specific evaluation metrics.
    • generate_report(model_path, x_test_path=None, y_test_path=None, format=None)
        - Orchestrates evaluation, loading, and dashboard compilation (JSON, HTML, or both).


4. INSTALLATION


1. Clone or download the repository:
       git clone https://github.com/Sahilkewat80085/OctoPy.git
       cd OctoPy

2. Install dependencies and package:
       pip install -e .


5. PLANNED UPGRADES

• Terminal CLI Support     - Direct command-line utility to train, compare, and report.
• Smart Dataset Analyzer   - Auto-diagnostics for data skewness, leaks, and quality.
