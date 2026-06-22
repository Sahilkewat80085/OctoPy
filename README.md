# OctoPy

A Modular Python Library for Machine Learning Automation and Orchestration without hiding the underlying code.

- **Author**: Sahil Kewat
- **Version**: 1.0.0
- **License**: MIT
- **Language**: Python 3.8+

---

## 1. Overview

**OctoPy** is a modular machine learning support library that automates data preprocessing, model selection suggestions, benchmarking, report generation, and explainability. Unlike traditional AutoML libraries that act as opaque "black boxes," OctoPy is transparent: it operates on standard pandas DataFrames and returns raw, standard `scikit-learn`, `xgboost`, and `lightgbm` estimator instances that you can inspect, modify, and serialize.

### Core Philosophy
* **Transparency First**: No hidden custom pipeline structures. Standard models, metrics, and parameters are exposed.
* **Modular Architecture**: Complete separation of concerns. Use the preprocessor, EDA tool, explainer, comparison engine, or report builder in isolation.
* **Developer-Centric**: Fast, text-first, and simple. Built for software engineers and data scientists requiring absolute control over their models.

---

## 2. Installation

Install OctoPy via PyPI:
```bash
pip install octopyx
```

Or install from source in editable mode:
```bash
git clone https://github.com/Sahilkewat80085/OctoPy.git
cd OctoPy
pip install -e .
```

---

## 3. End-to-End Quickstart Example

Create a file named `quickstart.py` and run the following script:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from OctoPy.prep import Preprocessor
from OctoPy.selector import ModelSelector
from OctoPy.comparison import ModelComparer
from OctoPy.explain import ModelExplainer

# 1. Load Dataset
print("Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")

# 2. Preprocess Data
preprocessor = Preprocessor(df)
preprocessor.encode_categorical(columns=["species"], method="label")
processed_df = preprocessor.get_processed_data()

X = processed_df.drop(columns=["species"])
y = processed_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Recommendations
selector = ModelSelector(processed_df, target="species")
selector.print_summary()

# 4. Compare Models
comparer = ModelComparer(processed_df, target="species", problem_type="classification")
leaderboard, trained_models = comparer.compare(models=["logistic", "randomforest", "histgb"], random_state=42)
comparer.print_leaderboard()
comparer.generate_html_report("comparison_report.html")

# 5. Explaining Model Predictions
rf_model = trained_models["randomforest"]
explainer = ModelExplainer(rf_model, X_train, y_train)
df_global, fig_global = explainer.explain_global(X_test, y_test, save_path="rf_global_importance.png")
sample = X_test.iloc[0]
df_local = explainer.explain_prediction(sample)
```

---

## 4. Comprehensive API Reference

### A. Preprocessing (`prep.py`)
Cleans, encodes, and scales features for machine learning pipelines.

#### `Preprocessor` (Class)
*   `__init__(self, df: pd.DataFrame)`
    *   Initializes the preprocessor with a copy of the input DataFrame.
*   `handle_missing(self, strategy='mean', columns=None)`
    *   Fills missing values in specified `columns` (or all columns if `None`).
    *   `strategy`: `'mean'`, `'median'`, `'mode'`, or `'drop'`.
*   `encode_categorical(self, columns=None, method='onehot')`
    *   Encodes categorical columns.
    *   `method`: `'onehot'` (OneHotEncoder) or `'label'` (LabelEncoder).
*   `scale_features(self, columns=None, method='standard')`
    *   Scales numeric columns.
    *   `method`: `'standard'` (StandardScaler) or `'minmax'` (MinMaxScaler).
*   `get_processed_data(self) -> pd.DataFrame`
    *   Returns the fully transformed pandas DataFrame.

---

### B. Automated EDA (`smart_eda.py`)
Automates visual and statistical data diagnostics.

#### `SmartEDA` (Class)
*   `__init__(self, df: pd.DataFrame)`
    *   Initializes the EDA engine.
*   `basic_info(self)`
    *   Prints shape, data types, missing values, duplicates, and summary statistics.
*   `value_counts(self)`
    *   Displays value counts for all categorical columns.
*   `distribution_plots(self, save_path: str = None)`
    *   Plots histograms for all numerical features. Saves to file if `save_path` is specified.
*   `boxplots(self, save_path: str = None)`
    *   Plots boxplots to highlight outliers. Saves separate files dynamically per column if `save_path` is specified.
*   `correlation_heatmap(self, save_path: str = None)`
    *   Plots numerical correlation matrix. Saves to file if `save_path` is specified.
*   `target_relation(self, target: str, save_path: str = None)`
    *   Plots relation boxplots between target and numerical variables. Saves to file if `save_path` is specified.
*   `run_all(self, target: str = None)`
    *   Sequentially runs all EDA functions.

---

### C. Model Recommender (`selector.py`)
Recommends candidate architectures using rule-based metrics on dataset dimensions.

#### `ModelSelector` (Class)
*   `__init__(self, df: pd.DataFrame, target: str, problem_type: str = None)`
    *   `target`: Target column name.
    *   `problem_type`: `'classification'`, `'regression'`, or `None` (auto-inferred).
*   `suggest_models(self) -> List[ModelType]`
    *   Returns list of un-fitted, recommended model objects (Logistic Regression, RandomForest, Gradient Boosting, SVM, etc.).
*   `print_summary(self)`
    *   Prints dataset dimensions, imbalance ratio, and recommended architectures.

---

### D. Pipeline Builder (`pipeline.py`)
Simplifies model instantiation, training loops, and scoring.

#### `PipelineBuilder` (Class)
*   `__init__(self, df: pd.DataFrame, target: str, problem_type: str = None)`
    *   Initializes features `self.X` and labels `self.y`.
*   `train(self, model_name='randomforest', test_size=0.2, random_state=42) -> tuple`
    *   Fits models using standard splits.
    *   `model_name`: Shorthand name for one of the 30+ supported models (e.g. `'logistic'`, `'randomforest'`, `'histgb'`, `'xgboost'`, `'lightgbm'`).
    *   Returns `(model, metrics)`: Trained model instance and evaluation metrics dictionary.

---

### E. Model Explainer (`explain.py`)
Demystifies model behavior using global indicators and prediction trace counterfactuals.

#### `ModelExplainer` (Class)
*   `__init__(self, model, X_train: pd.DataFrame, y_train, feature_names=None)`
    *   Initializes with a trained model and baseline training sets.
*   `explain_global(self, X_val: pd.DataFrame, y_val=None, save_path=None) -> Tuple[pd.DataFrame, plt.Figure]`
    *   Computes feature importances using SHAP values (if installed), tree-importances, coefficients, or Permutation Importance. Saves chart if `save_path` is specified.
*   `explain_prediction(self, sample_row: pd.Series) -> pd.DataFrame`
    *   Traces local prediction using a Leave-One-Feature-Out (LOFO) counterfactual analyzer, rendering a high-contrast terminal ASCII bar graph.

---

### F. Model Comparer (`comparison.py`)
Benchmarks and ranks multiple ML models on mathematically identical splits.

#### `ModelComparer` (Class)
*   `__init__(self, df: pd.DataFrame, target: str, problem_type: str = None)`
    *   Initializes comparer engine.
*   `compare(self, models: list = None, rank_by: str = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, dict]`
    *   Trains and evaluates models. Falls back to `ModelSelector` if `models` list is empty.
    *   Returns `(leaderboard_df, trained_models)`.
*   `print_leaderboard(self)`
    *   Prints ranked plain-text leaderboard to console.
*   `generate_comparison_plots(self) -> dict`
    *   Generates comparative charts (performance, training time) in SVG format.
*   `generate_html_report(self, output_path: str = "comparison_report.html")`
    *   Compiles stats and SVG charts into a premium HTML dashboard.

---

### G. Report Generation (`report.py`)
Compiles structured evaluation metrics and HTML dashboards.

#### Global Helper Functions
*   `load_model(model_path)`
    *   Loads serialized estimator models (`.pkl`, `.sav`, `.joblib`).
*   `load_test_data(x_path, y_path)`
    *   Loads test partitions from CSV files.
*   `extract_hyperparameters(model) -> dict`
    *   Extracts model parameters.
*   `detect_model_type(model) -> bool`
    *   Returns `True` if classifier, `False` if regressor.
*   `evaluate_model(model, X_test, y_test, is_clf: bool) -> dict`
    *   Calculates task-specific metrics (Accuracy, F1, MAE, RMSE, R²).
*   `generate_report(model_path, x_test_path=None, y_test_path=None, format=None)`
    *   Orchestrates evaluation and compiles reports. If `format` is `None`, prompts dynamically for choice.

---

## 5. Optional & Conditional Dependencies

OctoPy degrades gracefully if advanced libraries are missing:
*   **`xgboost` / `lightgbm`**: Used for gradient boosting estimators. If missing, `ModelSelector` and `PipelineBuilder` fall back to scikit-learn models.
*   **`shap`**: Used for game-theoretic global explanations. If missing, `ModelExplainer` falls back to permutation importances and coefficients.

Install optional packages:
```bash
pip install xgboost lightgbm shap
```

---

## 6. Deserialization Safety

> [!WARNING]
> OctoPy utilizes `pickle` and `joblib` for model loading in `report.py`. Never deserialize or load untrusted model files (`.pkl`, `.sav`, `.joblib`), as this can lead to arbitrary code execution.
