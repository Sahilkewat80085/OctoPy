# Model Evaluation Reports (`report`)

The `report` module consists of standalone functions designed to load pre-trained model files (`.pkl`, `.sav`, or `.joblib`), evaluate them on separate testing datasets (loaded from CSVs), and compile the results into static JSON files or standalone HTML dashboards.

---

## Key Standalone Functions

### 1. Model Loading
Loads pickled or serialized estimator files safely. Supports standard Python pickle/sav structures and joblib-compressed binaries:

```python
from OctoPy.report import load_model

# Returns a raw trained scikit-learn/XGBoost estimator object
model = load_model("path/to/rf_model.pkl")
```

---

### 2. Test Data Loading
Parses testing feature sets and targets from separate CSV files. It automatically flattens single-column targets to prevent common scikit-learn dimension warnings:

```python
from OctoPy.report import load_test_data

# Returns X_test (DataFrame) and y_test (Series)
X_test, y_test = load_test_data("x_test.csv", "y_test.csv")
```

---

### 3. Hyperparameter Extraction
Extracts model parameter settings dynamically. It queries the estimator's `get_params` API and filters for critical model configurations (e.g., `n_estimators`, `max_depth`, `learning_rate`):

```python
from OctoPy.report import extract_hyperparameters

# Returns a dictionary of parameter names and values
hyperparams = extract_hyperparameters(model)
```

---

### 4. Interactive Report Generation
The high-level entrypoint is the `generate_report` function. It evaluates the model, computes metrics, generates plots, and exports files:

```python
from OctoPy.report import generate_report

# Runs full evaluation and compiles reports
generate_report(
    model_path="rf_model.pkl",
    x_test_path="x_test.csv",
    y_test_path="y_test.csv",
    format="both" # 'json', 'html', or 'both'
)
```

#### Selection Prompts
If the `format` parameter is set to `None` (default), the function checks the running terminal context:
*   If in an **interactive terminal**, it prompts you to input your choice (`[1] JSON`, `[2] HTML`, or `[3] Both`).
*   If in a **non-interactive/headless script**, it defaults to compiling `"both"` formats.

---

## Output Formats

### 1. Structured JSON (`model_report.json`)
Saves a clean, serialized snapshot of the run, suitable for machine-to-machine integrations or pipeline logging. For example:
```json
{
    "Model Name": "rf_model.pkl",
    "Task Type": "Classification",
    "Hyperparameters": {
        "n_estimators": 50,
        "max_depth": null,
        "criterion": "gini"
    },
    "Evaluation Metrics": {
        "Accuracy": 0.9667,
        "F1_Score": 0.9667,
        "Precision": 0.9700,
        "Recall": 0.9667
    }
}
```
*(Note: NumPy-specific arrays like confusion matrices are excluded from the JSON output to maintain standard serialization compatibility).*

### 2. Dashboard HTML (`model_report.html`)
Compiles a self-contained, responsive dashboard with a minimalist light layout.
*   **Vector Charts**: Embedded natively in the HTML as base64-encoded SVG strings for sharp displays on all devices.
*   **Contained Visuals**: Includes a global feature importance impact chart (generated via `ModelExplainer`) alongside task-specific evaluation charts (e.g., confusion matrix heatmap for classifiers, actual-vs-predicted scatter plots for regressors).
