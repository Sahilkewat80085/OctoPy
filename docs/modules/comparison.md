# Multi-Model Benchmark Comparison (`comparison`)

The `comparison` module contains the `ModelComparer` class, designed to automate benchmarking sweeps across multiple model architectures on identical data partitions. It monitors model accuracy and execution speeds (training and prediction runtimes), rendering reports directly in the console or as standalone HTML dashboards.

---

## The `ModelComparer` Class

Initialize the comparer with a pandas DataFrame, target variable, and optional task type:

```python
from OctoPy.comparison import ModelComparer

comparer = ModelComparer(df, target="target_column")
```

---

## Key Features

### 1. Mathematical Split Integrity
Benchmarking models on different training/testing splits introduces random variance that invalidates comparison. `ModelComparer` prevents this by:
1.  Splitting features and targets **once** when `.compare()` is called.
2.  Supplying the exact same training split to every model's `.fit()` method.
3.  Evaluating predictions on the exact same test split.

---

### 2. Auto-Suggested Candidates Fallback
If you pass `models=None` to the compare loop, the class queries the `ModelSelector` internally to automatically suggest, instantiate, and benchmark optimal candidates based on your dataset profile.

```python
# Benchmarks auto-suggested models
leaderboard, models_dict = comparer.compare(models=None, random_state=42)
```

Alternatively, pass a list of shorthand strings or custom instantiated estimator objects:

```python
from sklearn.ensemble import ExtraTreesClassifier

# Mix string shorthands with custom scikit-learn estimator instances
leaderboard, models_dict = comparer.compare(
    models=["logistic", "randomforest", ExtraTreesClassifier(n_estimators=100)],
    random_state=42
)
```

---

### 3. Metric Sorting (Ranking)
The leaderboard is ranked using a user-specified metric. 
*   **Classification Default**: `"Accuracy"` (supports sorting by `Accuracy`, `F1_Score`, `Precision`, `Recall`).
*   **Regression Default**: `"R2_Score"` (supports sorting by `R2_Score`, `MAE`, `MSE`, `RMSE`).
*   **Directional Sorting**: Standard metrics are sorted in descending order (higher scores are ranked better). Speed metrics (`Train Time (s)`, `Pred Time (s)`) and error metrics (`MAE`, `MSE`, `RMSE`) are automatically sorted in ascending order (lower scores are ranked better).

---

## Visualizing Leaderboards

### Terminal ASCII Leaderboard
Call `print_leaderboard()` to output a high-contrast tabular report in the console:

```python
comparer.print_leaderboard()
```

Output:
```text
================================================================================
OCTOPY MODEL BENCHMARK LEADERBOARD (Ranked by Accuracy)
================================================================================
Rank         | Model        | Train Time (s | Pred Time (s) | Accuracy     | F1_Score     | Precision    | Recall      
---------------------------------------------------------------------------------------------------------------------
1            | histgb       | 0.0543        | 0.0012        | 1.0          | 1.0          | 1.0          | 1.0         
2            | randomforest | 0.0468        | 0.0041        | 1.0          | 1.0          | 1.0          | 1.0         
3            | logistic     | 0.0039        | 0.0003        | 1.0          | 1.0          | 1.0          | 1.0         
================================================================================
```

---

### HTML Comparison Dashboard
Generate a standalone HTML file containing interactive leaderboard tables and SVG bar chart comparisons for model performance and training speeds:

```python
# Saves the report to comparison_report.html
comparer.generate_html_report("comparison_report.html")
```

The compiled page uses a responsive CSS layout styled after our clean engineering design system, making it easy to share with project stakeholders.
