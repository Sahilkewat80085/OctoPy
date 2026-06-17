# Model Explainability (`explain`)

The `explain` module provides post-hoc model interpretability using the `ModelExplainer` class. It offers global feature impact analysis (relying on SHAP or standard fallback heuristics) and local single-instance prediction tracing (using Leave-One-Feature-Out counterfactual simulations).

---

## The `ModelExplainer` Class

Initialize the explainer with a pre-trained estimator, the training feature set, and the target arrays:

```python
from OctoPy.explain import ModelExplainer

explainer = ModelExplainer(model, X_train, y_train)
```

---

## 1. Global Feature Impact (`explain_global`)

Global interpretability explains which features are the most important across the entire dataset.

```python
df_impact, fig = explainer.explain_global(X_test, y_test, save_path="feature_impact.png")
```

### Fallback Heuristic Chain

If the `shap` package is installed and compatible with the model type, `ModelExplainer` generates a game-theoretic SHAP summary plot. If not, it executes a prioritized fallback sequence to ensure robustness:

1.  **SHAP summary plot**: Calculates average absolute SHAP values.
2.  **Gini / Tree Importance**: Fallback for tree-based ensembles (e.g. Random Forests) exposing `feature_importances_`.
3.  **Linear Coefficients**: Fallback for linear models (e.g. Logistic Regression) exposing `coef_`. (Computes absolute mean across classes for multi-class classifiers).
4.  **Permutation Importance**: Model-agnostic fallback for non-linear, non-tree classifiers (e.g. SVM with RBF kernels). Measures performance degradation on validation sets when columns are randomly shuffled.
5.  **Uniform Baseline**: Baseline fallback if all other diagnostics fail (assigns equal weights).

---

## 2. Local Prediction Tracing (`explain_prediction`)

Local interpretability explains why the model made a specific decision for a single data row. 

```python
# Trace prediction on the first row of test set
sample = X_test.iloc[0]
df_local = explainer.explain_prediction(sample)
```

### Leave-One-Feature-Out (LOFO) Counterfactuals
To calculate how a specific feature value affects a prediction:
1.  The explainer records the model's actual prediction probability (for classification) or output value (for regression) on the sample row.
2.  For each feature, the explainer temporarily replaces the feature's value with its corresponding training mean baseline value and re-computes the prediction.
3.  The difference (delta) between the actual prediction and the baseline-replaced prediction is recorded as that feature's **Contribution**.

### Console ASCII Visualization
Calling `explain_prediction()` prints a high-contrast console bar chart showing feature influences:

```text
================================================================================
OCTOPY INDIVIDUAL PREDICTION EXPLAINER (CLASSIFICATION (Class 2 prob: 98.24%))
================================================================================
Feature            | Value        | Contribution | Weight Contribution Graph
--------------------------------------------------------------------------------
petal width (cm)   | 2.5          | +0.428       | [----------==========          ]
petal length (cm)  | 6.0          | +0.312       | [----------=======             ]
sepal length (cm)  | 6.3          | -0.052       | [        ==----------------]
sepal width (cm)   | 3.3          | +0.015       | [----------                    ]
================================================================================
```

*   **Positive values (`+`)**: Push the model's prediction closer to the target class (bars render on the right).
*   **Negative values (`-`)**: Drag the prediction away from the target class (bars render on the left).
