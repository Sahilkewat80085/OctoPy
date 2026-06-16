# Frequently Asked Questions (FAQ)

This page lists common questions and troubleshooting tips for **Octopy**.

---

### Does Octopy support both Classification and Regression?
Yes. Octopy dynamically adjusts its selection heuristics, preprocessing steps, training loops, evaluation metrics, and visualization formats depending on the task type.
If you do not specify the problem type, Octopy infers it automatically based on your target variable:
*   If the target column has numeric values and $\le 20$ unique entries, it defaults to **classification**.
*   Otherwise, it defaults to **regression**.

---

### What models does Octopy support?
Octopy supports all standard scikit-learn models (logistic regression, random forests, linear regression, gradient boosting, KNNs, SVMs, etc.) and integrates seamlessly with `xgboost` and `lightgbm` estimators.
You can pass model identifiers as string shorthands (e.g. `'randomforest'`) or pass custom-configured scikit-learn estimator instances (e.g., `RandomForestClassifier(n_estimators=200, max_depth=10)`) directly into the comparison sweeps.

---

### How do I load a model that was saved using report generation?
Models are evaluated and stored in their raw python formats. If you want to load a serialized model file (`.pkl` or `.sav`):
```python
import joblib

# Load estimator
model = joblib.load("model_report.pkl")
```
No Octopy imports are required to run predictions with the saved model, making it simple to deploy in production.

---

### How does the explainability fallback logic work?
If the `shap` library is missing or fails on a custom estimator, `ModelExplainer` cascades through alternative diagnostics:
1.  **SHAP summary plot** (Primary)
2.  **Gini tree-based feature importances** (e.g., Random Forest)
3.  **Absolute coefficient weights** (e.g., Logistic Regression)
4.  **Permutation importance** (Model-agnostic fallback)
5.  **Uniform baseline** (Equal weights assignment fallback)

This guarantees that calling `explain_global()` will never crash your pipeline.

---

### Why are my plots blocking script execution?
In standard terminal sessions, `SmartEDA` renders interactive plot windows that block execution until closed.
To bypass this for headless scripts (e.g., cron jobs or Docker containers), configure a non-interactive matplotlib backend at the top of your python script:
```python
import matplotlib
matplotlib.use('Agg')
```

---

### How do I resolve dependency errors?
If you see module import warnings (e.g., for `xgboost`, `lightgbm`, or `shap`), you can install them using pip:
```bash
pip install xgboost lightgbm shap
```
Octopy will run normally using standard scikit-learn fallbacks if you choose not to install these optional dependencies.
