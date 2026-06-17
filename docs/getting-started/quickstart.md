# Quick Start Guide

This quick start guide will walk you through a complete, end-to-end machine learning workflow using **OctoPy** with a sample classification dataset.

We will cover:
1. Loading and preprocessing data.
2. Generating recommendations with `ModelSelector`.
3. Benchmarking multiple estimators with `ModelComparer`.
4. Generating global and local explanations with `ModelExplainer`.

---

## End-to-End Workflow Script

Create a new file named `quickstart.py` in the root of your project directory and add the following code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from OctoPy.prep import Preprocessor
from OctoPy.selector import ModelSelector
from OctoPy.comparison import ModelComparer
from OctoPy.explain import ModelExplainer

# ==========================================
# 1. Load the Dataset
# ==========================================
print("Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")
print(f"Original shape: {df.shape}\n")

# ==========================================
# 2. Preprocess Data
# ==========================================
print("Preprocessing dataset...")
# Initialize Preprocessor
preprocessor = Preprocessor(df)

# Label encode target categorical column 'species'
preprocessor.encode_categorical(columns=["species"], method="label")

# Get the processed DataFrame
processed_df = preprocessor.get_processed_data()

# Split features and target
X = processed_df.drop(columns=["species"])
y = processed_df["species"]

# Create train-test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Preprocessing complete.\n")

# ==========================================
# 3. Auto-Suggest Models
# ==========================================
print("Generating model recommendations based on dataset metadata...")
selector = ModelSelector(processed_df, target="species")
selector.print_summary()
print()

# ==========================================
# 4. Compare Models
# ==========================================
print("Benchmarking suggested models...")
comparer = ModelComparer(processed_df, target="species", problem_type="classification")

# Run benchmark across logistic regression, random forest, and gradient boosting
leaderboard, trained_models = comparer.compare(
    models=["logistic", "randomforest", "histgb"],
    random_state=42
)

# Print comparison leaderboard in terminal
comparer.print_leaderboard()

# Save comparison results to an HTML dashboard
comparer.generate_html_report("comparison_report.html")
print("HTML comparison report saved to 'comparison_report.html'.\n")

# ==========================================
# 5. Explaining Model Behavior
# ==========================================
print("Configuring Explainer for the Random Forest model...")
# Grab trained Random Forest model from comparison run
rf_model = trained_models["randomforest"]

# Initialize Explainer
explainer = ModelExplainer(rf_model, X_train, y_train)

# Calculate global feature importance (and save plot)
print("Computing global feature importance...")
df_global, fig_global = explainer.explain_global(
    X_test, y_test, save_path="rf_global_importance.png"
)
print("Global Feature Impact Leaderboard:")
print(df_global)
print()

# Explain a single prediction (Local Explainability)
sample = X_test.iloc[0]
print(f"Explaining individual prediction for sample:\n{sample.to_dict()}")
df_local = explainer.explain_prediction(sample)
```

---

## Running the Code

Execute the script from your terminal:

```bash
python quickstart.py
```

### Expected Output

1. **Model Suggestions Summary**: An analysis of the data size, shape, and balance, followed by recommendations of candidate models.
2. **ASCII Leaderboard**: A clean, formatted table in the console detailing training times, prediction times, and classification scores (Accuracy, F1, Precision, Recall).
3. **Local Explanation Bar Chart**: An ASCII-drawn bar chart in the console illustrating exactly how each feature value pushed the prediction probability away from the training baseline mean.
4. **Visual Artifacts**:
    * `comparison_report.html`: An interactive HTML dashboard showing rankings and speed benchmarks.
    * `rf_global_importance.png`: A horizontal bar chart visual of feature importances.
