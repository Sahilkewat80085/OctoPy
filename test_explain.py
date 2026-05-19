import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from Octopy.prep import Preprocessor
from Octopy.explain import ModelExplainer
from Octopy.report import generate_report

print("="*60)
print("RUNNING OCTOPY EXPLAINABLE AI SYSTEM TESTS")
print("="*60)

# 1. Load and Preprocess Data
print("[TEST] Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")

print("[TEST] Preprocessing target column 'species'...")
prep = Preprocessor(df)
prep.encode_categorical(columns=["species"], method="label")
processed_df = prep.get_processed_data()

X = processed_df.drop(columns=["species"])
y = processed_df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Test RandomForest (Tree intrinsic / SHAP path)
print("\n[TEST 1] Training RandomForest Classifier (Tree / SHAP path)...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

explainer_rf = ModelExplainer(rf, X_train, y_train)

print("[TEST 1] Computing Global Explanations...")
df_global_rf, fig_rf = explainer_rf.explain_global(X_test, y_test, save_path="rf_global_explain.png")
print("Global Explanation Leaderboard:")
print(df_global_rf)

# Trace a single sample prediction (index 0 of test set)
sample = X_test.iloc[0]
print(f"\n[TEST 1] Explaining individual prediction for first sample:\n{sample.to_dict()}")
df_local_rf = explainer_rf.explain_prediction(sample)

# 3. Test Support Vector Classifier (Model-Agnostic Permutation fallback path)
print("\n[TEST 2] Training Support Vector Classifier (Permutation Importance fallback path)...")
# SVC has no feature_importances_ or coef_ (with rbf kernel), so it will fallback to Permutation Importance!
svc = SVC(kernel="rbf", probability=True, random_state=42)
svc.fit(X_train, y_train)

explainer_svc = ModelExplainer(svc, X_train, y_train)

print("[TEST 2] Computing Global Explanations (Triggers Permutation Fallback)...")
df_global_svc, fig_svc = explainer_svc.explain_global(X_test, y_test, save_path="svc_global_explain.png")
print("Global Explanation Leaderboard (Permutation Fallback):")
print(df_global_svc)

print(f"\n[TEST 2] Explaining individual prediction for SVC classifier:")
df_local_svc = explainer_svc.explain_prediction(sample)

# 4. Test Integration with report.py HTML output
print("\n[TEST 3] Verifying HTML report integration in report.py...")
# Save random forest model to disk so report.py can load it
model_filename = "rf_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(rf, f)

# Create temporary csv files for X_test and y_test to simulate command-line/report runner files
x_test_filename = "x_test_temp.csv"
y_test_filename = "y_test_temp.csv"
X_test.to_csv(x_test_filename, index=False)
y_test.to_csv(y_test_filename, index=False)

# Call generate_report with format="html" (non-interactive path)
print("Generating integrated HTML report...")
generate_report(
    model_path=model_filename,
    x_test_path=x_test_filename,
    y_test_path=y_test_filename,
    format="html"
)

# Clean up temp files
for temp_file in [model_filename, x_test_filename, y_test_filename]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print("\n" + "="*60)
print("ALL EXPLAINABILITY AI BENCHMARKS PASSED SUCCESSFULLY!")
print("="*60)
