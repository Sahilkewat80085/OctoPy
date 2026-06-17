import pandas as pd
from OctoPy.prep import Preprocessor
from OctoPy.comparison import ModelComparer

print("="*60)
print("RUNNING OCTOPY MODEL COMPARISON SYSTEM TESTS")
print("="*60)

# 1. Load Dataset
print("[TEST] Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")
print(f"[TEST] Original Dataset shape: {df.shape}")

# 2. Preprocess Dataset
print("[TEST] Encoding target variable 'species' using Preprocessor...")
preprocessor = Preprocessor(df)
preprocessor.encode_categorical(columns=["species"], method="label")
processed_df = preprocessor.get_processed_data()
print(f"[TEST] Processed Target values:\n{processed_df['species'].value_counts()}")

# 3. Test Specific Models Comparison
print("\n[TEST 1] Benchmarking specific string shorthand models...")
comparer = ModelComparer(processed_df, target="species", problem_type="classification")

# Compare specific classifiers
leaderboard, trained_models = comparer.compare(
    models=["logistic", "randomforest", "histgb"],
    random_state=42
)

# Print leaderboard table in terminal
comparer.print_leaderboard()

# Export HTML report
html_out = "comparison_report.html"
comparer.generate_html_report(html_out)
print(f"[TEST 1] Standalone HTML dashboard successfully exported to {html_out}!")

# 4. Test Auto-suggested Selector Fallback
print("\n[TEST 2] Benchmarking auto-suggested models from selector...")
selector_comparer = ModelComparer(processed_df, target="species", problem_type="classification")

# Pass models=None to trigger automated suggestions based on dataset properties
suggested_leaderboard, suggested_models = selector_comparer.compare(
    models=None,
    random_state=42
)

selector_comparer.print_leaderboard()
print("[TEST 2] Auto-suggest test passed successfully!")

print("\n" + "="*60)
print("ALL COMPARISON BENCHMARK TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
