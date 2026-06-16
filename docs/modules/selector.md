# Heuristic Model Selection (`selector`)

The `selector` module contains the `ModelSelector` class, which analyzes dataset properties (dimensions, imbalance, data types) to recommend a curated list of candidate machine learning models. 

---

## The `ModelSelector` Class

The selector evaluates your dataset and suggests appropriate model instances from `scikit-learn` and `xgboost` without performing any training:

```python
from Octopy.selector import ModelSelector

# Target variable and automated problem inference
selector = ModelSelector(df, target="species")
```

---

## Heuristics Under the Hood

The suggestions are generated based on mathematical and statistical characteristics of the input DataFrame:

### 1. Problem Type Inference
If not explicitly provided (via the `problem_type` argument), the library infers it automatically:
*   If the target column is **not numeric**, it is classified as `'classification'`.
*   If the target column is **numeric** and has **20 or fewer unique values**, it is classified as `'classification'`.
*   If the target column is **numeric** and has **more than 20 unique values**, it is classified as `'regression'`.

---

### 2. Classification Suggestions Heuristic
When the task is classified as classification, `ModelSelector` dynamically builds a list of model instances:

*   **Small Datasets (`< 1000` rows)**: Suggests fast, low-variance linear/instance estimators:
    *   `LogisticRegression()`
    *   `KNeighborsClassifier()`
*   **Large Datasets (`>= 1000` rows)**: Suggests ensemble models:
    *   `RandomForestClassifier()`
    *   `XGBClassifier()`
    *   `GradientBoostingClassifier()`
*   **High-Dimensional Data (`> 50` features)**: Appends Support Vector Machines:
    *   `SVC()`
*   **Class Imbalance (`imbalance_ratio > 3`)**: Appends class-weight-adjusted tree models:
    *   `RandomForestClassifier(class_weight='balanced')`
*   **Baseline Benchmark**: Always adds a dummy classifier for zero-information baseline comparison:
    *   `DummyClassifier(strategy='most_frequent')`

---

### 3. Regression Suggestions Heuristic
When the task is classified as regression, the suggestions adjust accordingly:

*   **Small Datasets (`< 1000` rows)**:
    *   `LinearRegression()`
    *   `KNeighborsRegressor()`
*   **Large Datasets (`>= 1000` rows)**:
    *   `RandomForestRegressor()`
    *   `XGBRegressor()`
    *   `GradientBoostingRegressor()`
*   **High-Dimensional Data (`> 50` features)**:
    *   `SVR()`
*   **Baseline Benchmark**:
    *   `DummyRegressor(strategy='mean')`

---

## API Usage Example

```python
# Initialize Selector
selector = ModelSelector(processed_df, target="price")

# Print console summary of characteristics and recommended models
selector.print_summary()

# Retrieve raw, un-fitted estimator objects
recommended_models = selector.suggest_models()
```

### Example Console Output
```text
Problem type: regression
Samples: 1500
Features (excluding target): 12

Recommended model instances:
- RandomForestRegressor
- XGBRegressor
- GradientBoostingRegressor
- DummyRegressor
```
