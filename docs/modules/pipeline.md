# Training Pipeline Builder (`pipeline`)

The `pipeline` module features the `PipelineBuilder` class, which handles train-test splitting, mapping model names to standard estimators, and training and evaluating model configurations in a standardized loop.

---

## The `PipelineBuilder` Class

Initialize the builder with your processed DataFrame, target column name, and optional problem type:

```python
from Octopy.pipeline import PipelineBuilder

builder = PipelineBuilder(df, target="class_label")
```

---

## Supported Model Shorthands

`PipelineBuilder` maps string keys to standard model classes. Depending on the task (`classification` or `regression`), the keys resolve as follows:

### Classification Models
| Shorthand | Resolved Class |
| :--- | :--- |
| `'logistic'` | `LogisticRegression` |
| `'sgd'` | `SGDClassifier` |
| `'randomforest'` | `RandomForestClassifier` |
| `'gradientboosting'` | `GradientBoostingClassifier` |
| `'adaboost'` | `AdaBoostClassifier` |
| `'extratrees'` | `ExtraTreesClassifier` |
| `'svc'` | `SVC` |
| `'knn'` | `KNeighborsClassifier` |
| `'naivebayes'` | `GaussianNB` |
| `'decisiontree'` | `DecisionTreeClassifier` |
| `'qda'` | `QuadraticDiscriminantAnalysis` |
| `'lda'` | `LinearDiscriminantAnalysis` |
| `'mlp'` | `MLPClassifier(max_iter=500)` |
| `'bagging'` | `BaggingClassifier` |
| `'histgb'` | `HistGradientBoostingClassifier` |
| `'xgboost'` | `XGBClassifier` |
| `'lightgbm'` | `LGBMClassifier` |

---

### Regression Models
| Shorthand | Resolved Class |
| :--- | :--- |
| `'linear'` | `LinearRegression` |
| `'ridge'` | `Ridge` |
| `'lasso'` | `Lasso` |
| `'elasticnet'` | `ElasticNet` |
| `'randomforestreg'` | `RandomForestRegressor` |
| `'gradientboostreg'` | `GradientBoostingRegressor` |
| `'adaboostreg'` | `AdaBoostRegressor` |
| `'extratreesreg'` | `ExtraTreesRegressor` |
| `'svr'` | `SVR` |
| `'knnr'` | `KNeighborsRegressor` |
| `'decisiontreereg'` | `DecisionTreeRegressor` |
| `'mlpreg'` | `MLPRegressor(max_iter=500)` |
| `'baggingreg'` | `BaggingRegressor` |
| `'histgbreg'` | `HistGradientBoostingRegressor` |
| `'xgboostreg'` | `XGBRegressor` |
| `'lightgbmreg'` | `LGBMRegressor` |
| `'huberreg'` | `HuberRegressor` |
| `'theilsenreg'` | `TheilSenRegressor` |
| `'ransacreg'` | `RANSACRegressor` |

---

## Training and Evaluation

Use the `train()` method to initiate the training loop:

```python
model, metrics = builder.train(
    model_name="randomforest", 
    test_size=0.2, 
    random_state=42
)
```

### Parameters
*   **`model_name`** (`str`): The shorthand identifier of the estimator to train. Defaults to `'randomforest'`.
*   **`test_size`** (`float`): The split ratio for test evaluation (e.g., `0.2` for 20% test partition). Defaults to `0.2`.
*   **`random_state`** (`int`): Random seed to control data partitioning reproducibility. Defaults to `42`.

### Return Values

The method returns two variables:

1.  **`model`**: The raw, fitted scikit-learn or XGBoost model instance.
2.  **`metrics`** (`dict`): Evaluation metrics computed on the test split.
    *   *For Classification*:
        ```json
        {
          "Accuracy": 0.9667,
          "F1 Score": 0.9667,
          "Report": "             precision    recall  f1-score   support..."
        }
        ```
    *   *For Regression*:
        ```json
        {
          "R2 Score": 0.8912,
          "MSE": 1.234
        }
        ```
