# Data Preprocessing (`prep`)

The `prep` module provides standardized, repeatable data cleaning and scaling operations using the `Preprocessor` class. It wraps scikit-learn transformers and maintains references to them, which allows you to inspect or reverse-engineer your data transformation pipeline.

---

## The `Preprocessor` Class

The `Preprocessor` class takes a pandas DataFrame, duplicates it internally to prevent in-place side effects on your source data, and performs in-place cleanups on the duplicate.

### Typical Imports

```python
from Octopy.prep import Preprocessor
```

---

## Key Capabilities

### 1. Handling Missing Values
The `handle_missing` method allows imputing or removing null values across specified columns or the entire DataFrame:

```python
preprocessor = Preprocessor(df)

# Impute missing numeric values using the median
preprocessor.handle_missing(strategy="median", columns=["age", "salary"])

# Impute categorical values using the most frequent value (mode)
preprocessor.handle_missing(strategy="mode", columns=["department"])

# Drop rows containing missing values in specified columns
preprocessor.handle_missing(strategy="drop", columns=["email"])
```

**Supported Strategies**:
*   `'mean'`: Replaces missing numeric values with the column average (only applies to numeric columns).
*   `'median'`: Replaces missing numeric values with the column median (only applies to numeric columns).
*   `'mode'`: Replaces missing values with the most frequent value. Works on both numeric and categorical columns.
*   `'drop'`: Removes any rows containing nulls in the selected columns.

---

### 2. Categorical Variable Encoding
Categorical data must be converted into numerical formats for machine learning models. The `encode_categorical` method offers two standard methods:

```python
# One-hot encode string columns (creates binary indicator columns, dropping the first category)
preprocessor.encode_categorical(columns=["country", "device_type"], method="onehot")

# Label encode target/ordered categories (converts categories to integer labels: 0, 1, 2...)
preprocessor.encode_categorical(columns=["education_level"], method="label")
```

Under the hood:
*   **One-Hot Encoding**: Uses `OneHotEncoder(sparse=False, drop='first')`. Column headers are automatically renamed to `{column_name}_{category_name}`.
*   **Label Encoding**: Uses `LabelEncoder()`.
*   All created encoder objects are saved in the `Preprocessor.encoders` dictionary mapping `{column_name} -> EncoderInstance`. This allows you to inspect classes later.

---

### 3. Feature Scaling
Numerical features often require standardization or scaling to assist optimization routines (e.g., in SVMs, Linear Models, and KNNs):

```python
# Standard scaling (zero mean, unit variance)
preprocessor.scale_features(columns=["income", "years_experience"], method="standard")

# Min-Max scaling (scales values strictly to the range [0, 1])
preprocessor.scale_features(columns=["rating"], method="minmax")
```

Under the hood:
*   **Standard scaling**: Uses `StandardScaler()`.
*   **Min-Max scaling**: Uses `MinMaxScaler()`.
*   All scaler instances are stored in `Preprocessor.scalers` mapping `{column_name} -> ScalerInstance`.

---

## Retrieving the Preprocessed Data

Once all transformations are called, retrieve the cleaned DataFrame using `get_processed_data()`:

```python
processed_df = preprocessor.get_processed_data()
```

---

## Common Mistakes

> [!WARNING]
> **Imputing Missing Data After Scaling**: Scaling algorithms will throw errors (or generate `NaN` metrics) if missing values are present. Always call `handle_missing()` before invoking `scale_features()`.

> [!CAUTION]
> **Applying One-Hot Encoding to High-Cardinality Columns**: If a column has hundreds of unique string categories, calling `method='onehot'` will expand your DataFrame's columns drastically, potentially leading to slow runtimes and model overfitting. Use label encoding or pre-filter categories for columns with high cardinality.
