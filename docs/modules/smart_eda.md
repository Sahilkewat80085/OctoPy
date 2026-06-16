# Automated Exploratory Data Analysis (`smart_eda`)

The `smart_eda` module provides the `SmartEDA` class to automate exploratory data analysis (EDA) workflows. It generates terminal summaries and renders diagnostic visualizations using `matplotlib` and `seaborn`.

---

## The `SmartEDA` Class

Initialize the class by passing a pandas DataFrame:

```python
from Octopy.smart_eda import SmartEDA

eda = SmartEDA(df)
```

---

## Core Operations

### 1. Terminal Data Inspection
Print shapes, datatypes, missing values, duplicates, and general summary statistics:

```python
# Prints shape, duplicates, datatypes, null-counts, and numeric statistics
eda.basic_info()
```

### 2. Categorical Value Counts
For string/object columns, outputs the exact frequency distribution:

```python
# Prints a breakdown of occurrences for every category across object-type columns
eda.value_counts()
```

### 3. Distribution Visuals
Plots histograms with 30 bins for all numerical variables:

```python
# Renders a grid of histograms for all numeric features
eda.distribution_plots()
```

### 4. Outlier Diagnostics (Boxplots)
Generates boxplots for every numeric feature to identify skewness, range, and outliers:

```python
# Plots individual boxplots for each numeric feature
eda.boxplots()
```

### 5. Correlation Heatmap
Computes a Pearson correlation matrix for numerical features and renders a labeled color-graded heatmap:

```python
# Plots the correlation matrix of numerical variables
eda.correlation_heatmap()
```

### 6. Target Interaction Diagnostics
Plots boxplots of numerical features split against the classes of a specified target variable:

```python
# Plots feature distributions grouped by the target class
eda.target_relation(target="species")
```

---

## Running the Entire Suite

You can execute the entire analysis pipeline in sequence using `run_all()`:

```python
# Runs basic_info, value_counts, distribution_plots, boxplots, and correlation_heatmap.
# If a target column is specified, it also runs target_relation.
eda.run_all(target="species")
```

---

## Headless Execution Environments

> [!NOTE]
> **Matplotlib Blocking Behavior**: By default, `SmartEDA` methods call `plt.show()`, which opens interactive plot windows. In Jupyter notebooks, this displays plots inline. In standard python scripts, execution will pause until you close the window.
> If running in a headless server (such as a Docker container or an automated script), make sure to configure a headless backend before running `SmartEDA`:
> ```python
> import matplotlib
> matplotlib.use('Agg') # Force non-interactive SVG/PNG rendering backend
> ```
