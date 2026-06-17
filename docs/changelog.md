# Changelog

All notable changes to the **OctoPy** project will be documented on this page.

---

## [1.0.0] - 2026-06-16

This is the initial stable release of OctoPy, featuring a fully modular Python machine learning automation suite.

### Added
*   **`prep` Module**: Introduced the `Preprocessor` class supporting mean, median, mode, and row-dropping missing value imputation, one-hot encoding, label encoding, and MinMax/Standard feature scaling.
*   **`smart_eda` Module**: Introduced the `SmartEDA` class providing console reports and matplotlib-based charts (histograms, boxplots, correlation heatmaps, and target split interaction charts).
*   **`selector` Module**: Introduced the `ModelSelector` class implementing heuristic model selection rules based on dataset shape and class imbalance ratios.
*   **`pipeline` Module**: Introduced the `PipelineBuilder` class mapping 30+ classification and regression string identifiers to scikit-learn/XGBoost estimators, and automating training loops and evaluations.
*   **`explain` Module**: Introduced the `ModelExplainer` class supporting global feature impact (SHAP with fallback loops to tree importance, linear coefficients, or permutation diagnostics) and local single-instance tracing using Leave-One-Feature-Out (LOFO) logic visualized via terminal ASCII bar charts.
*   **`comparison` Module**: Introduced the `ModelComparer` class to sweep benchmarks across multiple estimators on mathematically identical splits, exporting console ASCII leaderboards and static HTML dashboards.
*   **`report` Module**: Introduced standalone functions to load models, load CSV test datasets, extract hyperparameters, and compile evaluations into structured JSON or interactive HTML reports.
*   **Documentation Site**: Initialized MkDocs-based documentation utilizing the Material theme, custom CSS, and auto-generated API pages linked via `mkdocstrings`.
