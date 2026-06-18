# OctoPy

**OctoPy** is a modular Python library designed to automate and simplify machine learning workflows without obscuring the underlying details. It acts as an orchestrator for data preprocessing, exploratory data analysis (EDA), model suggestion, training, explanation, and report generation, while remaining transparent and developer-friendly.

Unlike traditional AutoML libraries that function as opaque "black boxes," OctoPy processes standard pandas DataFrames and returns raw, standard `scikit-learn` and `xgboost` estimator instances that you can inspect, modify, and serialize.

---

## Core Philosophy

*   **Transparency First**: No hidden pipelines. The library automates tedious code but exposes standard models, metrics, and parameters.
*   **Modular Architecture**: You are not forced into an end-to-end workflow. You can use the preprocessing module, the EDA engine, the explainability module, or the report compiler in complete isolation.
*   **Engineering-Oriented**: Text-first, fast, and simple. OctoPy is designed as a tool for developers and data scientists who require full control over their models.

---

## Core Modules

OctoPy is divided into seven main logical components:

| Module | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| **`prep`** | Standardized data preprocessing and feature scaling | `Preprocessor` |
| **`smart_eda`** | Automated exploratory data analysis and distribution plotting | `SmartEDA` |
| **`selector`** | Heuristic-driven model recommendation based on dataset metadata | `ModelSelector` |
| **`pipeline`** | Standardized model training and metric logging | `PipelineBuilder` |
| **`explain`** | Global feature importances (SHAP/permutation) and local instance explanations (LOFO) | `ModelExplainer` |
| **`comparison`** | Multi-model benchmark sweeps and visual execution speed comparisons | `ModelComparer` |
| **`report`** | Static HTML evaluation dashboards and structured JSON report compiles | `generate_report` |

---

## Simple Installation

Install OctoPy via PyPI:

```bash
pip install octopyx
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/sahilkewat80085/OctoPy.git
cd OctoPy

# Install package dependencies
pip install -e .
```

