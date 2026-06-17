# Installation Guide

This guide covers setting up your Python environment and installing **OctoPy** along with its dependencies.

---

## Prerequisites

OctoPy requires **Python 3.8 or higher** (tested up to Python 3.12). 

We recommend installing OctoPy inside a virtual environment (`venv` or `conda`) to avoid package dependency conflicts with other projects.

---

## Installing via Virtual Environment (Recommended)

### Using `venv` (Standard Python)

1. Navigate to your project directory:
   ```bash
   cd my-machine-learning-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. Install the package locally:
   ```bash
   pip install -e /path/to/OctoPy
   ```

---

## Installing from Source

If you want to contribute to OctoPy, run the package tests, or customize the modules:

```bash
# Clone the repository
git clone https://github.com/sahilkewat80085/OctoPy.git
cd OctoPy

# Install in editable mode with development defaults
pip install -e .
```

---

## Dependencies

### Core Dependencies
These packages are automatically installed when you run `pip install -e .` (as specified in `setup.py`):

*   **`pandas`**: Data manipulation and DataFrame structure.
*   **`numpy`**: Mathematical computations and array handling.
*   **`scikit-learn`**: Underlying ML algorithms, split utilities, and metrics.
*   **`matplotlib`**: Headless plotting engine for visualizations.
*   **`seaborn`**: High-level statistical visualization formatting.
*   **`joblib`**: Model serialization and parallel processing helper.

### Optional & Conditional Dependencies
OctoPy is designed to degrade gracefully if advanced external libraries are missing:

*   **`xgboost` / `lightgbm`**: Recommended for gradient boosting classification and regression. If they are not installed, `PipelineBuilder` and `ModelSelector` will bypass them and fallback to standard scikit-learn tree estimators.
*   **`shap`**: Required for game-theoretic global feature explanations. If `shap` is missing, `ModelExplainer` will automatically fall back to standard Gini/Intrinsic feature importances, permutation importances, or linear coefficients without raising an error.

To install these optional libraries:
```bash
pip install xgboost lightgbm shap
```
