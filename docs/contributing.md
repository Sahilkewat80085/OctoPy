# Contributing to Octopy

Thank you for your interest in contributing to **Octopy**! We welcome bug fixes, documentation improvements, feature additions, and test enhancements.

---

## Code of Conduct

We aim to foster an open, professional, and welcoming developer community. Please ensure all interactions are collaborative and respectful.

---

## Getting Started

1.  **Fork the Repository**: Create a personal copy of the repository on GitHub.
2.  **Clone the Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/Octopy.git
    cd Octopy
    ```
3.  **Set Up a Development Environment**:
    We recommend setting up a virtual environment and installing the package in editable mode with development defaults:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    pip install -e .
    ```
4.  **Create a Feature Branch**:
    ```bash
    git checkout -b feature/my-new-feature
    ```

---

## Coding Standards

To maintain code quality and consistency:

*   **PEP 8 Compliance**: Follow standard Python styling guidelines (4 spaces for indentation, clean spacing).
*   **Clear Docstrings**: Every new class and public function should include descriptive docstrings detailing the purpose, arguments, and return types.
*   **Transparency**: Avoid wrapping estimators in custom classes that hide their API. Keep the design modular and clean.

---

## Running Local Tests

Before submitting a pull request, run the local verification tests to ensure no regressions were introduced:

```bash
# Run Explainability benchmarks
python test_explain.py

# Run Model Comparison benchmarks
python test_comparison.py
```

Ensure all benchmark leaderboards print successfully and HTML report outputs compile without errors.

---

## Submitting Pull Requests

1.  Commit your changes with descriptive commit messages.
2.  Push your feature branch to your fork on GitHub.
3.  Open a **Pull Request** against the `main` branch of the official Octopy repository.
4.  Provide a clear summary of the changes and link any related open issues.
