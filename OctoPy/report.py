# OctoPy/report.py

import os
import pickle
import json
import joblib
import io
import base64
import datetime
import pandas as pd
import numpy as np

# Set matplotlib backend to Agg to ensure headless script compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import is_classifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)


def load_model(model_path):
    try:
        if model_path.endswith(".pkl") or model_path.endswith(".sav"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None


def load_test_data(x_path, y_path):
    try:
        X_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path)
        # Flatten target if it has a single column to prevent warnings
        if isinstance(y_test, pd.DataFrame):
            if y_test.shape[1] == 1:
                y_test = y_test.iloc[:, 0]
        return X_test, y_test
    except Exception as e:
        print(f"[ERROR] Error loading test data: {e}")
        return None, None


def extract_hyperparameters(model):
    try:
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            important_keys = [
                'n_estimators', 'max_depth', 'learning_rate', 'kernel', 'C', 
                'alpha', 'gamma', 'criterion', 'min_samples_split', 'solver'
            ]
            return {k: v for k, v in all_params.items() if k in important_keys or isinstance(v, (int, float, str))}
        return {}
    except Exception as e:
        print(f"[ERROR] Error extracting hyperparameters: {e}")
        return {}


def detect_model_type(model) -> bool:
    """
    Detects model type.
    Returns:
        True if the model is a classifier, False if it is a regressor.
    """
    # Check standard sklearn classifier
    if is_classifier(model):
        return True
    # Fallback checks for common wrappers or estimators with classes_ attribute
    if hasattr(model, "classes_"):
        return True
    # Check common regressor attributes
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        # Let's inspect the model class name as a fallback
        class_name = model.__class__.__name__.lower()
        if "regressor" in class_name or "regression" in class_name or "svr" in class_name:
            return False
    return False


def evaluate_model(model, X_test, y_test, is_clf: bool) -> dict:
    """
    Evaluates a model dynamically based on its classification or regression type.
    """
    try:
        y_pred = model.predict(X_test)
        
        # Ensure we have consistent numpy formats
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        
        if is_clf:
            accuracy = accuracy_score(y_test_arr, y_pred_arr)
            precision = precision_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0)
            recall = recall_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0)
            f1 = f1_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0)
            
            # Simple unique labels sorted
            labels = sorted(list(set(y_test_arr)))
            cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)
            
            return {
                "Accuracy": round(accuracy, 4),
                "F1_Score": round(f1, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "Confusion_Matrix": cm.tolist(),
                "Labels": [str(lbl) for lbl in labels]
            }
        else:
            mae = mean_absolute_error(y_test_arr, y_pred_arr)
            mse = mean_squared_error(y_test_arr, y_pred_arr)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_arr, y_pred_arr)
            return {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2_Score": round(r2, 4)
            }
    except Exception as e:
        print(f"[ERROR] Error evaluating model: {e}")
        return {}


def generate_plots_base64(model, X_test, y_test, is_clf: bool) -> dict:
    """
    Generates Matplotlib plots and returns them as base64-encoded SVG strings
    to ensure crisp visualization in the HTML dashboard.
    """
    plots_data = {}
    
    # 1. Feature Importance / Explainability Plot (Unified)
    try:
        from OctoPy.explain import ModelExplainer
        explainer = ModelExplainer(model, X_test, y_test)
        _, fig = explainer.explain_global(X_test, y_test)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        plots_data["feature_importance"] = buf.getvalue().decode('utf-8')
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Feature impact explainability plot generation skipped: {e}")

    # 2. Task-specific evaluation plots
    try:
        y_test_arr = np.array(y_test)
        y_pred = model.predict(X_test)
        y_pred_arr = np.array(y_pred)
        
        if is_clf:
            # Confusion Matrix Heatmap
            labels = sorted(list(set(y_test_arr)))
            cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                cbar=False,
                xticklabels=[str(lbl) for lbl in labels],
                yticklabels=[str(lbl) for lbl in labels],
                ax=ax,
                annot_kws={"size": 11, "weight": "bold"}
            )
            ax.set_title("Confusion Matrix Heatmap", fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
            ax.set_xlabel("Predicted Label", fontsize=10, color="#34495e")
            ax.set_ylabel("True Label", fontsize=10, color="#34495e")
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            plots_data["eval_plot"] = buf.getvalue().decode('utf-8')
            plt.close(fig)
        else:
            # Regression Actual vs Predicted plot
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test_arr, y_pred_arr, color="#34495e", alpha=0.6, edgecolors='none', s=40)
            
            # Perfect reference diagonal line
            min_val = min(y_test_arr.min(), y_pred_arr.min())
            max_val = max(y_test_arr.max(), y_pred_arr.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', color="#c0392b", linewidth=1.5, label="Perfect Fit")
            
            ax.set_title("Actual vs. Predicted Plot", fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
            ax.set_xlabel("Actual Values", fontsize=10, color="#34495e")
            ax.set_ylabel("Predicted Values", fontsize=10, color="#34495e")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            plots_data["eval_plot"] = buf.getvalue().decode('utf-8')
            plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Task evaluation plot generation skipped: {e}")
        
    return plots_data


def compile_html_report(model_name, is_clf, metrics, hyperparameters, plots, dataset_info):
    """
    Compiles variables into a clean, minimalist, highly professional HTML page template.
    """
    # Compile hyperparameters table rows
    params_rows = ""
    if hyperparameters:
        for k, v in sorted(hyperparameters.items()):
            params_rows += f"<tr><td class='key'>{k}</td><td class='val'>{v}</td></tr>"
    else:
        params_rows = "<tr><td colspan='2' style='color:#7f8c8d; text-align:center;'>No hyperparameters detected.</td></tr>"

    # Compile metrics KPIs
    kpi_cards = ""
    if is_clf:
        kpi_cards = f"""
        <div class="kpi-card">
            <span class="kpi-label">ACCURACY</span>
            <span class="kpi-val">{metrics.get("Accuracy", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">F1 SCORE (WEIGHTED)</span>
            <span class="kpi-val">{metrics.get("F1_Score", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">PRECISION (WEIGHTED)</span>
            <span class="kpi-val">{metrics.get("Precision", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">RECALL (WEIGHTED)</span>
            <span class="kpi-val">{metrics.get("Recall", "N/A")}</span>
        </div>
        """
    else:
        kpi_cards = f"""
        <div class="kpi-card">
            <span class="kpi-label">R² SCORE</span>
            <span class="kpi-val">{metrics.get("R2_Score", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">MEAN ABSOLUTE ERROR</span>
            <span class="kpi-val">{metrics.get("MAE", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">ROOT MEAN SQUARED ERROR</span>
            <span class="kpi-val">{metrics.get("RMSE", "N/A")}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">MEAN SQUARED ERROR</span>
            <span class="kpi-val">{metrics.get("MSE", "N/A")}</span>
        </div>
        """

    # Compile dataset summary information
    dataset_rows = ""
    for k, v in dataset_info.items():
        dataset_rows += f"<tr><td class='key'>{k}</td><td class='val'>{v}</td></tr>"

    # SVG charts embedding
    feat_chart = plots.get("feature_importance", "<div style='color:#7f8c8d; text-align:center; padding:50px 0;'>Feature Importances not available for this model.</div>")
    eval_chart = plots.get("eval_plot", "<div style='color:#7f8c8d; text-align:center; padding:50px 0;'>Evaluation Chart not available.</div>")

    # Pure CSS minimalist Layout
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctoPy Evaluation Report - {model_name}</title>
    <style>
        :root {{
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-main: #2c3e50;
            --text-sub: #7f8c8d;
            --border-color: #e2e8f0;
            --primary: #2c3e50;
            --primary-accent: #34495e;
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.6;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        /* Header block */
        header {{
            background-color: var(--primary);
            color: #ffffff;
            padding: 24px 32px;
            border-radius: 6px;
            margin-bottom: 24px;
            border-bottom: 3px solid #1a252f;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        header h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        .header-meta {{
            font-size: 13px;
            opacity: 0.8;
            margin-top: 6px;
            display: flex;
            gap: 20px;
        }}

        .badge {{
            background-color: rgba(255, 255, 255, 0.15);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}

        /* Key Metrics Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .kpi-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-sub);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}

        .kpi-val {{
            font-size: 26px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.5px;
        }}

        /* Secondary Grid Section */
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}

        @media (max-width: 768px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 24px;
        }}

        .card-title {{
            font-size: 14px;
            font-weight: 700;
            color: var(--primary);
            border-bottom: 1.5px solid var(--border-color);
            padding-bottom: 8px;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        th, td {{
            padding: 8px 0;
            text-align: left;
            border-bottom: 1px solid #f1f5f9;
        }}

        .key {{
            font-weight: 600;
            color: #475569;
            width: 50%;
        }}

        .val {{
            color: #1e293b;
            text-align: right;
            font-family: monospace;
            font-size: 12px;
        }}

        /* Chart grids */
        .charts-card {{
            grid-column: span 2;
        }}

        @media (max-width: 768px) {{
            .charts-card {{
                grid-column: span 1;
            }}
        }}

        .charts-container {{
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
            align-items: center;
        }}

        @media (max-width: 768px) {{
            .charts-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-box {{
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid #f1f5f9;
            border-radius: 4px;
            padding: 12px;
            background-color: #fafbfc;
        }}

        .chart-box svg {{
            width: 100%;
            height: auto;
            max-height: 380px;
        }}

        /* Footer */
        footer {{
            text-align: center;
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
            letter-spacing: 0.2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <h1>OctoPy Model Evaluation Report</h1>
                <span class="badge">{"Classification" if is_clf else "Regression"}</span>
            </div>
            <div class="header-meta">
                <span><strong>Model Name:</strong> {model_name}</span>
                <span><strong>Date:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
            </div>
        </header>

        <!-- KPI Metrics -->
        <div class="kpi-grid">
            {kpi_cards}
        </div>

        <!-- Details Grid -->
        <div class="main-grid">
            <!-- Left Card: Model Hyperparameters -->
            <div class="card">
                <div class="card-title">Model Hyperparameters</div>
                <table>
                    <tbody>
                        {params_rows}
                    </tbody>
                </table>
            </div>

            <!-- Right Card: Test Dataset Details -->
            <div class="card">
                <div class="card-title">Test Dataset Overview</div>
                <table>
                    <tbody>
                        {dataset_rows}
                    </tbody>
                </table>
            </div>

            <!-- Visual Charts Block -->
            <div class="card charts-card">
                <div class="card-title">Evaluation & Feature Importance Visualizations</div>
                <div class="charts-container">
                    <div class="chart-box">
                        {feat_chart}
                    </div>
                    <div class="chart-box">
                        {eval_chart}
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Report generated by OctoPy — Automation without hiding the ML workflow.
        </footer>
    </div>
</body>
</html>
"""
    return html_content


def generate_report(model_path, x_test_path=None, y_test_path=None, format=None):
    """
    Loads model, evaluates it, extracts parameters, and generates report formats.
    
    If format is not specified (None), it prompts the user in interactive terminal sessions.
    If no interactive terminal is detected or input errors occur, it defaults to 'both'.
    """
    model = load_model(model_path)
    if not model:
        print("[ERROR] Could not load model. Aborting report generation.")
        return

    is_clf = detect_model_type(model)
    hyperparameters = extract_hyperparameters(model)
    model_name = os.path.basename(model_path)

    # Core report dict
    report_dict = {
        "Model Name": model_name,
        "Task Type": "Classification" if is_clf else "Regression",
        "Hyperparameters": hyperparameters
    }

    metrics = {}
    dataset_info = {}
    plots = {}

    if x_test_path and y_test_path:
        X_test, y_test = load_test_data(x_test_path, y_test_path)
        if X_test is not None and y_test is not None:
            metrics = evaluate_model(model, X_test, y_test, is_clf)
            report_dict["Evaluation Metrics"] = metrics
            
            # Extract basic dataset metrics
            dataset_info = {
                "Sample Count": len(X_test),
                "Feature Count": X_test.shape[1],
                "Missing Values": int(X_test.isnull().sum().sum()),
                "Target Classes": len(set(y_test)) if is_clf else "Continuous"
            }
            
            # Generate Base64 SVG plots for the HTML format
            plots = generate_plots_base64(model, X_test, y_test, is_clf)
        else:
            print("[WARNING] Skipping evaluation: X_test or y_test failed to load.")
    else:
        print("[INFO] Skipping evaluation: X_test or y_test path not provided.")

    # Determine report format (interactive check if None)
    if format is None:
        try:
            print("\nSelect Output Format for OctoPy Evaluation Report:")
            print("[1] JSON Report")
            print("[2] HTML Report")
            print("[3] Both JSON and HTML Reports")
            choice = input("Enter choice (1, 2, or 3) [default: 2]: ").strip()
            
            if choice == "1":
                format = "json"
            elif choice == "3":
                format = "both"
            else:
                format = "html"
        except Exception:
            # Non-interactive or headless fallbacks
            format = "both"

    # Save outputs
    if format in ["json", "both"]:
        json_output_path = "model_report.json"
        # We strip Confusion Matrix numpy lists out of clean JSON to keep it perfectly structured
        json_report = report_dict.copy()
        if "Evaluation Metrics" in json_report and "Confusion_Matrix" in json_report["Evaluation Metrics"]:
            del json_report["Evaluation Metrics"]["Confusion_Matrix"]
            if "Labels" in json_report["Evaluation Metrics"]:
                del json_report["Evaluation Metrics"]["Labels"]
                
        with open(json_output_path, "w") as f:
            json.dump(json_report, f, indent=4)
        print(f"[INFO] JSON Report saved to {json_output_path}")

    if format in ["html", "both"]:
        html_output_path = "model_report.html"
        html_content = compile_html_report(model_name, is_clf, metrics, hyperparameters, plots, dataset_info)
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[INFO] HTML Report saved to {html_output_path}")
