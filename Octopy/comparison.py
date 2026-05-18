# Octopy/comparison.py

import time
import os
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from Octopy.pipeline import PipelineBuilder
from Octopy.selector import ModelSelector
from typing import List, Union, Dict, Tuple


class ModelComparer:
    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
        """
        Initializes the ModelComparer.
        Arguments:
            df: Input pandas DataFrame.
            target: Name of the target column.
            problem_type: 'classification' or 'regression'. If None, auto-inferred.
        """
        self.df = df.copy()
        self.target = target
        self.problem_type = problem_type or self._infer_problem_type()
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]
        
        self.leaderboard_df = None
        self.trained_models = {}
        self.metrics_list = []
        self.rank_metric = "Accuracy" if self.problem_type == "classification" else "R2_Score"

    def _infer_problem_type(self) -> str:
        if pd.api.types.is_numeric_dtype(self.df[self.target]):
            if self.df[self.target].nunique() <= 20:
                return 'classification'
            else:
                return 'regression'
        return 'classification'

    def compare(self, models: list = None, rank_by: str = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, dict]:
        """
        Trains and compares multiple machine learning models on mathematically identical splits.
        Arguments:
            models: List of model strings (e.g. ['randomforest', 'xgboost']) or custom estimator instances.
                    If None, uses ModelSelector to automatically select optimal candidates.
            rank_by: Metric column to sort the leaderboard. Defaults to Accuracy or R2_Score.
            test_size: Split ratio for testing data.
            random_state: Reproducibility state.
        Returns:
            leaderboard_df: Ranked comparison pandas DataFrame.
            trained_models: Dictionary of trained estimator instances.
        """
        self.trained_models = {}
        self.metrics_list = []

        # 1. Fallback to selector suggestions if model list is not specified
        if not models:
            print("[INFO] No models provided. Automatically querying ModelSelector for optimal suggestions...")
            selector = ModelSelector(self.df, self.target, self.problem_type)
            models = selector.suggest_models()

        if not models:
            raise ValueError("[ERROR] No models available to compare.")

        # 2. Mathematical Split: Splitting once guarantees identical splits across all models
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Flatten y targets if single-column DataFrames are passed
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]

        # Standardize arrays to prevent scikit-learn warnings
        y_train_arr = np.array(y_train)
        y_test_arr = np.array(y_test)

        print(f"[INFO] Initializing comparison across {len(models)} models...")
        print(f"[INFO] Task: {self.problem_type.upper()} | Train samples: {len(X_train)} | Test samples: {len(X_test)}")

        # 3. Model Benchmark Loop
        builder = PipelineBuilder(self.df, self.target, self.problem_type)
        
        for item in models:
            model_name = ""
            model_instance = None

            # Resolve model string or custom instances
            if isinstance(item, str):
                model_name = item
                try:
                    model_instance = builder._get_model(item)
                except Exception as e:
                    print(f"[WARNING] Skipping '{item}': Could not map string model. Error: {e}")
                    continue
            else:
                model_instance = item
                model_name = item.__class__.__name__

            if model_instance is None:
                print(f"[WARNING] Skipping '{model_name}': Estimator instance is null.")
                continue

            print(f" -> Benchmarking: {model_name}...")

            # Profile Training Runtime
            t_fit_start = time.perf_counter()
            try:
                model_instance.fit(X_train, y_train_arr)
                fit_duration = time.perf_counter() - t_fit_start
            except Exception as e:
                print(f" [ERROR] Model '{model_name}' failed to train. Error: {e}")
                continue

            # Profile Evaluation Runtime
            t_pred_start = time.perf_counter()
            try:
                y_pred = model_instance.predict(X_test)
                pred_duration = time.perf_counter() - t_pred_start
            except Exception as e:
                print(f" [ERROR] Model '{model_name}' failed during prediction. Error: {e}")
                continue

            y_pred_arr = np.array(y_pred)
            self.trained_models[model_name] = model_instance

            # 4. Compute Metrics
            metrics_dict = {
                "Model": model_name,
                "Train Time (s)": round(fit_duration, 4),
                "Pred Time (s)": round(pred_duration, 4)
            }

            if self.problem_type == "classification":
                metrics_dict["Accuracy"] = round(accuracy_score(y_test_arr, y_pred_arr), 4)
                metrics_dict["F1_Score"] = round(f1_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0), 4)
                metrics_dict["Precision"] = round(precision_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0), 4)
                metrics_dict["Recall"] = round(recall_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0), 4)
            else:
                metrics_dict["R2_Score"] = round(r2_score(y_test_arr, y_pred_arr), 4)
                metrics_dict["MAE"] = round(mean_absolute_error(y_test_arr, y_pred_arr), 4)
                metrics_dict["MSE"] = round(mean_squared_error(y_test_arr, y_pred_arr), 4)
                metrics_dict["RMSE"] = round(np.sqrt(metrics_dict["MSE"]), 4)

            self.metrics_list.append(metrics_dict)

        if not self.metrics_list:
            raise RuntimeError("[ERROR] All model benchmarks failed.")

        # 5. Leaderboard Compiler
        self.leaderboard_df = pd.DataFrame(self.metrics_list)
        
        # Determine sorting metric
        if rank_by:
            if rank_by in self.leaderboard_df.columns:
                self.rank_metric = rank_by
            else:
                print(f"[WARNING] Requested sorting metric '{rank_by}' not in metrics. Falling back to default.")
        
        # Sort values: standard metric scores should be descending (higher is better) 
        # except MAE, MSE, RMSE, or Train Time which should be ascending
        ascending_metrics = ["MAE", "MSE", "RMSE", "Train Time (s)", "Pred Time (s)"]
        sort_ascending = True if self.rank_metric in ascending_metrics else False
        
        self.leaderboard_df = self.leaderboard_df.sort_values(
            by=self.rank_metric, ascending=sort_ascending
        ).reset_index(drop=True)
        
        # Add a Rank column
        self.leaderboard_df.insert(0, "Rank", range(1, len(self.leaderboard_df) + 1))

        return self.leaderboard_df, self.trained_models

    def print_leaderboard(self):
        """
        Prints a highly readable, high-contrast visual ASCII table on the terminal.
        """
        if self.leaderboard_df is None:
            print("[WARNING] No leaderboard available. Call compare() first.")
            return

        print("\n" + "="*80)
        print(f"OCTOPY MODEL BENCHMARK LEADERBOARD (Ranked by {self.rank_metric})")
        print("="*80)
        
        # Format table columns dynamically
        cols = list(self.leaderboard_df.columns)
        header_str = " | ".join([f"{col:<12}" for col in cols])
        print(header_str)
        print("-" * len(header_str))

        for _, row in self.leaderboard_df.iterrows():
            row_str = " | ".join([f"{str(row[col]):<12}" for col in cols])
            print(row_str)
        print("="*80 + "\n")

    def generate_comparison_plots(self) -> dict:
        """
        Generates comparative static benchmark charts in vector SVG format.
        """
        plots_data = {}
        if self.leaderboard_df is None:
            return plots_data

        try:
            # 1. Performance Metric Chart
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.barplot(
                data=self.leaderboard_df,
                x=self.rank_metric,
                y="Model",
                color="#2c3e50",
                ax=ax,
                errorbar=None
            )
            ax.set_title(f"Model Performance Comparison ({self.rank_metric})", fontsize=11, fontweight="bold", pad=12, color="#2c3e50")
            ax.set_xlabel(self.rank_metric, fontsize=10, color="#34495e")
            ax.set_ylabel("Model Name", fontsize=10, color="#34495e")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            plots_data["perf_comparison"] = buf.getvalue().decode('utf-8')
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Performance comparison plot skipped: {e}")

        try:
            # 2. Execution Time comparison
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.barplot(
                data=self.leaderboard_df,
                x="Train Time (s)",
                y="Model",
                color="#7f8c8d",
                ax=ax,
                errorbar=None
            )
            ax.set_title("Training Execution Speed (Seconds)", fontsize=11, fontweight="bold", pad=12, color="#2c3e50")
            ax.set_xlabel("Time (s) - Lower is Faster", fontsize=10, color="#34495e")
            ax.set_ylabel("Model Name", fontsize=10, color="#34495e")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            plots_data["speed_comparison"] = buf.getvalue().decode('utf-8')
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Speed comparison plot skipped: {e}")

        return plots_data

    def generate_html_report(self, output_path: str = "comparison_report.html"):
        """
        Compiles the benchmark statistics into a sleek, premium, self-contained HTML page.
        """
        if self.leaderboard_df is None:
            print("[WARNING] No benchmark results to export. Call compare() first.")
            return

        plots = self.generate_comparison_plots()
        
        # 1. Compile leaderboard rows
        table_headers = "".join([f"<th>{col}</th>" for col in self.leaderboard_df.columns])
        table_rows = ""
        for _, row in self.leaderboard_df.iterrows():
            table_rows += "<tr>"
            for col in self.leaderboard_df.columns:
                val = row[col]
                # Bold the rank and model name
                if col == "Rank":
                    table_rows += f"<td style='font-weight:700; color:#2c3e50;'>{val}</td>"
                elif col == "Model":
                    table_rows += f"<td style='font-weight:600; text-align:left;'>{val}</td>"
                else:
                    table_rows += f"<td>{val}</td>"
            table_rows += "</tr>"

        # 2. Render SVG plots
        perf_chart = plots.get("perf_comparison", "<div style='padding:50px 0;'>Performance chart not available.</div>")
        speed_chart = plots.get("speed_comparison", "<div style='padding:50px 0;'>Execution chart not available.</div>")

        # 3. HTML layout compiler
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctoPy Benchmark Report</title>
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

        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 24px;
            margin-bottom: 24px;
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
            margin-top: 8px;
        }}

        th {{
            background-color: #fafbfc;
            font-weight: 700;
            color: #475569;
            padding: 12px 8px;
            border-bottom: 2px solid var(--border-color);
            text-align: right;
        }}

        th:first-child, th:nth-child(2) {{
            text-align: left;
        }}

        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
            text-align: right;
        }}

        td:first-child, td:nth-child(2) {{
            text-align: left;
        }}

        tr:hover {{
            background-color: #fafbfc;
        }}

        /* Graphic Grid layouts */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        @media (max-width: 768px) {{
            .charts-grid {{
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
                <h1>OctoPy Model Comparison Dashboard</h1>
                <span class="badge">{self.problem_type}</span>
            </div>
            <div class="header-meta">
                <span><strong>Target Feature:</strong> {self.target}</span>
                <span><strong>Benchmarked On:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
            </div>
        </header>

        <!-- Leaderboard Table Card -->
        <div class="card">
            <div class="card-title">Ranking Leaderboard</div>
            <table>
                <thead>
                    <tr>
                        {table_headers}
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <!-- Comparative Charts Grid -->
        <div class="charts-grid">
            <div class="card">
                <div class="card-title">Performance Benchmark</div>
                <div class="chart-box">
                    {perf_chart}
                </div>
            </div>
            <div class="card">
                <div class="card-title">Execution Speed (Fit Time)</div>
                <div class="chart-box">
                    {speed_chart}
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
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[INFO] Comparison HTML dashboard successfully saved to {output_path}")
