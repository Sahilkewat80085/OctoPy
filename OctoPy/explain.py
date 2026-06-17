# OctoPy/explain.py

import time
import io
import base64
import numpy as np
import pandas as pd

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from typing import List, Union, Dict, Tuple


class ModelExplainer:
    def __init__(self, model, X_train: pd.DataFrame, y_train, feature_names=None):
        """
        Initializes the ModelExplainer.
        Arguments:
            model: Trained scikit-learn/XGBoost/LightGBM model instance.
            X_train: Training feature DataFrame.
            y_train: Training target Series/Array.
            feature_names: Optional list of feature names.
        """
        self.model = model
        self.X_train = X_train.copy()
        self.y_train = np.array(y_train)
        self.feature_names = feature_names or list(X_train.columns)

    def explain_global(self, X_val: pd.DataFrame, y_val=None, save_path=None) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Generates global feature explanations using SHAP, or elegant mathematical fallbacks.
        Arguments:
            X_val: Validation/Test feature DataFrame.
            y_val: Validation/Test target Series/Array (required for Permutation Importance).
            save_path: Optional path to save the generated plot.
        Returns:
            df_impact: DataFrame of ranked features and impact scores.
            fig: Matplotlib Figure containing the visual explanation.
        """
        df_impact = None
        fig, ax = plt.subplots(figsize=(7, 4.5))
        explanation_method = ""

        # 1. Attempt SHAP Explanation
        try:
            import shap
            print("[INFO] SHAP detected. Attempting SHAP game-theoretic summary calculation...")
            
            # Use general SHAP Explainer
            explainer = shap.Explainer(self.model, self.X_train)
            shap_values = explainer(X_val)
            
            # Generate SHAP Summary plot
            plt.close(fig) # close previous standard figure
            fig = plt.figure(figsize=(7, 4.5))
            shap.summary_plot(shap_values, X_val, show=False)
            plt.title("SHAP Feature Impact Summary", fontsize=11, fontweight="bold", pad=12)
            plt.tight_layout()
            
            # Compile average feature impacts
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            if len(mean_shap.shape) > 1:
                mean_shap = mean_shap.mean(axis=1) # Average over target classes if multi-class
            
            df_impact = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": mean_shap
            })
            explanation_method = "SHAP Explainer"
        except Exception as e:
            # SHAP is missing or failed on this model type: fallback to default sklearn methods
            if "shap" in locals() or "shap" in globals():
                print(f"[INFO] SHAP failed on this estimator: {e}. Falling back to scikit-learn standard diagnostics...")
            else:
                print("[INFO] SHAP is not installed. Falling back to scikit-learn standard diagnostics...")
            
            # Reset figure
            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 4.5))

            # A. Tree-based Feature Importance fallback
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                df_impact = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": importances
                })
                explanation_method = "Intrinsic Feature Importance"

            # B. Linear Model Coefficients fallback
            elif hasattr(self.model, "coef_"):
                coefs = self.model.coef_
                # If multi-class, coef_ is shape (classes, features). Take absolute mean across classes
                if len(coefs.shape) > 1:
                    importances = np.mean(np.abs(coefs), axis=0)
                else:
                    importances = np.abs(coefs)
                
                df_impact = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": importances
                })
                explanation_method = "Intrinsic Linear Coefficients"

            # C. Model-Agnostic Permutation Importance fallback
            else:
                if y_val is None:
                    print("[WARNING] Target values (y_val) not provided. Fitting Permutation Importance on y_train baseline...")
                    X_p = self.X_train
                    y_p = self.y_train
                else:
                    X_p = X_val
                    y_p = np.array(y_val)

                print("[INFO] Computing model-agnostic Permutation Importance on test partition...")
                try:
                    result = permutation_importance(
                        self.model, X_p, y_p, n_repeats=5, random_state=42, n_jobs=-1
                    )
                    df_impact = pd.DataFrame({
                        "Feature": self.feature_names,
                        "Importance": result.importances_mean
                    })
                    explanation_method = "Model-Agnostic Permutation Importance"
                except Exception as ex:
                    print(f"[ERROR] Permutation Importance failed: {ex}. Falling back to default uniform scale.")
                    df_impact = pd.DataFrame({
                        "Feature": self.feature_names,
                        "Importance": np.ones(len(self.feature_names)) / len(self.feature_names)
                    })
                    explanation_method = "Uniform Baseline Fallback"

            # 2. Render Fallback Visualizations
            df_impact = df_impact.sort_values(by="Importance", ascending=False).reset_index(drop=True)
            
            # Render horizontal bar chart
            sns.barplot(
                data=df_impact.head(15), # Limit to top 15 features for clean viewing
                x="Importance",
                y="Feature",
                color="#2c3e50",
                ax=ax,
                errorbar=None
            )
            ax.set_title(f"Global Feature Impact ({explanation_method})", fontsize=11, fontweight="bold", pad=12, color="#2c3e50")
            ax.set_xlabel("Importance/Impact Score", fontsize=10, color="#34495e")
            ax.set_ylabel("Feature Name", fontsize=10, color="#34495e")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

        # 3. Save plot if path is provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Global explainability plot successfully saved to {save_path}")

        # Standardize df output
        df_impact = df_impact.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df_impact, fig

    def explain_prediction(self, sample_row: pd.Series) -> pd.DataFrame:
        """
        Traces a single instance prediction using a Leave-One-Feature-Out (LOFO) counterfactual contribution analyzer.
        Calculates how each specific feature value drives the prediction away from training baseline means.
        Arguments:
            sample_row: Pandas Series representing the record to explain.
        Returns:
            df_local: Local contributions DataFrame containing values and weights.
        """
        # Ensure row matches training features
        sample_row = sample_row[self.X_train.columns]
        
        # 1. Base Prediction on the actual sample
        pred_class = None
        proba_val = None
        
        try:
            if hasattr(self.model, "predict_proba"):
                # For classification, explain predicted class probability deviation
                probs = self.model.predict_proba(sample_row.to_frame().T)[0]
                pred_class = self.model.predict(sample_row.to_frame().T)[0]
                proba_val = probs[pred_class]
                y_base = proba_val
            else:
                # For regression, explain output target value deviation
                y_base = self.model.predict(sample_row.to_frame().T)[0]
        except Exception as e:
            raise RuntimeError(f"[ERROR] Could not compute prediction baseline for sample row: {e}")

        # 2. Counterfactual LOFO (Leave-One-Feature-Out) Loop
        contributions = {}
        for col in self.X_train.columns:
            x_temp = sample_row.copy()
            # Set feature j to its baseline training mean
            x_temp[col] = self.X_train[col].mean()
            
            try:
                if pred_class is not None:
                    # Class probability delta
                    y_temp = self.model.predict_proba(x_temp.to_frame().T)[0][pred_class]
                else:
                    # Regressor output delta
                    y_temp = self.model.predict(x_temp.to_frame().T)[0]
                
                # Contribution is how much the actual value shifts the output away from training mean baseline
                contributions[col] = round(y_base - y_temp, 4)
            except:
                contributions[col] = 0.0

        # 3. Assemble local df
        local_data = []
        for col in self.X_train.columns:
            local_data.append({
                "Feature": col,
                "Value": round(sample_row[col], 4) if isinstance(sample_row[col], (int, float, np.integer, np.floating)) else str(sample_row[col]),
                "Contribution": contributions[col]
            })

        df_local = pd.DataFrame(local_data)
        
        # Sort by absolute contribution strength (major factors first)
        df_local["Abs_Contribution"] = df_local["Contribution"].abs()
        df_local = df_local.sort_values(by="Abs_Contribution", ascending=False).drop(columns=["Abs_Contribution"]).reset_index(drop=True)

        # 4. High-Contrast Terminal ASCII Visualization
        print("\n" + "="*80)
        task_str = f"CLASSIFICATION (Class {pred_class} prob: {proba_val:.2%})" if pred_class is not None else "REGRESSION"
        print(f"OCTOPY INDIVIDUAL PREDICTION EXPLAINER ({task_str})")
        print("="*80)
        print(f"{'Feature':<18} | {'Value':<12} | {'Contribution':<12} | Weight Contribution Graph")
        print("-" * 80)

        # Find maximum contribution for visual scaling
        max_c = df_local["Contribution"].abs().max()
        max_c = max_c if max_c > 0 else 1.0

        for _, row in df_local.iterrows():
            feat = row["Feature"]
            val = str(row["Value"])
            contrib = row["Contribution"]
            
            # Create horizontal ASCII bar chart
            # 20-character width total
            num_bars = int(round((abs(contrib) / max_c) * 10))
            if contrib >= 0:
                # Positive push: bars on the right
                bar_str = "[" + "-"*10 + "="*num_bars + " "* (10 - num_bars) + "]"
            else:
                # Negative push: bars on the left
                bar_str = "[" + " "*(10 - num_bars) + "="*num_bars + "-"*10 + "]"

            contrib_str = f"+{contrib}" if contrib >= 0 else str(contrib)
            print(f"{feat:<18} | {val:<12} | {contrib_str:<12} | {bar_str}")
        print("="*80 + "\n")

        return df_local
