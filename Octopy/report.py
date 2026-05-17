import os
import pickle
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def load_model(model_path):
    try:
        if model_path.endswith(".pkl") or model_path.endswith(".sav"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_test_data(x_path, y_path):
    try:
        X_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path)
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4)
        }
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}


def extract_hyperparameters(model):
    try:
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            important_keys = ['n_estimators', 'max_depth', 'learning_rate', 'kernel', 'C', 'alpha', 'gamma']
            return {k: v for k, v in all_params.items() if k in important_keys or isinstance(v, (int, float, str))}
        return {}
    except Exception as e:
        print(f"Error extracting hyperparameters: {e}")
        return {}


def generate_report(model_path, x_test_path=None, y_test_path=None):
    model = load_model(model_path)
    if not model:
        return

    report = {
        "Model Name": os.path.basename(model_path),
        "Hyperparameters": extract_hyperparameters(model)
    }

    if x_test_path and y_test_path:
        X_test, y_test = load_test_data(x_test_path, y_test_path)
        if X_test is not None and y_test is not None:
            report["Evaluation Metrics"] = evaluate_model(model, X_test, y_test)
    else:
        print("Skipping evaluation: X_test or y_test not provided.")

    output_path = "model_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[INFO] Report saved to {output_path}")

# Gives user
# Mean Absolute Error/ Mean Squared Error/ Root Mean Squared Error/ Coefficient of Determination(Alway ranges between -inf to 1) score
# very usefull if you want to see the efficiency of the any pickle trained model


# Example usage:
# generate_report("/path/to/model.pkl", "/path/to/X_test.csv", "/path/to/y_test.csv")
# generate_report("/path/to/model.pkl")
