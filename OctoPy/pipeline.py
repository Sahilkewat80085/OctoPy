# octopy/pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_squared_error
)

# Classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
)
from sklearn.neural_network import MLPClassifier

# Optional classification models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, TheilSenRegressor, RANSACRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Optional regression models
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


class PipelineBuilder:
    """A helper class to prepare data, select, and train machine learning models.

    This class handles the end-to-end process of splitting features and target,
    inferring the machine learning task type (classification or regression),
    and training any of the 30+ supported scikit-learn, XGBoost, or LightGBM models.
    """

    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
        """Initializes the PipelineBuilder.

        Args:
            df (pd.DataFrame): The input DataFrame containing features and target column.
            target (str): The name of the target column in df.
            problem_type (str, optional): The machine learning task type, either 'classification'
                or 'regression'. If None, the problem type is automatically inferred based
                on the target column data type and cardinality.
        """
        self.df = df
        self.target = target
        self.problem_type = problem_type or self._infer_problem_type()
        self.X = df.drop(columns=[target])
        self.y = df[target]

    def _infer_problem_type(self):
        if pd.api.types.is_numeric_dtype(self.df[self.target]):
            if self.df[self.target].nunique() <= 20:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def _get_model(self, name: str):
        name = name.lower()

        classification_models = {
            'logistic': LogisticRegression(),
            'sgd': SGDClassifier(),
            'randomforest': RandomForestClassifier(),
            'gradientboosting': GradientBoostingClassifier(),
            'adaboost': AdaBoostClassifier(),
            'extratrees': ExtraTreesClassifier(),
            'svc': SVC(),
            'knn': KNeighborsClassifier(),
            'naivebayes': GaussianNB(),
            'decisiontree': DecisionTreeClassifier(),
            'qda': QuadraticDiscriminantAnalysis(),
            'lda': LinearDiscriminantAnalysis(),
            'mlp': MLPClassifier(max_iter=500),
            'bagging': BaggingClassifier(),
            'histgb': HistGradientBoostingClassifier(),
            'xgboost': XGBClassifier() if XGBClassifier is not None else None,
            'lightgbm': LGBMClassifier() if LGBMClassifier is not None else None
        }

        regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elasticnet': ElasticNet(),
            'randomforestreg': RandomForestRegressor(),
            'gradientboostreg': GradientBoostingRegressor(),
            'adaboostreg': AdaBoostRegressor(),
            'extratreesreg': ExtraTreesRegressor(),
            'svr': SVR(),
            'knnr': KNeighborsRegressor(),
            'decisiontreereg': DecisionTreeRegressor(),
            'mlpreg': MLPRegressor(max_iter=500),
            'baggingreg': BaggingRegressor(),
            'histgbreg': HistGradientBoostingRegressor(),
            'xgboostreg': XGBRegressor() if XGBRegressor is not None else None,
            'lightgbmreg': LGBMRegressor() if LGBMRegressor is not None else None,
            'huberreg': HuberRegressor(),
            'theilsenreg': TheilSenRegressor(),
            'ransacreg': RANSACRegressor()
        }

        model = classification_models.get(name) if self.problem_type == 'classification' else regression_models.get(name)
        if model is None:
            if name in ['xgboost', 'xgboostreg'] and XGBClassifier is None:
                raise ImportError("XGBoost is not installed. Please run 'pip install xgboost' to use this model.")
            if name in ['lightgbm', 'lightgbmreg'] and LGBMClassifier is None:
                raise ImportError("LightGBM is not installed. Please run 'pip install lightgbm' to use this model.")
        return model

    def train(self, model_name='randomforest', test_size=0.2, random_state=42) -> tuple:
        """Trains a machine learning model and returns the fitted model and evaluation metrics.

        Args:
            model_name (str, optional): Shorthand name of the model to train. Defaults to 'randomforest'.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

        Returns:
            tuple: A tuple containing:
                - model: The trained estimator object (scikit-learn, XGBoost, or LightGBM model).
                - metrics (dict): A dictionary of evaluation metrics (e.g. Accuracy, F1 Score, R2 Score, MSE).

        Raises:
            ValueError: If the model_name is invalid or not supported.
            ImportError: If the requested model (e.g., xgboost, lightgbm) is not installed.
        """
        print(f"Training model: {model_name}")
        model = self._get_model(model_name)
        if model is None:
            raise ValueError("Invalid model name. Please check the supported model list.")

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error fitting model: {e}")
            raise

        y_pred = model.predict(X_test)

        if self.problem_type == 'classification':
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average='weighted'),
                "Report": classification_report(y_test, y_pred)
            }
        else:
            metrics = {
                "R2 Score": r2_score(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred)
            }

        return model, metrics

#code is sucessfully able to give the proper metrics following the ML model