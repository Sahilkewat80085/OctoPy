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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class PipelineBuilder:
    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
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
            'xgboost': XGBClassifier(),
            'lightgbm': LGBMClassifier()
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
            'xgboostreg': XGBRegressor(),
            'lightgbmreg': LGBMRegressor(),
            'huberreg': HuberRegressor(),
            'theilsenreg': TheilSenRegressor(),
            'ransacreg': RANSACRegressor()
        }

        return classification_models.get(name) if self.problem_type == 'classification' else regression_models.get(name)

    def train(self, model_name='randomforest', test_size=0.2, random_state=42):
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