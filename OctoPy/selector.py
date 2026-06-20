# octopy/selector.py

from typing import List, Union
import pandas as pd
import numpy as np

# Actual models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
# Optional models
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

ModelType = Union[
    LogisticRegression, KNeighborsClassifier, RandomForestClassifier,
    GradientBoostingClassifier, Union[None, XGBClassifier], DummyClassifier,
    LinearRegression, KNeighborsRegressor, RandomForestRegressor,
    GradientBoostingRegressor, SVR, Union[None, XGBRegressor], DummyRegressor
]


class ModelSelector:
    """A rule-based model recommender system.

    ModelSelector evaluates the characteristics of a dataset (such as number of samples,
    number of features, and target imbalance ratio) to recommend suitable machine learning models
    from scikit-learn and XGBoost.
    """

    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
        """Initializes the ModelSelector.

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
        self.num_samples = len(df)
        self.num_features = df.shape[1] - 1
        self.target_unique = df[target].nunique()
        self.imbalance_ratio = self._calculate_imbalance_ratio()

    def _infer_problem_type(self) -> str:
        if pd.api.types.is_numeric_dtype(self.df[self.target]):
            return 'classification' if self.df[self.target].nunique() <= 20 else 'regression'
        return 'classification'

    def _calculate_imbalance_ratio(self) -> float:
        if self.problem_type != 'classification':
            return 1.0
        counts = self.df[self.target].value_counts()
        return counts.max() / counts.min() if len(counts) > 1 else 1.0

    def suggest_models(self) -> List[ModelType]:
        """Suggests a list of recommended machine learning model instances based on the dataset metrics.

        Returns:
            List[ModelType]: A list of un-fitted estimator objects (from scikit-learn or XGBoost)
                recommended for the dataset.
        """
        return (
            self._suggest_classification_models()
            if self.problem_type == 'classification'
            else self._suggest_regression_models()
        )

    def _suggest_classification_models(self) -> List[ModelType]:
        models = []

        if self.num_samples < 1000:
            models += [LogisticRegression(), KNeighborsClassifier()]
        else:
            models += [RandomForestClassifier()]
            if XGBClassifier is not None:
                models.append(XGBClassifier())
            models += [GradientBoostingClassifier()]

        if self.num_features > 50:
            models.append(SVC())

        if self.imbalance_ratio > 3:
            models.append(RandomForestClassifier(class_weight='balanced'))

        models.append(DummyClassifier(strategy='most_frequent'))

        return models

    def _suggest_regression_models(self) -> List[ModelType]:
        models = []

        if self.num_samples < 1000:
            models += [LinearRegression(), KNeighborsRegressor()]
        else:
            models += [RandomForestRegressor()]
            if XGBRegressor is not None:
                models.append(XGBRegressor())
            models += [GradientBoostingRegressor()]

        if self.num_features > 50:
            models.append(SVR())

        models.append(DummyRegressor(strategy='mean'))

        return models

    def print_summary(self):
        """Prints a human-readable summary of the dataset metrics and recommended model names."""
        print(f"Problem type: {self.problem_type}")
        print(f"Samples: {self.num_samples}")
        print(f"Features (excluding target): {self.num_features}")
        if self.problem_type == 'classification':
            print(f"Target classes: {self.target_unique}")
            print(f"Class imbalance ratio: {self.imbalance_ratio:.2f}")
        print("\nRecommended model instances:")
        for model in self.suggest_models():
            print(f"- {model.__class__.__name__}")

#able to analize the insights of the dataset and giveing the best ML model on you should train the dataset