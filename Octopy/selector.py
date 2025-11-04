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
from xgboost import XGBClassifier, XGBRegressor

ModelType = Union[
    LogisticRegression, KNeighborsClassifier, RandomForestClassifier,
    GradientBoostingClassifier, XGBClassifier, DummyClassifier,
    LinearRegression, KNeighborsRegressor, RandomForestRegressor,
    GradientBoostingRegressor, SVR, XGBRegressor, DummyRegressor
]


class ModelSelector:
    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
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
            models += [
                RandomForestClassifier(),
                XGBClassifier(),
                GradientBoostingClassifier()
            ]

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
            models += [
                RandomForestRegressor(),
                XGBRegressor(),
                GradientBoostingRegressor()
            ]

        if self.num_features > 50:
            models.append(SVR())

        models.append(DummyRegressor(strategy='mean'))

        return models

    def print_summary(self):
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