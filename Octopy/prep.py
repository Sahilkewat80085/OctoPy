# octopy/prep.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the input DataFrame.
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}

    def handle_missing(self, strategy='mean', columns=None):
        """
        Handle missing values in specified columns or all numeric columns.
        strategy: 'mean', 'median', 'mode', or 'drop'
        """
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean' and self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)

    def encode_categorical(self, columns=None, method='onehot'):
        """
        Encode categorical columns using 'onehot' or 'label' encoding.
        Stores encoders for possible inverse transform.
        """
        if columns is None:
            columns = self.df.select_dtypes(include='object').columns

        for col in columns:
            if method == 'onehot':
                ohe = OneHotEncoder(sparse=False, drop='first')
                transformed = ohe.fit_transform(self.df[[col]])
                ohe_df = pd.DataFrame(transformed, columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]],
                                      index=self.df.index)
                self.df = pd.concat([self.df.drop(columns=[col]), ohe_df], axis=1)
                self.encoders[col] = ohe
            elif method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le

    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features using StandardScaler or MinMaxScaler.
        Stores scalers for inverse transform.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        for col in columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Scaling method must be 'standard' or 'minmax'")

            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler

    def get_processed_data(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame.
        """
        return self.df
