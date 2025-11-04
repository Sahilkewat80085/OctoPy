# octopy/smart_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SmartEDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        """
        self.df = df

    def basic_info(self):
        """
        Print basic information: shape, data types, missing values, duplicates, and summary statistics.
        """
        print("Data Shape:", self.df.shape)
        print("\nData Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())
        print("\nDuplicate Rows:", self.df.duplicated().sum())
        print("\nSummary Statistics:\n", self.df.describe(include='all'))

    def value_counts(self):
        """
        Display value counts for all categorical columns.
        """
        print("\nCategorical Value Counts:")
        for col in self.df.select_dtypes(include='object'):
            print(f"\n{col}:\n", self.df[col].value_counts())

    def distribution_plots(self):
        """
        Plot histograms for all numerical features.
        """
        print("\nPlotting Distributions:")
        num_cols = self.df.select_dtypes(include=np.number).columns
        self.df[num_cols].hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    def boxplots(self):
        """
        Plot boxplots for all numerical features to detect outliers.
        """
        print("\nGenerating Boxplots:")
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.grid(True)
            plt.show()

    def correlation_heatmap(self):
        """
        Plot correlation heatmap of numerical features.
        """
        print("\nCorrelation Heatmap:")
        plt.figure(figsize=(10, 8))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def target_relation(self, target: str):
        """
        Plot boxplots between target variable and all numerical features.
        """
        print(f"\nTarget Interaction with: {target}")
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col != target:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=self.df[target], y=self.df[col])
                plt.title(f"{col} vs {target}")
                plt.xlabel(target)
                plt.ylabel(col)
                plt.grid(True)
                plt.show()

    def run_all(self, target: str = None):
        """
        Run all EDA functions in sequence. If target is provided, includes interaction plots.
        """
        self.basic_info()
        self.value_counts()   
        self.distribution_plots()
        self.boxplots()
        self.correlation_heatmap()
        if target:
            self.target_relation(target)
