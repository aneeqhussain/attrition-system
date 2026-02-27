"""
Module 2: Data Preprocessing
Handles missing values, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_cols = ["Department", "JobRole"]
        self.numerical_cols = [
            "Age", "Salary", "YearsAtCompany", "PerformanceRating",
            "JobSatisfaction", "WorkHours", "Promotions",
            "PromotionFrequency", "TenureRatio", "SalaryPerYear"
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Promotion frequency (promotions per year)
        df["PromotionFrequency"] = df.apply(
            lambda r: r["Promotions"] / r["YearsAtCompany"] if r["YearsAtCompany"] > 0 else 0, axis=1
        )
        # Tenure ratio (years at company / age)
        df["TenureRatio"] = df["YearsAtCompany"] / df["Age"]
        # Salary per year of experience
        df["SalaryPerYear"] = df.apply(
            lambda r: r["Salary"] / r["YearsAtCompany"] if r["YearsAtCompany"] > 0 else r["Salary"], axis=1
        )
        return df

    def fit_transform(self, df: pd.DataFrame):
        df = self.engineer_features(df)

        # Encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Target
        target = (df["Attrition"] == "Yes").astype(int)

        feature_cols = self.categorical_cols + self.numerical_cols
        self.feature_columns = feature_cols
        X = df[feature_cols]

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_cols
        )

        return X_scaled, target

    def transform(self, df: pd.DataFrame):
        df = self.engineer_features(df)
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

        X = df[self.feature_columns]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns
        )
        return X_scaled

    def transform_single(self, employee_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame([employee_dict])
        return self.transform(df)

    def save(self, path: str = None):
        if path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "models", "preprocessor.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = None):
        if path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "models", "preprocessor.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    print("Feature shape:", X.shape)
    print("Attrition rate:", y.mean())
    preprocessor.save()
