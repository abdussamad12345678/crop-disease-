"""
model.py

Machine Learning module for crop disease prediction.
Uses Random Forest Classifier on environmental features.
"""

import os
import pickle
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier


MODEL_FILE = "model.pkl"
DATA_FILE = "data.csv"


class DiseasePredictionModel:
    """
    Handles training, loading, and prediction of disease risk.
    """

    def __init__(self):
        self.model = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from CSV.

        Returns:
            X (DataFrame): Features
            y (Series): Target labels
        """
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError("Dataset file not found.")

        data = pd.read_csv(DATA_FILE)

        required_cols = ["temperature", "humidity", "rainfall", "disease"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError("Dataset missing required columns.")

        X = data[["temperature", "humidity", "rainfall"]]
        y = data["disease"]

        return X, y

    def train(self) -> RandomForestClassifier:
        """
        Train the Random Forest model.

        Returns:
            Trained model
        """
        X, y = self.load_data()

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        # Save model
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        self.model = model
        return model

    def load(self) -> RandomForestClassifier:
        """
        Load model from file or train if not available.

        Returns:
            Loaded model
        """
        if not os.path.exists(MODEL_FILE):
            return self.train()

        try:
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
        except Exception:
            # Retrain if corrupted
            return self.train()

        return self.model

    def predict_proba(self, temp: float, humidity: float, rainfall: float) -> float:
        """
        Predict probability of disease.

        Args:
            temp (float): Temperature
            humidity (float): Humidity
            rainfall (float): Rainfall

        Returns:
            float: Probability of disease (0–1)
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load() first.")

        features = [[temp, humidity, rainfall]]
        prob = self.model.predict_proba(features)[0][1]

        return float(prob)


# -------------------------------
# Convenience functions
# -------------------------------

def load_model() -> DiseasePredictionModel:
    """
    Load model instance.

    Returns:
        DiseasePredictionModel
    """
    model = DiseasePredictionModel()
    model.load()
    return model


def predict_risk(model: DiseasePredictionModel,
                 temp: float,
                 humidity: float,
                 rainfall: float) -> float:
    """
    Wrapper for prediction.

    Returns:
        float: Disease probability
    """
    return model.predict_proba(temp, humidity, rainfall)
