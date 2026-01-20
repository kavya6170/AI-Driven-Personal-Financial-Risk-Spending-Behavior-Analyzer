import joblib
import pandas as pd
from src.MLOPs import logger

class RiskPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_df: pd.DataFrame):
        prob = self.model.predict_proba(input_df)[0][1]

        if prob < 0.30:
            risk = "LOW"
        elif prob < 0.60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        result = {
            "risk_probability": round(float(prob), 4),
            "risk_level": risk
        }

        logger.info(f"Prediction result: {result}")
        return result
