import joblib
import pandas as pd
import numpy as np
from src.MLOPs import logger

class RiskPredictor:
    def __init__(self, model_path, scaler_path="artifacts/data_transformation/scaler.pkl"):
        self.model = joblib.load(model_path)
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
        except FileNotFoundError:
            logger.warning(f"Scaler not found at {scaler_path}. Predictions will use unscaled features.")
            self.scaler = None

    def predict(self, input_df: pd.DataFrame):
        # Scale the input features if scaler is available
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_df)
            input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
        else:
            input_df_scaled = input_df
        
        # Get probabilities for all three classes
        probabilities = self.model.predict_proba(input_df_scaled)[0]
        classes = self.model.classes_
        
        # Create a dictionary mapping class names to probabilities
        prob_dict = {
            classes[i]: float(probabilities[i]) 
            for i in range(len(classes))
        }
        
        # Get the predicted class based on highest probability
        predicted_class = classes[np.argmax(probabilities)]
        
        # Business rule override: Force HIGH if expense ratio >= 0.9
        expense_ratio = input_df["Expense_Income_Ratio"].values[0]
        if expense_ratio >= 0.9:
            predicted_class = "HIGH"
        
        result = {
            "risk_level": predicted_class,
            "probability": prob_dict.get(predicted_class, 0.0)
        }

        logger.info(f"Prediction result: {result}")
        return result

