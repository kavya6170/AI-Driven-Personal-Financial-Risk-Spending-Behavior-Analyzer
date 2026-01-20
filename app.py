from fastapi import FastAPI
import pandas as pd
from src.MLOPs.components.prediction import RiskPredictor

app = FastAPI(
    title="AI-Driven Personal Financial Risk Analyzer",
    description="Predicts financial risk level based on spending behavior",
    version="1.0"
)

MODEL_PATH = "artifacts/model_trainer/model.pkl"
predictor = RiskPredictor(MODEL_PATH)


@app.post("/predict")
def predict_risk(input_data: dict):
    """
    Input: JSON with feature values
    Output: Risk probability and risk level
    """
    df = pd.DataFrame([input_data])
    result = predictor.predict(df)
    return result


@app.get("/")
def health_check():
    return {"status": "API is running"}
