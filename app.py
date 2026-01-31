from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.MLOPs.components.prediction import RiskPredictor
from src.MLOPs.components.transaction_processor import TransactionProcessor, Transaction

app = FastAPI(
    title="AI-Driven Personal Financial Risk Analyzer",
    description="Predicts financial risk level based on spending behavior",
    version="1.0"
)

MODEL_PATH = "artifacts/model_trainer/model.pkl"
predictor = RiskPredictor(MODEL_PATH)
transaction_processor = TransactionProcessor()

class TransactionInput(BaseModel):
    Total_Income: float
    Total_Expense: float
    Num_Transactions: int
    Avg_Expense: float
    Max_Expense: float
    Low_Balance_Freq: int
    Expense_Income_Ratio: float
    Top_Category_Spend: float

@app.post("/predict")
def predict_risk(input_data: TransactionInput):
    """
    Input: JSON with feature values
    Output: Risk probability and risk level
    """
    df = pd.DataFrame([input_data.dict()])
    result = predictor.predict(df)
    return result


class TransactionBasedInput(BaseModel):
    """Input model for transaction-based prediction"""
    monthly_income: float
    transactions: List[Transaction]


@app.post("/predict/transactions")
def predict_risk_from_transactions(input_data: TransactionBasedInput):
    """
    Input: JSON with monthly income and list of transactions
    Output: Risk probability and risk level
    
    Example:
    {
        "monthly_income": 40000,
        "transactions": [
            {"amount": 500, "category": "food"},
            {"amount": 1200, "category": "rent"},
            {"amount": 150, "category": "entertainment"}
        ]
    }
    """
    # Calculate features from transactions
    features_df = transaction_processor.calculate_features(
        monthly_income=input_data.monthly_income,
        transactions=input_data.transactions
    )
    
    # Get prediction
    result = predictor.predict(features_df)
    
    return result


@app.get("/")
def health_check():
    return {"status": "API is running"}
