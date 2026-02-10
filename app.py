from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.MLOPs.components.prediction import RiskPredictor
from src.MLOPs.components.transaction_processor import TransactionProcessor, Transaction
from src.MLOPs.components.bank_statement_parser import BankStatementParser
from src.MLOPs.components.statement_feature_extractor import StatementFeatureExtractor
from src.MLOPs.config.configuration import ConfigurationManager
from fastapi import UploadFile, File
import os
import shutil
from pathlib import Path

app = FastAPI(
    title="AI-Driven Personal Financial Risk Analyzer",
    description="Predicts financial risk level based on spending behavior",
    version="1.0"
)

MODEL_PATH = "artifacts/model_trainer/model.pkl"
predictor = RiskPredictor(MODEL_PATH)
transaction_processor = TransactionProcessor()

# Initialize configuration for statement processing
config_manager = ConfigurationManager()
statement_config = config_manager.get_bank_statement_processing_config()
statement_parser = BankStatementParser(statement_config)
feature_extractor = StatementFeatureExtractor(statement_config)

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


@app.post("/upload/bank-statement")
async def upload_bank_statement(file: UploadFile = File(...)):
    """
    Upload a bank statement (CSV/PDF) and get risk prediction
    """
    # Create temp directory for uploads
    upload_dir = Path("artifacts/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Parse statement
        if file.filename.endswith('.csv'):
            transactions_df = statement_parser.parse_csv(file_path)
        elif file.filename.endswith('.pdf'):
            transactions_df = statement_parser.parse_pdf(file_path)
        else:
            return {"error": "Unsupported file format. Please upload CSV or PDF."}
            
        # 2. Extract features
        features_df = feature_extractor.extract_features(transactions_df)
        
        # 3. Predict (takes first record if multiple accounts found)
        if not features_df.empty:
            result = predictor.predict(features_df.head(1))
            return {
                "filename": file.filename,
                "accounts_processed": len(features_df),
                "prediction": result
            }
        else:
            return {"error": "No features could be extracted from the statement."}
            
    except Exception as e:
        return {"error": f"Internal processing error: {str(e)}"}
    finally:
        # Cleanup
        if file_path.exists():
            os.remove(file_path)


@app.get("/")
def health_check():
    return {"status": "API is running"}
