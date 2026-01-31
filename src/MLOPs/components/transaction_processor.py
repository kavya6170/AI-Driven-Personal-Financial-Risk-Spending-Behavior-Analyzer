from typing import List
import pandas as pd
from pydantic import BaseModel
from src.MLOPs import logger


class Transaction(BaseModel):
    """Individual transaction model"""
    amount: float
    category: str


class TransactionProcessor:
    """Processes raw transaction data and calculates ML features"""
    
    def __init__(self):
        pass
    
    def calculate_features(self, monthly_income: float, transactions: List[Transaction]) -> pd.DataFrame:
        """
        Calculate all required ML features from raw transaction data
        
        Args:
            monthly_income: User's monthly income
            transactions: List of Transaction objects
            
        Returns:
            DataFrame with all 8 required features
        """
        if not transactions:
            raise ValueError("At least one transaction is required")
        
        if monthly_income <= 0:
            raise ValueError("Monthly income must be positive")
        
        # Extract transaction amounts and categories
        amounts = [t.amount for t in transactions]
        categories = [t.category for t in transactions]
        
        # Calculate basic metrics
        total_expense = sum(amounts)
        num_transactions = len(transactions)
        avg_expense = total_expense / num_transactions
        max_expense = max(amounts)
        
        # Calculate expense-income ratio
        expense_income_ratio = total_expense / monthly_income
        
        # Calculate top category spend
        category_totals = {}
        for transaction in transactions:
            category = transaction.category
            amount = transaction.amount
            category_totals[category] = category_totals.get(category, 0) + amount
        
        top_category_spend = max(category_totals.values()) if category_totals else 0
        
        # Estimate low balance frequency based on expense-income ratio
        # This is an approximation since we don't have running balance data
        if expense_income_ratio >= 0.9:
            low_balance_freq = 10
        elif expense_income_ratio >= 0.7:
            low_balance_freq = 7
        else:
            low_balance_freq = 3
        
        # Create feature dictionary
        features = {
            "Total_Income": monthly_income,
            "Total_Expense": total_expense,
            "Num_Transactions": num_transactions,
            "Avg_Expense": avg_expense,
            "Max_Expense": max_expense,
            "Low_Balance_Freq": low_balance_freq,
            "Expense_Income_Ratio": expense_income_ratio,
            "Top_Category_Spend": top_category_spend
        }
        
        logger.info(f"Calculated features from {num_transactions} transactions")
        logger.info(f"Features: {features}")
        
        # Return as DataFrame (matching the model's expected input format)
        return pd.DataFrame([features])
