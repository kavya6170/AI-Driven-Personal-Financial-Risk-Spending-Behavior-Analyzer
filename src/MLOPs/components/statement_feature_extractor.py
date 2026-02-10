import pandas as pd
import yaml
from pathlib import Path
from src.MLOPs import logger
from src.MLOPs.entity.config_entity import BankStatementProcessingConfig
from src.MLOPs.utils.common import read_yaml

class StatementFeatureExtractor:
    def __init__(self, config: BankStatementProcessingConfig):
        self.config = config
        self.categories = read_yaml(Path(self.config.category_keywords_path))['categories']

    def categorize_narration(self, narration: str) -> str:
        """Categorize transaction based on narration keywords"""
        narration = str(narration).lower()
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in narration:
                    return category
        return "others"

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ML features from raw transactions"""
        try:
            m = self.config.mapping
            
            # Ensure timestamp is datetime
            df[m.timestamp_col] = pd.to_datetime(df[m.timestamp_col])
            
            # Categorize transactions
            df['category'] = df[m.narration_col].apply(self.categorize_narration)
            
            # Group by reference (Account) and potentially by month
            # For this MVP, we'll aggregate by reference
            results = []
            for ref, group in df.groupby(m.reference_col):
                credits = group[group[m.type_col] == 'CREDIT']
                debits = group[group[m.type_col] == 'DEBIT']
                
                total_income = credits[m.amount_col].sum()
                total_expense = debits[m.amount_col].sum()
                num_transactions = len(group)
                
                avg_expense = total_expense / len(debits) if len(debits) > 0 else 0
                max_expense = debits[m.amount_col].max() if len(debits) > 0 else 0
                
                # Low Balance Frequency
                low_balance_freq = len(group[group[m.balance_col] < self.config.low_balance_threshold])
                
                # Expense Income Ratio
                expense_income_ratio = total_expense / total_income if total_income > 0 else 0
                
                # Top Category Spend (excluding salary/income categories for expense analysis)
                category_spend = debits.groupby('category')[m.amount_col].sum()
                top_category_spend = category_spend.max() if not category_spend.empty else 0
                
                feature_dict = {
                    "Total_Income": total_income,
                    "Total_Expense": total_expense,
                    "Num_Transactions": num_transactions,
                    "Avg_Expense": avg_expense,
                    "Max_Expense": max_expense,
                    "Low_Balance_Freq": low_balance_freq,
                    "Expense_Income_Ratio": expense_income_ratio,
                    "Top_Category_Spend": top_category_spend
                }
                
                # Add Month as a placeholder if model expects it (some schemas had it)
                # For inference, we might just return the single feature set
                results.append(feature_dict)
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise e
