import pandas as pd
import pdfplumber
import os
from pathlib import Path
from src.MLOPs import logger
from src.MLOPs.entity.config_entity import BankStatementProcessingConfig
from typing import List, Dict

class BankStatementParser:
    def __init__(self, config: BankStatementProcessingConfig):
        self.config = config

    def parse_csv(self, file_path: Path) -> pd.DataFrame:
        """Parse bank statement from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded CSV bank statement from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV bank statement: {e}")
            raise e

    def parse_pdf(self, file_path: Path) -> pd.DataFrame:
        """Parse bank statement from PDF file (Mock implementation/Basic)"""
        try:
            # This is a basic implementation, actual PDF parsing depends heavily on bank format
            all_text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
            
            logger.info(f"Extracted text from PDF bank statement {file_path}")
            # In a real scenario, we'd use regex or table extraction (page.extract_table())
            # For now, we return an empty DF or raise error if real parsing is needed immediately
            raise NotImplementedError("Advanced PDF parsing requires specific bank format templates.")
            
        except Exception as e:
            logger.error(f"Error parsing PDF bank statement: {e}")
            raise e

    def get_transactions(self) -> pd.DataFrame:
        """Get transactions from the configured bank statement file"""
        file_path = self.config.bank_statements_file
        extension = file_path.suffix.lower()

        if extension == '.csv':
            return self.parse_csv(file_path)
        elif extension == '.pdf':
            return self.parse_pdf(file_path)
        elif extension in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
