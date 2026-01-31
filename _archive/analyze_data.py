import pandas as pd
import numpy as np

try:
    file_path = "Dataset/final_financial_risk_dataset.csv"
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n--- DATASET INFO ---")
    print(df.info())
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- DUPLICATES ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df.describe())
    
    # Check if target column exists (assuming appropriate name from schema.yaml or observation)
    if 'Risk_Label' in df.columns:
        print("\n--- TARGET DISTRIBUTION (Risk_Label) ---")
        print(df['Risk_Label'].value_counts(normalize=True))
    elif 'Risk_Level' in df.columns:
        print("\n--- TARGET DISTRIBUTION (Risk_Level) ---")
        print(df['Risk_Level'].value_counts(normalize=True))
        
except Exception as e:
    print(f"Error analyzing dataset: {e}")
