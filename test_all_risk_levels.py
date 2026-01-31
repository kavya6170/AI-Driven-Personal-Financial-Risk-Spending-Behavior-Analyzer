import pandas as pd
from src.MLOPs.components.prediction import RiskPredictor

model_path = "artifacts/model_trainer/model.pkl"
predictor = RiskPredictor(model_path)

print("="*60)
print("TESTING ALL THREE RISK LEVELS")
print("="*60)

# Test 1: LOW RISK
print("\n[TEST 1] LOW RISK SCENARIO")
print("-" * 60)
low_risk = pd.DataFrame([{
    'Total_Income': 50000,
    'Total_Expense': 15000,
    'Num_Transactions': 20,
    'Avg_Expense': 750,
    'Max_Expense': 2000,
    'Low_Balance_Freq': 2,
    'Expense_Income_Ratio': 0.30,
    'Top_Category_Spend': 5000
}])
result_low = predictor.predict(low_risk)
print(f"Risk Level: {result_low['risk_level']}")
print(f"Probability: {result_low['probability']:.4f} ({result_low['probability']*100:.2f}%)")

# Test 2: MEDIUM RISK
print("\n[TEST 2] MEDIUM RISK SCENARIO")
print("-" * 60)
medium_risk = pd.DataFrame([{
    'Total_Income': 35000,
    'Total_Expense': 25000,
    'Num_Transactions': 150,
    'Avg_Expense': 450,
    'Max_Expense': 5000,
    'Low_Balance_Freq': 8,
    'Expense_Income_Ratio': 0.71,
    'Top_Category_Spend': 12000
}])
result_medium = predictor.predict(medium_risk)
print(f"Risk Level: {result_medium['risk_level']}")
print(f"Probability: {result_medium['probability']:.4f} ({result_medium['probability']*100:.2f}%)")

# Test 3: HIGH RISK
print("\n[TEST 3] HIGH RISK SCENARIO")
print("-" * 60)
high_risk = pd.DataFrame([{
    'Total_Income': 30000,
    'Total_Expense': 45000,
    'Num_Transactions': 300,
    'Avg_Expense': 800,
    'Max_Expense': 12000,
    'Low_Balance_Freq': 20,
    'Expense_Income_Ratio': 1.50,
    'Top_Category_Spend': 20000
}])
result_high = predictor.predict(high_risk)
print(f"Risk Level: {result_high['risk_level']}")
print(f"Probability: {result_high['probability']:.4f} ({result_high['probability']*100:.2f}%)")

print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
