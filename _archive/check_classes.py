import joblib
import pandas as pd

try:
    model = joblib.load("artifacts/model_trainer/model.pkl")
    print(f"Model classes: {model.classes_}")
except Exception as e:
    print(f"Error: {e}")
