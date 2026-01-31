import pandas as pd
from src.MLOPs.components.prediction import RiskPredictor

model_path = "artifacts/model_trainer/model.pkl"
predictor = RiskPredictor(model_path)

# sample input (must match X_train columns)
sample = pd.read_csv("artifacts/data_transformation/X_test.csv").head(1)

result = predictor.predict(sample)
print(result)
