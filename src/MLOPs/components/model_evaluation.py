import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from src.MLOPs.entity.config_entity import ModelEvaluationConfig
from src.MLOPs import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        model = joblib.load(self.config.model_path)
        X_test = pd.read_csv(self.config.test_features_path)
        y_test = pd.read_csv(self.config.test_labels_path).values.ravel()

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "accuracy": acc,
            "confusion_matrix": cm
        }

        with open(f"{self.config.root_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Model evaluation completed with accuracy: {acc}")
