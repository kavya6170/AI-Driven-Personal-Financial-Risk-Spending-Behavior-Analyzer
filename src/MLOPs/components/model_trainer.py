import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.MLOPs.entity.config_entity import ModelTrainerConfig
from src.MLOPs import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load data
        X_train = pd.read_csv(f"{self.config.train_data_path}/X_train.csv")
        y_train = pd.read_csv(f"{self.config.train_data_path}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{self.config.test_data_path}/X_test.csv")
        y_test = pd.read_csv(f"{self.config.test_data_path}/y_test.csv").values.ravel()

        # 🔽 ADD THIS (STEP 1)
        feature_names = list(X_train.columns)

        with mlflow.start_run():
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)

            mlflow.sklearn.log_model(model, "model")

            # Save model
            model_path = f"{self.config.root_dir}/model.pkl"    
            joblib.dump(model, model_path)

            # 🔽 ADD THIS (STEP 2)
            feature_path = f"{self.config.root_dir}/feature_names.json"
            with open(feature_path, "w") as f:
                json.dump(feature_names, f)

            logger.info(f"Model trained with accuracy: {test_acc}")
            logger.info(f"Model saved at: {model_path}")
            logger.info(f"Feature names saved at: {feature_path}")
