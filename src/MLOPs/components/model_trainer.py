import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.MLOPs.entity.config_entity import ModelTrainerConfig
from src.MLOPs import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(f"{self.config.train_data_path}/X_train.csv")
        y_train = pd.read_csv(f"{self.config.train_data_path}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{self.config.test_data_path}/X_test.csv")
        y_test = pd.read_csv(f"{self.config.test_data_path}/y_test.csv").values.ravel()

        with mlflow.start_run():
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            model_path = f"{self.config.root_dir}/model.pkl"
            joblib.dump(model, model_path)

            logger.info(f"Model trained with accuracy: {acc}")
            logger.info(f"Model saved at: {model_path}")
