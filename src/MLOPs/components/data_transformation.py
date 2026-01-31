import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.MLOPs.entity.config_entity import DataTransformationConfig
from src.MLOPs import logger
from src.MLOPs.utils.common import create_directories
from src.MLOPs.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.params = ConfigurationManager().params

    def transform_data(self):
        df = pd.read_csv(self.config.data_path)

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.params.train_test_split.test_size,
            random_state=self.params.train_test_split.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include='number'))
        X_test_scaled = scaler.transform(X_test.select_dtypes(include='number'))

        X_train_num = pd.DataFrame(X_train_scaled, columns=X_train.select_dtypes(include='number').columns)
        X_test_num = pd.DataFrame(X_test_scaled, columns=X_test.select_dtypes(include='number').columns)

        create_directories([self.config.root_dir])

        X_train_num.to_csv(f"{self.config.root_dir}/X_train.csv", index=False)
        X_test_num.to_csv(f"{self.config.root_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{self.config.root_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{self.config.root_dir}/y_test.csv", index=False)
        
        # Save the scaler for use during prediction
        scaler_path = f"{self.config.root_dir}/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved at: {scaler_path}")

        logger.info("Data transformation completed successfully")
