import pandas as pd
from src.MLOPs import logger
from src.MLOPs.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self) -> bool:
        try:
            df = pd.read_csv(self.config.data_path)
            schema_columns = self.config.schema_file.COLUMNS.keys()

            validation_status = True

            for col in schema_columns:
                if col not in df.columns:
                    if col == 'Risk_Label':
                        logger.warning(f"Target column {col} missing. This is OK for prediction but not for training.")
                    else:
                        validation_status = False
                        logger.error(f"Missing column: {col}")

            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e
