from src.MLOPs.config.configuration import ConfigurationManager
from src.MLOPs.components.data_validation import DataValidation
from src.MLOPs import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=validation_config)
        data_validation.validate_columns()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
