from src.MLOPs.config.configuration import ConfigurationManager
from src.MLOPs.components.data_transformation import DataTransformation
from src.MLOPs import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        transformation_config = config.get_data_transformation_config()
        transformation = DataTransformation(config=transformation_config)
        transformation.transform_data()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
