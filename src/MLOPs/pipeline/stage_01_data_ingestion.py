from src.MLOPs.config.configuration import ConfigurationManager
from src.MLOPs.components.data_ingestion import DataIngestion
from src.MLOPs import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )

        # ✅ LOCAL FILE INGESTION (not download)
        data_ingestion.copy_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
