from src.MLOPs.pipeline.stage_01_data_ingetion import DataIngestionTrainingPipeline
from src.MLOPs.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.MLOPs import logger

if __name__ == "__main__":
    try:
        logger.info(">>>>>> Data Ingestion stage started <<<<<<")
        ingestion = DataIngestionTrainingPipeline()
        ingestion.main()
        logger.info(">>>>>> Data Ingestion stage completed <<<<<<\n")

        logger.info(">>>>>> Data Validation stage started <<<<<<")
        validation = DataValidationTrainingPipeline()
        validation.main()
        logger.info(">>>>>> Data Validation stage completed <<<<<<\n")

    except Exception as e:
        logger.exception(e)
        raise e
