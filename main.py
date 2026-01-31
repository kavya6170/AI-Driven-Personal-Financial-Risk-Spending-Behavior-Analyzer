from src.MLOPs.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.MLOPs.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.MLOPs.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.MLOPs.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.MLOPs.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

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

        logger.info(">>>>>> Data Transformation stage started <<<<<<")
        transformation = DataTransformationTrainingPipeline()
        transformation.main()
        logger.info(">>>>>> Data Transformation stage completed <<<<<<\n")

        logger.info(">>>>>> Model Trainer stage started <<<<<<")
        trainer = ModelTrainerTrainingPipeline()
        trainer.main()
        logger.info(">>>>>> Model Trainer stage completed <<<<<<")

        logger.info(">>>>>> Model Evaluation stage started <<<<<<")
        evaluation = ModelEvaluationTrainingPipeline()
        evaluation.main()
        logger.info(">>>>>> Model Evaluation stage completed <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
