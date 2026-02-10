from src.MLOPs.constants import *
from src.MLOPs.utils.common import read_yaml, create_directories
from src.MLOPs.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, 
                                            ModelTrainerConfig, ModelEvaluationConfig, BankStatementProcessingConfig, BankStatementMapping)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_file=config.source_file,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_path=self.config.data_ingestion.local_data_file,
            schema_file=self.schema,
            status_file=config.status_file
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=self.config.data_ingestion.local_data_file,
            target_column=config.target_column
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=self.config.data_transformation.root_dir,
            test_data_path=self.config.data_transformation.root_dir,
            model_name=config.model_name,
            C=params.C,
            max_iter=params.max_iter
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=f"{self.config.model_trainer.root_dir}/model.pkl",
            test_features_path=f"{self.config.data_transformation.root_dir}/X_test.csv",
            test_labels_path=f"{self.config.data_transformation.root_dir}/y_test.csv"
        )

    
    def get_bank_statement_processing_config(self) -> BankStatementProcessingConfig:
        config = self.config.bank_statement_processing
        mapping_config = config.mapping

        create_directories([config.root_dir])

        mapping = BankStatementMapping(
            type_col=mapping_config.type_col,
            amount_col=mapping_config.amount_col,
            balance_col=mapping_config.balance_col,
            timestamp_col=mapping_config.timestamp_col,
            narration_col=mapping_config.narration_col,
            reference_col=mapping_config.reference_col
        )

        return BankStatementProcessingConfig(
            root_dir=Path(config.root_dir),
            bank_statements_file=Path(config.bank_statements_file),
            processed_data_file=Path(config.processed_data_file),
            category_keywords_path=Path(config.category_keywords_path),
            mapping=mapping,
            low_balance_threshold=config.low_balance_threshold
        )
