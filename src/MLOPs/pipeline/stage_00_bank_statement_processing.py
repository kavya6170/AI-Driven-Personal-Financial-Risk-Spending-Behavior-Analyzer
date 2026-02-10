from src.MLOPs.config.configuration import ConfigurationManager
from src.MLOPs.components.bank_statement_parser import BankStatementParser
from src.MLOPs.components.statement_feature_extractor import StatementFeatureExtractor
from src.MLOPs import logger

STAGE_NAME = "Bank Statement Processing stage"

class BankStatementProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        statement_config = config.get_bank_statement_processing_config()
        
        # 1. Parse Statement
        parser = BankStatementParser(config=statement_config)
        transactions_df = parser.get_transactions()
        logger.info(f"Parsed {len(transactions_df)} transactions from statement")
        
        # 2. Extract Features
        extractor = StatementFeatureExtractor(config=statement_config)
        features_df = extractor.extract_features(transactions_df)
        logger.info(f"Extracted features for {len(features_df)} accounts/records")
        
        # 3. Save Processed Features (to be picked up by Data Ingestion or Transformation)
        features_df.to_csv(statement_config.processed_data_file, index=False)
        logger.info(f"Saved processed features to {statement_config.processed_data_file}")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BankStatementProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
