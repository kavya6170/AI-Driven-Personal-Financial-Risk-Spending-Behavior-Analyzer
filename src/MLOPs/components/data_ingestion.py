import os
import shutil
from src.MLOPs import logger
from src.MLOPs.utils.common import get_size
from pathlib import Path
from src.MLOPs.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def copy_local_file(self):
        try:
            source = self.config.source_url
            destination = self.config.local_data_file

            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)

            logger.info(f"File copied from {source} to {destination}")
            logger.info(f"File size: {get_size(destination)}")

        except Exception as e:
            raise e
