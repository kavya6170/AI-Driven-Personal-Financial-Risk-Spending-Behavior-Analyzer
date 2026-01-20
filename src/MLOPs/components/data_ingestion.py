import os
import shutil
from pathlib import Path

from src.MLOPs import logger
from src.MLOPs.utils.common import get_size, create_directories
from src.MLOPs.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def copy_file(self):
        try:
            create_directories([self.config.root_dir])

            source = self.config.source_file
            destination = self.config.local_data_file

            logger.info(f"[DEBUG] Source path: {source}")
            logger.info(f"[DEBUG] Destination path: {destination}")
            logger.info(f"[DEBUG] Copying from {source} to {destination}")


            shutil.copy(source, destination)

            logger.info("Local dataset copied successfully")
            logger.info(f"File size: {get_size(Path(destination))}")

        except Exception as e:
            logger.exception(e)
            raise e

