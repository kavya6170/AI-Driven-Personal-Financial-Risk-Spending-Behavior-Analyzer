from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_file: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    schema_file: Path
    status_file: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    C: float
    max_iter: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_features_path: Path
    test_labels_path: Path
