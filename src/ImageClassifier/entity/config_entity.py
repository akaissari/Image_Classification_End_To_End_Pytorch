from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    data_name:str


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class CallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    data_path: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_weights: str
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    test_dir: Path
    trained_model_path: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int