import os
from ImageClassifier.constants import *
from ImageClassifier.utils.common import read_yaml, create_directories
from ImageClassifier.entity.config_entity import *

class ConfigurationManager:
    def __init__(
            self, 
            config_filepath: Path = CONFIG_FILE_PATH, 
            params_filepath: Path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        create_directories([config.train_dir])
        create_directories([config.val_dir])
        create_directories([config.test_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return base_model_config
    
    def get_callback_config(self) -> CallbacksConfig:
        config = self.config.prepare_callbacks
        model_checkpoints_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_checkpoints_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        callback_config =  CallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )
        return callback_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        base_model_config = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, self.config.data_ingestion.data_name)
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            data_path = Path(training.data_path),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(base_model_config.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        eval_config = EvaluationConfig(
            data_path=evaluation.data_path,
            model_path=evaluation.trained_model_path,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
