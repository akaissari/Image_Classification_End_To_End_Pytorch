from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.training import Training
from ImageClassifier import logger



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        callbacks_config = config.get_callback_config()
        training_config = config.get_training_config()
        base_model_config = config.get_base_model_config()
        training = Training(training_config=training_config, base_model_config = base_model_config, callback_config=callbacks_config)
        training.prepare_data()
        training.train()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        