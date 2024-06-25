from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.base_model import BaseModel
from ImageClassifier import logger

STAGE_NAME = "Prepare Base Model"

class BaseModelPreparationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self) -> None:
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
