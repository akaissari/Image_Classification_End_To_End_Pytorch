from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.evaluation import Evaluation
from ImageClassifier import logger



STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        test_config = config.get_evaluation_config()
        config_model = config.get_base_model_config()
        evaluator = Evaluation(config_eval=test_config, config_model=config_model)
        evaluator.evaluation()
        evaluator.save_score()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        