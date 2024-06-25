from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.data_ingestion import DataIngestion
from ImageClassifier import logger




STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self) -> None:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e