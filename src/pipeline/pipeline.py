from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import os,sys

class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.run_step()

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e,sys)
        



    def run_pipeline(self):
        try:
            logging.info("===============Training Pipleline is start running=============")

            TrainPipeline.is_pipeline_running= True
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            
            logging.info("=============Training Pipleline has successfully completed!!!===========")

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()

