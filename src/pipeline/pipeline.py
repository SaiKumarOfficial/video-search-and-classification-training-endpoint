from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataPreparationConfig,ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataPreparationArtifact, ModelTrainerArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preparation import DataPreparation
from src.components.model_trainer import ModelTrainer
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
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact= data_ingestion_artifact,
                                             data_validation_config= self.data_validation_config)
            data_validation_artifact = data_validation.run_steps()

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_preparation(self, data_validataion_artifact: DataValidationArtifact) -> DataPreparationArtifact:
        try:
            self.data_preparation_config = DataPreparationConfig(self.training_pipeline_config)
            data_preparation = DataPreparation(data_validataion_artifact,
                                                self.data_preparation_config)
            data_preparation_artifact = data_preparation.run_steps()

            return data_preparation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_training(self, data_preparation_artifact: DataPreparationArtifact,
                             data_validation_artifact: DataValidationArtifact):
        try:
            self.model_training_config = ModelTrainerConfig(self.training_pipeline_config)
            model_training = ModelTrainer(self.model_training_config,
                                          data_validation_artifact,
                                          data_preparation_artifact)
            model_training_artifact = model_training.run_steps()
            return model_training_artifact
        except Exception as e:
            raise CustomException(e,sys)
    def run_pipeline(self):
        try:
            logging.info("===============Training Pipleline is start running=============")

            TrainPipeline.is_pipeline_running= True
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validataion_artifact: DataValidationArtifact  = self.start_data_validation(data_ingestion_artifact)
            data_preparation_artifact: DataPreparationArtifact = self.start_data_preparation(data_validataion_artifact)
            model_training_artifact: ModelTrainerArtifact = self.start_model_training(data_preparation_artifact, data_validataion_artifact)
            logging.info("=============Training Pipleline has successfully completed!!!===========")

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()

