from src.entity.config_entity import ModelPusherConfig 
from src.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact

from src.exception import CustomException
from src.logger import logging
import os,sys, shutil

class ModelPusher:
    
    def __init__(self,model_pusher_config: ModelPusherConfig,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def run_steps(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            #creating model pusher dir to save model,this model is for training purpose
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok= True)
            shutil.copy(src = trained_model_path,dst = model_file_path)

            #saved model dir , this model is for production purpose
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok= True)
            shutil.copy(src = trained_model_path,dst= saved_model_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path,model_file_path=model_file_path)

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e,sys)