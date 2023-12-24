from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact,DataPreparationArtifact
from src.utils.common import ModelResolver
from src.exception import CustomException
from src.utils.common import load_numpy_array_data,load_model,get_classification_score,write_yaml_file
from src.logger import logging
import os,sys
import numpy as np
class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                    data_preparation_artifact: DataPreparationArtifact,
                    model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config=model_eval_config
            self.data_preparation_artifact = data_preparation_artifact
            self.model_trainer_artifact=model_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys)
    def run_steps(self):
        try:
            logging.info("=============Starting Model evaluation phase==========")
            features_test_data = load_numpy_array_data(self.data_preparation_artifact.features_test_file_path)
            labels_test_data = load_numpy_array_data(self.data_preparation_artifact.labels_test_file_path)
            labels_test_data = np.argmax(labels_test_data, axis=1)

            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True
            if not model_resolver.is_model_exist():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=trained_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_model(file_path=latest_model_path)
            train_model = load_model(file_path=trained_model_file_path)

            
            y_trained_pred = train_model.predict(features_test_data)
            y_trained_pred = np.argmax(y_trained_pred, axis=1)
        
            y_latest_pred  = latest_model.predict(features_test_data)
            y_latest_pred = np.argmax(y_latest_pred, axis=1)

            
            trained_metric = get_classification_score(labels_test_data, y_trained_pred)
            latest_metric = get_classification_score(labels_test_data, y_latest_pred)

            improved_accuracy = trained_metric.f1_score-latest_metric.f1_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=trained_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)

            model_eval_report = model_evaluation_artifact.__dict__

            logging.info("Saving the report ")
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info("============= Completed model evaluation phase =============")
            return model_evaluation_artifact


            
        except Exception as e:
            raise CustomException(e,sys)