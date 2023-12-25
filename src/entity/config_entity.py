from src.constants import training_pipeline
from src.entity.artifact_entity import DataValidationArtifact
from datetime import datetime
from from_root import from_root
import os


    



class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%y_%H_%M_%S")
        self.pipelinename: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,timestamp)
        self.timestamp: str = timestamp
    

class DataIngestionConfig:
    def __init__(self ,training_pipeline_config:TrainingPipelineConfig):
        
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                            training_pipeline.DATA_INGESTION_DIR_NAME)
                                   
        # self.feature_store_dir: str = os.path.join(self.data_ingestion_dir,
        #                     training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)

        self.raw_data_dir: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_RAW_DIR)

        # self.training_data_dir: str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INGESTION_TRAIN_DIR)

        # self.testing_data_dir: str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INEGSTION_TEST_DIR)

        # self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        # self.seed_value: int = training_pipeline.DATA_INGESTION_SEED_VALUE


class DataValidationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                     training_pipeline.DATA_VALIDATION_DIR_NAME)
        
        self.labels_schema_file_path:str = os.path.join(self.data_validation_dir, 
                                                    training_pipeline.LABELS_SCHEMA_FILE_DIR,
                                                    training_pipeline.LABELS_SCHEMA_FILE_NAME)
        
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, 
                                                training_pipeline.DATA_VALIDATION_VALID_DATA_DIR)

        self.minimum_required_videos: int = training_pipeline.MINIMUM_NO_OF_VIDEOS

class DataPreparationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.data_preparation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                      training_pipeline.DATA_PREPARATION_DIR_NAME)
        
        self.features_train_file_path: str = os.path.join(self.data_preparation_dir,
                                                         training_pipeline.DATA_PREPARATION_SPLITTED_DATA_DIR,
                                                         training_pipeline.TRAIN_FILE_PATH,
                                                         training_pipeline.FEATURES_TRAIN_FILE_NAME)
        self.features_test_file_path: str = os.path.join(self.data_preparation_dir,
                                                        training_pipeline.DATA_PREPARATION_SPLITTED_DATA_DIR,
                                                        training_pipeline.TEST_FILE_PATH,
                                                        training_pipeline.FEATURES_TEST_FILE_NAME)
        self.labels_train_file_path: str = os.path.join(self.data_preparation_dir,
                                                        training_pipeline.DATA_PREPARATION_SPLITTED_DATA_DIR,
                                                        training_pipeline.TRAIN_FILE_PATH,
                                                        training_pipeline.LABELS_TRAIN_FILE_NAME)
        self.labels_test_file_path: str = os.path.join(self.data_preparation_dir,
                                                        training_pipeline.DATA_PREPARATION_SPLITTED_DATA_DIR,
                                                        training_pipeline.TEST_FILE_PATH,
                                                        training_pipeline.LABELS_TEST_FILE_NAME)
        self.train_test_split_ratio: float = training_pipeline.DATA_PREPARATION_TRAIN_TEST_SPLIT_RATIO
        
        self.seed_value: int = training_pipeline.DATA_PREPARATION_SEED_VALUE

        self.shuffle: bool = training_pipeline.DATA_PREPARATION_SHUFFLE

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.model_trainer_dir: str = os.path.join( training_pipeline_config.artifact_dir,
                                        training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_dir: str = os.path.join(self.model_trainer_dir,
                                            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR)
        self.model_file_path: str = os.path.join(self.trained_model_dir,
                                             training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuracy: str = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        
class ModelEvaluationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, 
            training_pipeline.MODEL_EVALUATION_DIR_NAME)
        self.report_file_path = os.path.join(self.model_evaluation_dir,training_pipeline.MODEL_EVALUATION_REPORT_NAME)
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path = os.path.join(self.model_pusher_dir,training_pipeline.MODEL_FILE_NAME)

        timestamp = round(datetime.now().timestamp())
        self.saved_model_path=os.path.join(
            training_pipeline.SAVED_MODEL_DIR,
            f"{timestamp}",
            training_pipeline.MODEL_FILE_NAME)   
        
class VideoFolderConfig:
    def __init__(self,data_validataion_artifact: DataValidationArtifact):
        self.ROOT_DIR = data_validataion_artifact.valid_data_dir
        self.IMAGE_SIZE = training_pipeline.IMAGE_HEIGHT
        self.LABEL_MAP = {}
        self.BUCKET: str = "isro-documentary-videos"
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/videos/{1}/{2}"

    def get_video_folder_config(self):
        return self.__dict__


# class EmbeddingsConfig:
#     def __init__(self, model_pusher_config: ModelPusherConfig):
#         self.MODEL_STORE_PATH = model_pusher_config.saved_model_path

#     def get_embeddings_config(self):
#         return self.__dict__


class AnnoyConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.embeddings_store_dir = os.path.join(training_pipeline_config.artifact_dir,
                                                training_pipeline.EMBEDDINGS_STORE_DIR_NAME)
        self.embeddings_store_path = os.path.join(self.embeddings_store_dir,
                                                  training_pipeline.EMBEDDINGS_STORE_FILE_PATH)

    def get_annoy_config(self):
        return self.__dict__
    
