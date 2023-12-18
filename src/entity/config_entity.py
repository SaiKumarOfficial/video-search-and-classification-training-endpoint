from src.constants import training_pipeline
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
                                   
        self.feature_store_dir: str = os.path.join(self.data_ingestion_dir,
                            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)

        self.raw_data_dir: str = os.path.join(self.feature_store_dir,training_pipeline.DATA_INGESTION_RAW_DIR)

        self.training_data_dir: str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INGESTION_TRAIN_DIR)

        self.testing_data_dir: str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INEGSTION_TEST_DIR)

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        self.seed_value: int = training_pipeline.DATA_INGESTION_SEED_VALUE


    