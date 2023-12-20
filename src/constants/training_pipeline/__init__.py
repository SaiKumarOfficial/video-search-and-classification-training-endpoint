import os
"""
Defining common constant variables for training pipeline

"""

ARTIFACT_DIR = "artifact"
PIPELINE_NAME = "training_pipeline"


SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_FILE_NAME = "preprocessing.pkl"

"""
Data ingestion related constants start with DATA_INGESTION VAR NAME
"""

# DATA_INGESTION_COLLECTION_NAME = ""
DATA_INGESTION_DIR_NAME = "data_ingestion"
# DATA_INGESTION_FEATURE_STORE_DIR= "feature_store"
# DATA_INGESTION_INGESTED_DIR= "ingested_videos"
DATA_INGESTION_RAW_DIR = "raw"
# DATA_INGESTION_TRAIN_DIR = "splitted/train"
# DATA_INEGSTION_TEST_DIR = "splitted/test"

# DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.8
# DATA_INGESTION_SEED_VALUE = 47

"""
Data Validation related constants
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
MINIMUM_NO_OF_VIDEOS: int = 2
LABELS_SCHEMA_FILE_DIR: str = "schema"
LABELS_SCHEMA_FILE_NAME = 'labels_schema.yaml'
DATA_VALIDATION_VALID_DATA_DIR: str = "valid"

SCHEMA_KEY = "LABELS"
VALID_EXTENSIONS = ['.mp4']


"""
Data Preparation related constants
"""

DATA_PREPARATION_DIR_NAME: str = "data_preparation"
DATA_PREPARATION_SPLITTED_DATA_DIR: str = "splitted"
TRAIN_FILE_PATH: str = "train"
TEST_FILE_PATH: str = "test"

FEATURES_TRAIN_FILE_NAME: str = "features_train.npy"
LABELS_TRAIN_FILE_NAME: str = "labels_train.npy"

FEATURES_TEST_FILE_NAME: str = "features_test.npy"
LABELS_TEST_FILE_NAME: str = "labels_test.npy"

IMAGE_WIDTH, IMAGE_HEIGHT = 112,112
SEQUENCE_LENGTH: int = 50

DATA_PREPARATION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_PREPARATION_SEED_VALUE: int  = 47
DATA_PREPARATION_SHUFFLE: bool = True

