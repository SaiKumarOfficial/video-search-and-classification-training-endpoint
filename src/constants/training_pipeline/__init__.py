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

"""
Model Trainer related constants
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str ="trained_models"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.75
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float= 0.06
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.h5"

SAVED_MODEL_DIR:str = os.path.join("saved_models")
MODEL_FILE_NAME:str = "model.pkl"

OPTIMIZER_TYPE = "Adam"
OPTIMIZER_LR = 0.001
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]

BATCH_SIZE = 4
EPOCHS = 30
ER_STOP_PATIENCE = 20


# """
# Model Evaluation related constants start with MODEL EVALUATION VAR NAME
# """
# MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
# MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
# MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# """
# Model pusher related constanst start with MODEL PUSHER
# """
# MODEL_PUSHER_DIR_NAME: str = "model_pusher"
# MODEL_PUSHER_SAVED_MODEL_DIR: str = SAVED_MODEL_DIR


