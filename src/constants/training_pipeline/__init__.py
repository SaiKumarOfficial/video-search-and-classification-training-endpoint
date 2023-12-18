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
DATA_INGESTION_FEATURE_STORE_DIR= "feature_store"
# DATA_INGESTION_INGESTED_DIR= "ingested_videos"
DATA_INGESTION_RAW_DIR = "raw"
DATA_INGESTION_TRAIN_DIR = "splitted/train"
DATA_INEGSTION_TEST_DIR = "splitted/test"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.8
DATA_INGESTION_SEED_VALUE = 47