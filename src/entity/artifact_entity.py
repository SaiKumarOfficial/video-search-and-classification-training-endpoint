from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_dir: str

@dataclass 
class DataValidationArtifact:
    labels_schema_file_path: str
    valid_data_dir: str

@dataclass
class DataPreparationArtifact:
    features_train_file_path: str
    labels_train_file_path: str
    features_test_file_path: str
    labels_test_file_path: str
    