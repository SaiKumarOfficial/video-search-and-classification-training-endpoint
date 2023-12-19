from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_dir: str

@dataclass 
class DataValidationArtifact:
    labels_schema_file_path: str
    valid_data_dir: str
