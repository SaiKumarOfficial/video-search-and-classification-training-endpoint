from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_data_dir: str
    testing_data_dir: str
    