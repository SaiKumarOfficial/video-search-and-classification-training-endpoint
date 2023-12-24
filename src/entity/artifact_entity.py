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
    
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy: float
    conf_matrix: int

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    saved_model_path: str
    model_file_path: str