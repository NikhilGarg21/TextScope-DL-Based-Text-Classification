from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the information related to data ingestion process.
    """
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformation_object_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score : float
    precision_score : float
    recall_score : float
    accuracy : float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str
    metric_artifact : ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    trained_model_path: str
    trained_preprocessing_path: str

@dataclass
class ModelPusherArtifact:
    hf_repo_id: str
    hf_model_path: str
    hf_preprocessing_path: str

    