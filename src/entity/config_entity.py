import os
from src.constants import *
from dataclasses import dataclass

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    report_page_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    transformed_data_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(transformed_data_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , TRAIN_FILE_NAME.replace('csv' , 'npy'))
    transformed_test_file_path: str = os.path.join(transformed_data_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , TEST_FILE_NAME.replace('csv' , 'npy'))
    transformed_object_file_path: str = os.path.join(transformed_data_dir, PREPROCSSING_OBJECT_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    vocab_size: int = MODEL_TRAINER_VOCAB_SIZE
    max_length: int = MODEL_TRAINER_MAX_LENGTH
    embedding_dim: int = MODEL_TRAINER_EMBEDDING_DIM
    gru_units: int = MODEL_TRAINER_GRU_UNITS
    dense_units: int = MODEL_TRAINER_DENSE_UNITS
    dropout: float = MODEL_TRAINER_DROPOUT
    num_classes: int = MODEL_TRAINER_NUM_CLASSES
    batch_size: int = MODEL_TRAINER_BATCH_SIZE
    epochs: int = MODEL_TRAINER_EPOCHS
    learning_rate: float = MODEL_TRAINER_LEARNING_RATE
    validation_split: float = MODEL_TRAINER_VALIDATION_SPLIT
    early_stopping_patience: int = MODEL_TRAINER_EARLY_STOPPING_PATIENCE
    early_stopping_monitor: str = MODEL_TRAINER_EARLY_STOPPING_MONITOR

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    hf_repo_id: str = HF_REPO_ID
    hf_model_path: str = MODEL_FILE_NAME
    hf_preprocessing_path: str = PREPROCSSING_OBJECT_FILE_NAME

@dataclass
class ModelPusherConfig:
    hf_repo_id: str = HF_REPO_ID
    hf_model_path: str = MODEL_FILE_NAME
    hf_preprocessing_path: str = PREPROCSSING_OBJECT_FILE_NAME
    