import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


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
