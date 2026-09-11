import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_util import save_artifact_json, load_artifact_json
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    ingestion_dict = load_artifact_json("data_ingestion")
    validation_dict = load_artifact_json("data_validation")

    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)
    data_validation_artifact = DataValidationArtifact(**validation_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_data_transformation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_artifact=data_validation_artifact,
    )

    save_artifact_json("data_transformation", {
        "transformed_train_file_path": artifact.transformed_train_file_path,
        "transformed_test_file_path": artifact.transformed_test_file_path,
        "transformation_object_path": artifact.transformation_object_path,
    })