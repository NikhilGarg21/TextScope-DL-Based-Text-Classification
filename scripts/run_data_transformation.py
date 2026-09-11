import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    with open("artifact/dvc_meta/data_ingestion.json") as f:
        ingestion_dict = json.load(f)
    with open("artifact/dvc_meta/data_validation.json") as f:
        validation_dict = json.load(f)

    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)
    data_validation_artifact = DataValidationArtifact(**validation_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_data_transformation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_artifact=data_validation_artifact,
    )

    artifact_dict = {
        "transformed_train_file_path": artifact.transformed_train_file_path,
        "transformed_test_file_path": artifact.transformed_test_file_path,
        "transformation_object_path": artifact.transformation_object_path,
    }

    os.makedirs("artifact/dvc_meta", exist_ok=True)
    with open("artifact/dvc_meta/data_transformation.json", "w") as f:
        json.dump(artifact_dict, f, indent=4)