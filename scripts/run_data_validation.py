import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.entity.artifact_entity import DataIngestionArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    with open("artifact/dvc_meta/data_ingestion.json") as f:
        ingestion_dict = json.load(f)
    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

    artifact_dict = {
        "validation_status": artifact.validation_status,
        "message": artifact.message,
        "report_file_path": artifact.report_file_path,
    }

    os.makedirs("artifact/dvc_meta", exist_ok=True)
    with open("artifact/dvc_meta/data_validation.json", "w") as f:
        json.dump(artifact_dict, f, indent=4)