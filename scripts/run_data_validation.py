import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_util import save_artifact_json, load_artifact_json
from src.entity.artifact_entity import DataIngestionArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    ingestion_dict = load_artifact_json("data_ingestion")
    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

    save_artifact_json("data_validation", {
        "validation_status": artifact.validation_status,
        "message": artifact.message,
        "report_file_path": artifact.report_file_path,
    })