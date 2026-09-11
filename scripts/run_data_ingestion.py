import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.dvc_util import save_artifact_json
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    pipeline = TrainPipeline()
    artifact = pipeline.start_data_ingestion()
    save_artifact_json("data_ingestion", {
        "train_file_path": artifact.train_file_path,
        "test_file_path": artifact.test_file_path,
    })