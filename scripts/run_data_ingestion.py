import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    pipeline = TrainPipeline()
    artifact = pipeline.start_data_ingestion()
    artifact_dict = {
        "train_file_path": artifact.train_file_path,
        "test_file_path": artifact.test_file_path,
    }
    os.makedirs("artifact/dvc_meta", exist_ok=True)
    with open("artifact/dvc_meta/data_ingestion.json", "w") as f:
        json.dump(artifact_dict, f, indent=4)