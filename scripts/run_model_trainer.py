import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.entity.artifact_entity import DataTransformationArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    with open("artifact/dvc_meta/data_transformation.json") as f:
        transformation_dict = json.load(f)
    data_transformation_artifact = DataTransformationArtifact(**transformation_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
    artifact_dict = {
        "trained_model_file_path": artifact.trained_model_file_path,
        "metric_artifact": {
            "f1_score": artifact.metric_artifact.f1_score,
            "precision_score": artifact.metric_artifact.precision_score,
            "recall_score": artifact.metric_artifact.recall_score,
            "accuracy": artifact.metric_artifact.accuracy,
        },
    }
    os.makedirs("artifact/dvc_meta", exist_ok=True)
    with open("artifact/dvc_meta/model_trainer.json", "w") as f:
        json.dump(artifact_dict, f, indent=4)