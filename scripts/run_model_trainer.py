# run_model_trainer.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_util import save_artifact_json, load_artifact_json
from src.entity.artifact_entity import DataTransformationArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    transformation_dict = load_artifact_json("data_transformation")
    data_transformation_artifact = DataTransformationArtifact(**transformation_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

    save_artifact_json("model_trainer", {
        "trained_model_file_path": artifact.trained_model_file_path,
        "metric_artifact": {
            "f1_score": artifact.metric_artifact.f1_score,
            "precision_score": artifact.metric_artifact.precision_score,
            "recall_score": artifact.metric_artifact.recall_score,
            "accuracy": artifact.metric_artifact.accuracy,
        },
    })