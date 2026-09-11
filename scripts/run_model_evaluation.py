import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_util import save_artifact_json, load_artifact_json
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    ingestion_dict = load_artifact_json("data_ingestion")
    transformation_dict = load_artifact_json("data_transformation")
    trainer_dict = load_artifact_json("model_trainer")

    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)
    data_transformation_artifact = DataTransformationArtifact(**transformation_dict)

    metric_artifact = ClassificationMetricArtifact(**trainer_dict["metric_artifact"])
    trainer_dict["metric_artifact"] = metric_artifact
    model_trainer_artifact = ModelTrainerArtifact(**trainer_dict)

    pipeline = TrainPipeline()
    artifact = pipeline.start_model_evaluation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_transformation_artifact=data_transformation_artifact,
        model_trainer_artifact=model_trainer_artifact,
    )

    save_artifact_json("model_evaluation", {
        "is_model_accepted": artifact.is_model_accepted,
        "changed_accuracy": artifact.changed_accuracy,
        "trained_model_path": artifact.trained_model_path,
        "trained_preprocessing_path": artifact.trained_preprocessing_path,
    })