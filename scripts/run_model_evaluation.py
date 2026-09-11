import sys
import os
import json

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
    DataTransformationArtifact,
)
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":

    with open("artifact/dvc_meta/data_ingestion.json") as f:
        ingestion_dict = json.load(f)

    with open("artifact/dvc_meta/model_trainer.json") as f:
        trainer_dict = json.load(f)

    with open("artifact/dvc_meta/data_transformation.json") as f:
        transformation_dict = json.load(f)

    data_ingestion_artifact = DataIngestionArtifact(**ingestion_dict)

    data_transformation_artifact = DataTransformationArtifact(**transformation_dict)

    metric_dict = trainer_dict["metric_artifact"]

    metric_artifact = ClassificationMetricArtifact(**metric_dict)

    trainer_dict["metric_artifact"] = metric_artifact

    model_trainer_artifact = ModelTrainerArtifact(**trainer_dict)

    pipeline = TrainPipeline()

    artifact = pipeline.start_model_evaluation(
        data_ingestion_artifact=data_ingestion_artifact,
        model_trainer_artifact=model_trainer_artifact,
        data_transformation_artifact=data_transformation_artifact,
    )

    artifact_dict = {
        "is_model_accepted": artifact.is_model_accepted,
        "changed_accuracy": artifact.changed_accuracy,
        "trained_model_path": artifact.trained_model_path,
        "trained_preprocessing_path": (artifact.trained_preprocessing_path),
    }

    os.makedirs(
        "artifact/dvc_meta",
        exist_ok=True,
    )

    with open(
        "artifact/dvc_meta/model_evaluation.json",
        "w",
    ) as f:
        json.dump(
            artifact_dict,
            f,
            indent=4,
        )
