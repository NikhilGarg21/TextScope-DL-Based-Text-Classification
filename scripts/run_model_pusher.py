import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity.artifact_entity import ModelEvaluationArtifact
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":

    with open("artifact/dvc_meta/model_evaluation.json") as f:
        evaluation_dict = json.load(f)

    model_evaluation_artifact = ModelEvaluationArtifact(**evaluation_dict)

    pipeline = TrainPipeline()

    artifact = pipeline.start_model_pusher(
        model_evaluation_artifact=model_evaluation_artifact,
    )

    artifact_dict = {
        "hf_repo_id": artifact.hf_repo_id,
        "hf_model_path": artifact.hf_model_path,
        "hf_preprocessing_path": artifact.hf_preprocessing_path,
    }

    os.makedirs("artifact/dvc_meta", exist_ok=True)

    with open(
        "artifact/dvc_meta/model_pusher.json",
        "w",
    ) as f:
        json.dump(
            artifact_dict,
            f,
            indent=4,
        )
