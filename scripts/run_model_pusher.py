import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.dvc_util import save_artifact_json, load_artifact_json
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.pipeline.training_pipeline import TrainPipeline
from src.logger import logging

if __name__ == "__main__":
    evaluation_dict = load_artifact_json("model_evaluation")
    model_evaluation_artifact = ModelEvaluationArtifact(**evaluation_dict)

    if not model_evaluation_artifact.is_model_accepted:
        logging.info("Model rejected during evaluation stage. Skipping push to Hugging Face Hub.")
        print("Model rejected during evaluation. Skipping model pusher stage.")
        
        save_artifact_json("model_pusher", {
            "status": "skipped",
            "reason": "Model was rejected during evaluation stage."
        })
        sys.exit(0)

    logging.info("Model accepted during evaluation stage. Proceeding to push to Hugging Face Hub.")
    pipeline = TrainPipeline()
    artifact = pipeline.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)

    save_artifact_json("model_pusher", {
        "status": "pushed",
        "hf_repo_id": artifact.hf_repo_id,
        "hf_model_path": artifact.hf_model_path,
        "hf_preprocessing_path": artifact.hf_preprocessing_path,
    })