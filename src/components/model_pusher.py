import sys

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact,
)
from src.entity.config_entity import ModelPusherConfig
from src.entity.hf_estimator import HFModelEstimator


class ModelPusher:

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.hf_estimator = HFModelEstimator(
            repo_id=model_pusher_config.hf_repo_id,
            model_path=model_pusher_config.hf_model_path,
            preprocessing_path=(model_pusher_config.hf_preprocessing_path),
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:

        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading trained model to Hugging Face Hub")

            self.hf_estimator.save_model(
                from_file=(self.model_evaluation_artifact.trained_model_path)
            )

            logging.info("Uploading preprocessing object to Hugging Face Hub")

            self.hf_estimator.save_preprocessor(
                from_file=(self.model_evaluation_artifact.trained_preprocessing_path)
            )

            model_pusher_artifact = ModelPusherArtifact(
                hf_repo_id=self.model_pusher_config.hf_repo_id,
                hf_model_path=(self.model_pusher_config.hf_model_path),
                hf_preprocessing_path=(self.model_pusher_config.hf_preprocessing_path),
            )

            logging.info("Model and preprocessing object uploaded successfully")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e
