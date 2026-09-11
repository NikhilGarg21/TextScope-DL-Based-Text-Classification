import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.entity.hf_estimator import HFModelEstimator
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact : DataTransformationArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[HFModelEstimator]:
        """
        Get the production model and preprocessing object
        from Hugging Face Hub.
        """
        try:
            hf_estimator = HFModelEstimator(
                repo_id=self.model_eval_config.hf_repo_id,
                model_path=self.model_eval_config.hf_model_path,
                preprocessing_path=self.model_eval_config.hf_preprocessing_path,
            )

            if hf_estimator.is_model_and_preprocessor_present():
                logging.info(
                    "Production model and preprocessing object "
                    "found in Hugging Face Hub."
                )
                return hf_estimator

            logging.info(
                "Production model or preprocessing object "
                "not found in Hugging Face Hub."
            )

            return None

        except Exception as e:
            raise MyException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compare the newly trained model with the
        production model stored in Hugging Face Hub.
        """
        try:
            logging.info("Loading test dataset")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Test data loaded with shape: {test_df.shape}")
            logging.info(
                "Combining title and content into text column "
                "for production model evaluation."
            )

            x = test_df["title"].astype(str) + " " + test_df["content"].astype(str)
            y = test_df[TARGET_COLUMN].astype(np.int32)
            trained_model_f1_score = (
                self.model_trainer_artifact.metric_artifact.f1_score
            )

            logging.info(f"New trained model F1 Score: " f"{trained_model_f1_score}")
            best_model_f1_score = None

            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Computing F1 Score for production model")
                y_pred = best_model.predict(x)

                best_model_f1_score = f1_score(
                    y,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )

                logging.info(f"Production Model F1 Score: " f"{best_model_f1_score}")
                logging.info(
                    f"New Trained Model F1 Score: " f"{trained_model_f1_score}"
                )

            tmp_best_model_score = (
                0 if best_model_f1_score is None else best_model_f1_score
            )

            difference = trained_model_f1_score - tmp_best_model_score

            is_model_accepted = (
                difference >= self.model_eval_config.changed_threshold_score
            )

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )

            logging.info(f"Model evaluation result: {result}")

            return result

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:
        """
        Initiates the model evaluation component.
        """
        logging.info("Entered initiate_model_evaluation " "method of ModelEvaluation")

        try:
            print("-" * 100)

            logging.info("Starting Model Evaluation Component")

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_preprocessing_path=self.data_transformation_artifact.transformation_object_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: " f"{model_evaluation_artifact}")

            logging.info(
                "Exited initiate_model_evaluation " "method of ModelEvaluation"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
