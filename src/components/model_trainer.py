import sys
from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input,
    Embedding,
    GRU,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    save_keras_model,
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self,
        train: np.ndarray,
        test: np.ndarray,
    ) -> Tuple[object, ClassificationMetricArtifact]:

        try:
            logging.info("Starting GRU model training")

            x_train = train[:, :-1].astype(np.int32)
            y_train = train[:, -1].astype(np.int32)

            x_test = test[:, :-1].astype(np.int32)
            y_test = test[:, -1].astype(np.int32)

            logging.info(f"Training feature shape: {x_train.shape}")
            logging.info(f"Testing feature shape: {x_test.shape}")
            logging.info(f"Number of classes: {len(np.unique(y_train))}")

            logging.info("Building GRU model")

            gru_model = Sequential(
                [
                    Input(
                        shape=(self.model_trainer_config.max_length,),
                        dtype="int32",
                    ),
                    Embedding(
                        input_dim=self.model_trainer_config.vocab_size,
                        output_dim=self.model_trainer_config.embedding_dim,
                        mask_zero=True,
                    ),
                    GRU(
                        units=self.model_trainer_config.gru_units,
                        dropout=self.model_trainer_config.dropout,
                    ),
                    Dense(
                        units=self.model_trainer_config.dense_units,
                        activation="relu",
                    ),
                    Dropout(self.model_trainer_config.dropout),
                    Dense(
                        units=self.model_trainer_config.num_classes,
                        activation="softmax",
                    ),
                ]
            )

            logging.info("Compiling GRU model")

            optimizer = Adam(
                learning_rate=self.model_trainer_config.learning_rate,
                clipnorm=1.0,
            )

            gru_model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stopping = EarlyStopping(
                monitor=self.model_trainer_config.early_stopping_monitor,
                patience=self.model_trainer_config.early_stopping_patience,
                restore_best_weights=True,
            )

            logging.info("GRU model summary")
            gru_model.summary()

            logging.info("Model training started")

            gru_model.fit(
                x_train,
                y_train,
                validation_split=self.model_trainer_config.validation_split,
                epochs=self.model_trainer_config.epochs,
                batch_size=self.model_trainer_config.batch_size,
                callbacks=[early_stopping],
                shuffle=True,
                verbose=1,
            )

            logging.info("Model training completed successfully")
            logging.info("Predicting on test data")

            y_pred_probs = gru_model.predict(
                x_test,
                batch_size=self.model_trainer_config.batch_size,
                verbose=1,
            )

            y_pred = np.argmax(
                y_pred_probs,
                axis=1,
            )

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            logging.info(f"Test Accuracy: {accuracy}")
            logging.info(f"Test F1 Score: {f1}")
            logging.info(f"Test Precision: {precision}")
            logging.info(f"Test Recall: {recall}")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                accuracy=accuracy,
            )

            return gru_model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:

        logging.info("Entered initiate_model_trainer method of ModelTrainer")

        try:
            print("-" * 100)
            print("Starting Model Trainer Component")

            logging.info("Loading transformed training data")

            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )

            logging.info("Loading transformed testing data")

            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(f"Training array shape: {train_arr.shape}")
            logging.info(f"Testing array shape: {test_arr.shape}")

            trained_model, metric_artifact = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr,
            )

            logging.info("Checking model performance")

            if metric_artifact.accuracy < self.model_trainer_config.expected_accuracy:
                logging.info("Model accuracy is below expected threshold")
                raise Exception(
                    f"Model accuracy {metric_artifact.accuracy} "
                    f"is below expected accuracy {self.model_trainer_config.expected_accuracy}"
                )

            logging.info("Model performance is above expected threshold")
            logging.info("Saving trained Keras model")

            save_keras_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                model=trained_model,
            )

            logging.info("Trained model saved successfully")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
