import sys
from typing import Any, Optional

import numpy as np

from src.cloud_storage.hf_storage import HuggingFaceStorage
from src.exception import MyException


class HFModelEstimator:
    """Handles saving, loading, and predicting with the TextScope Keras model

    and preprocessing object.
    """

    def __init__(
        self,
        repo_id: str,
        model_path: str,
        preprocessing_path: str,
    ):
        try:
            self.repo_id = repo_id
            self.model_path = model_path
            self.preprocessing_path = preprocessing_path

            self.hf = HuggingFaceStorage()

            self.loaded_model: Optional[Any] = None
            self.loaded_preprocessor: Optional[Any] = None

        except Exception as e:
            raise MyException(e, sys) from e

    def is_model_present(self) -> bool:
        """Check whether the trained Keras model exists in the Hugging Face

        repository.
        """

        try:
            return self.hf.model_path_available(
                repo_id=self.repo_id,
                path_in_repo=self.model_path,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def is_preprocessor_present(self) -> bool:
        """Check whether the preprocessing object exists in the Hugging Face

        repository.
        """

        try:
            return self.hf.model_path_available(
                repo_id=self.repo_id,
                path_in_repo=self.preprocessing_path,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def is_model_and_preprocessor_present(self) -> bool:
        """Check whether both model and preprocessing objects are available."""

        try:
            return self.is_model_present() and self.is_preprocessor_present()

        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self) -> Any:
        """Load the Keras model from Hugging Face Hub."""

        try:
            return self.hf.load_keras_model(
                model_path=self.model_path,
                repo_id=self.repo_id,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def load_preprocessor(self) -> Any:
        """Load the preprocessing pipeline from Hugging Face Hub."""

        try:
            return self.hf.load_object(
                object_path=self.preprocessing_path,
                repo_id=self.repo_id,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def save_model(
        self,
        from_file: str,
        remove: bool = False,
    ) -> None:
        """Upload Keras model to Hugging Face Hub."""

        try:
            self.hf.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                repo_id=self.repo_id,
                repo_type="model",
                remove=remove,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def save_preprocessor(
        self,
        from_file: str,
        remove: bool = False,
    ) -> None:
        """Upload preprocessing object to Hugging Face Hub."""

        try:
            self.hf.upload_file(
                from_filename=from_file,
                to_filename=self.preprocessing_path,
                repo_id=self.repo_id,
                repo_type="model",
                remove=remove,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, text_data: Any) -> np.ndarray:
        """Transform raw text using the preprocessing pipeline and return

        predicted class labels.
        """

        try:
            if self.loaded_preprocessor is None:
                self.loaded_preprocessor = self.load_preprocessor()

            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            transformed_data = self.loaded_preprocessor.transform(text_data)

            probabilities = self.loaded_model.predict(transformed_data)

            predictions = np.argmax(
                probabilities,
                axis=1,
            )

            return predictions

        except Exception as e:
            raise MyException(e, sys) from e