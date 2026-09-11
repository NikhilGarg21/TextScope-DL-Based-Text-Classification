import os
import sys

import dill
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from pandas import DataFrame, read_csv
from tensorflow.keras.models import load_model  # type: ignore

from src.configuration.hf_connection import HuggingFaceClient
from src.exception import MyException
from src.logger import logging


class HuggingFaceStorage:
    """A class for interacting with Hugging Face Hub.

    Supports:
    - Keras model storage (.h5)
    - Dill object storage (.pkl)
    - Dataset and CSV storage
    """

    def __init__(self):
        try:
            hf_client = HuggingFaceClient()
            self.api = hf_client.api

        except Exception as e:
            raise MyException(e, sys) from e

    def model_path_available(
        self,
        repo_id: str,
        path_in_repo: str,
    ) -> bool:
        """Check whether a file exists in the Hugging Face model repository."""

        try:
            files = self.api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
            )

            return path_in_repo in files

        except RepositoryNotFoundError:
            return False

        except Exception as e:
            raise MyException(e, sys) from e

    def load_keras_model(
        self,
        model_path: str,
        repo_id: str,
    ):
        """Download and load a TensorFlow/Keras model from Hugging Face Hub."""

        try:
            logging.info("Downloading Keras model from Hugging Face Hub")

            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_path,
                repo_type="model",
            )

            model = load_model(local_path)

            logging.info("Keras model loaded successfully")

            return model

        except Exception as e:
            raise MyException(e, sys) from e

    def load_object(
        self,
        object_path: str,
        repo_id: str,
    ) -> object:
        """Download and load a dill serialized object from Hugging Face Hub."""

        try:
            logging.info("Downloading preprocessing object from Hugging Face Hub")

            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=object_path,
                repo_type="model",
            )

            with open(local_path, "rb") as file:
                obj = dill.load(file)

            logging.info("Preprocessing object loaded successfully")

            return obj

        except Exception as e:
            raise MyException(e, sys) from e

    def create_repo(
        self,
        repo_id: str,
        repo_type: str = "model",
    ) -> None:
        """Create a Hugging Face repository if it does not already exist."""

        logging.info("Entered create_repo method")

        try:
            self.api.create_repo(
                repo_id=repo_id,
                exist_ok=True,
                repo_type=repo_type,
            )

            logging.info("Repository created or already exists")

        except Exception as e:
            raise MyException(e, sys) from e

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        repo_id: str,
        repo_type: str = "model",
        remove: bool = False,
    ) -> None:
        """Upload a local file to a Hugging Face repository."""

        logging.info("Entered upload_file method")

        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {repo_id}")

            self.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
            )

            self.api.upload_file(
                path_or_fileobj=from_filename,
                path_in_repo=to_filename,
                repo_id=repo_id,
                repo_type=repo_type,
            )

            logging.info(f"Uploaded {from_filename} successfully")

            if remove:
                os.remove(from_filename)

                logging.info(f"Removed local file: {from_filename}")

            logging.info("Exited upload_file method")

        except Exception as e:
            raise MyException(e, sys) from e

    def file_path_available(
        self,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "dataset",
    ) -> bool:
        """Check whether a file exists in a repository."""

        try:
            files = self.api.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
            )

            return path_in_repo in files

        except RepositoryNotFoundError:
            return False

        except Exception as e:
            raise MyException(e, sys) from e

    def download_file(
        self,
        filename: str,
        repo_id: str,
        repo_type: str = "dataset",
    ) -> str:
        """Download a file from Hugging Face Hub."""

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )

            return local_path

        except Exception as e:
            raise MyException(e, sys) from e

    def upload_df_as_csv(
        self,
        data_frame: DataFrame,
        local_filename: str,
        repo_filename: str,
        repo_id: str,
        repo_type: str = "dataset",
    ) -> None:
        """Upload a DataFrame as a CSV file to Hugging Face Hub."""

        logging.info("Entered upload_df_as_csv method")

        try:
            data_frame.to_csv(
                local_filename,
                index=False,
                header=True,
            )

            self.upload_file(
                from_filename=local_filename,
                to_filename=repo_filename,
                repo_id=repo_id,
                repo_type=repo_type,
            )

            logging.info("Exited upload_df_as_csv method")

        except Exception as e:
            raise MyException(e, sys) from e

    def read_csv(
        self,
        filename: str,
        repo_id: str,
        repo_type: str = "dataset",
    ) -> DataFrame:
        """Read a CSV file from Hugging Face Hub and return it as a DataFrame."""

        logging.info("Entered read_csv method")

        try:
            local_path = self.download_file(
                filename=filename,
                repo_id=repo_id,
                repo_type=repo_type,
            )

            df = read_csv(
                local_path,
                na_values="na",
            )

            logging.info("Exited read_csv method")

            return df

        except Exception as e:
            raise MyException(e, sys) from e
