import sys
import pandas as pd
from datasets import load_dataset
from src.configuration.hf_connection import HuggingFaceClient
from src.constants import HF_USERNAME, HF_REPO_NAME
from src.exception import MyException


class TextScopeData:
    """
    A class to export Hugging Face dataset records
    as pandas DataFrames.
    """

    def __init__(self) -> None:
        """
        Initializes the Hugging Face client connection.
        """
        try:
            self.hf_client = HuggingFaceClient()

        except Exception as e:
            raise MyException(e, sys)

    def export_train_test_as_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Exports train and test datasets from Hugging Face
        as pandas DataFrames.
        """

        try:
            dataset_name = f"{HF_USERNAME}/{HF_REPO_NAME}"
            print("Fetching train and test data from Hugging Face")

            dataset = load_dataset(dataset_name, token=self.hf_client.token)

            train_df = dataset["train"].to_pandas()
            test_df = dataset["test"].to_pandas()

            print(f"Train data fetched with shape: {train_df.shape}")
            print(f"Test data fetched with shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            raise MyException(e, sys)
