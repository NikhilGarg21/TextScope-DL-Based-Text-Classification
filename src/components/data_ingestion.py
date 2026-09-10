import os
import sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.exception import MyException
from src.data_access.TextScope_data import TextScopeData

class DataIngestion:
    """
    Data Ingestion class to handle the data ingestion process.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_as_train_test(self) -> None:
        """
        Exports train and test datasets from Hugging Face as pandas DataFrames
        and saves them to the specified file paths.

        Returns:
            None
        """
        try:
            logging.info("Starting data ingestion process.")
            text_scope_data = TextScopeData()
            train_df, test_df = text_scope_data.export_train_test_as_dataframe()

            logging.info(f"Shape of train dataframe: {train_df.shape}")
            logging.info(f"Shape of test dataframe: {test_df.shape}")

            data_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(data_path, exist_ok=True)
            logging.info(
                f"Saving exported data into training file path: {self.data_ingestion_config.training_file_path} and testing file path: {self.data_ingestion_config.testing_file_path}"
            )
            train_df.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process and returns the artifact.

        Returns:
            DataIngestionArtifact: Artifact containing paths to the ingested train and test files.
        """
        try:
            self.export_data_as_train_test()
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)