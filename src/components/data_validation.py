import json
import sys
import os
import pandas as pd
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            return MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of the text columns and the label column

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_text_columns = []
            for column in self._schema_config["text_columns"]:
                if column not in dataframe_columns:
                    missing_text_columns.append(column)

            if len(missing_text_columns) > 0:
                logging.info(f"Missing text column: {missing_text_columns}")

            label_column = self._schema_config["label_column"]
            missing_label_column = label_column not in dataframe_columns
            if missing_label_column:
                logging.info(f"Missing label column: {label_column}")

            return False if len(missing_text_columns) > 0 or missing_label_column else True
        except Exception as e:
            raise MyException(e, sys) from e

    def is_label_valid(self, df: DataFrame) -> bool:
        """
        Method Name :   is_label_valid
        Description :   Checks the label column has no nulls and every value falls
                        within [0, num_classes - 1]. A bad HF pull or a corrupted
                        split would slip past is_column_exist but show up here.

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            label_column = self._schema_config["label_column"]
            num_classes = self._schema_config["num_classes"]

            null_count = df[label_column].isnull().sum()
            if null_count > 0:
                logging.info(f"Label column '{label_column}' has {null_count} null values")
                return False

            out_of_range = ((df[label_column] < 0) | (df[label_column] > num_classes - 1)).sum()
            if out_of_range > 0:
                logging.info(f"Label column '{label_column}' has {out_of_range} values outside [0, {num_classes - 1}]")
                return False

            return True
        except Exception as e:
            raise MyException(e, sys) from e

    def is_text_valid(self, df: DataFrame) -> bool:
        """
        Method Name :   is_text_valid
        Description :   Checks title/content have no nulls and no empty strings
                        after stripping whitespace.
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            all_valid = True
            for column in self._schema_config["text_columns"]:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    logging.info(f"Text column '{column}' has {null_count} null values")
                    all_valid = False

                empty_count = (df[column].astype(str).str.strip() == "").sum()
                if empty_count > 0:
                    logging.info(f"Text column '{column}' has {empty_count} empty values")
                    all_valid = False

            return all_valid
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path),
            )

            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All required columns present in training dataframe: {status}")

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Text or label columns are missing in training dataframe. "
            else:
                logging.info(f"All text/label columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Text or label columns are missing in test dataframe. "
            else:
                logging.info(f"All text/label columns present in testing dataframe: {status}")


            status = self.is_label_valid(df=train_df)
            if not status:
                validation_error_msg += f"Label column invalid in training dataframe. "
            else:
                logging.info(f"Label column valid in training dataframe: {status}")

            status = self.is_label_valid(df=test_df)
            if not status:
                validation_error_msg += f"Label column invalid in test dataframe. "
            else:
                logging.info(f"Label column valid in test dataframe: {status}")


            status = self.is_text_valid(df=train_df)
            if not status:
                validation_error_msg += f"Text columns have null/empty values in training dataframe. "
            else:
                logging.info(f"Text columns valid in training dataframe: {status}")

            status = self.is_text_valid(df=test_df)
            if not status:
                validation_error_msg += f"Text columns have null/empty values in test dataframe. "
            else:
                logging.info(f"Text columns valid in testing dataframe: {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                report_file_path=self.data_validation_config.report_page_file_path,
            )

            report_dir = os.path.dirname(self.data_validation_config.report_page_file_path)
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip(),
            }

            with open(self.data_validation_config.report_page_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e