import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils.TextTokenizer import TextTokenizerTransformer
from src.constants import MAX_LENGTH, MAX_WORDS, TEXT_COLUMN , SCHEMA_FILE_PATH , TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def shuffle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffles the DataFrame to randomize the order of samples.
        """
        try:
            logging.info("Shuffling dataset")
            shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info(f"Dataset shuffled successfully. Shape: {shuffled_df.shape}")
            return shuffled_df
        except Exception as e:
            raise MyException(e, sys) from e

    def _combine_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines the title and content columns into a single text column
        while preserving the target label column.
        """
        try:
            logging.info("Combining title and content into a single text column")
            new_df = pd.DataFrame()
            new_df[TEXT_COLUMN] = (
                df["title"].astype(str) + " " + df["content"].astype(str)
            )
            new_df[TARGET_COLUMN] = df[TARGET_COLUMN]
            logging.info(f"Text column created successfully. Shape: {new_df.shape}")
            return new_df

        except Exception as e:
            raise MyException(e, sys) from e

    def _tokenise_and_pad_text(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple:
        """
        Tokenizes and pads train and test text data,
        combines transformed features with labels,
        and returns the preprocessing pipeline.
        """
        try:
            logging.info("Initializing text tokenization and padding pipeline")
            texttokenizer = TextTokenizerTransformer(
                max_length=MAX_LENGTH, max_words=MAX_WORDS
            )

            final_pipeline = Pipeline(steps=[("text_preprocessor", texttokenizer)])
            logging.info("Text preprocessing pipeline created")

            train_labels = np.array(train_df[TARGET_COLUMN])
            test_labels = np.array(test_df[TARGET_COLUMN])

            logging.info("Separating train and test labels")

            train_text_arr = final_pipeline.fit_transform(train_df[TEXT_COLUMN])
            logging.info(
                f"Training text transformed with shape: {train_text_arr.shape}"
            )

            test_text_arr = final_pipeline.transform(test_df[TEXT_COLUMN])
            logging.info(f"Testing text transformed with shape: {test_text_arr.shape}")

            train_arr = np.c_[train_text_arr, train_labels]
            test_arr = np.c_[test_text_arr, test_labels]

            logging.info(f"Final training array shape: {train_arr.shape}")
            logging.info(f"Final testing array shape: {test_arr.shape}")

            tokenizer = final_pipeline.named_steps["text_preprocessor"].tokenizer_

            num_classes = len(np.unique(train_labels))
            total_vocab_size = len(tokenizer.word_index) + 1
            vocab_size = min(MAX_WORDS, total_vocab_size)

            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Vocabulary size used: {vocab_size}")

            logging.info("Text tokenization and padding completed successfully")
            return (final_pipeline, train_arr, test_arr)

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path : str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(
                file_path=self.data_ingestion_artifact.train_file_path
            )
            test_df = self.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )
            logging.info("Train-Test data loaded")

            train_df = self._combine_text_column(train_df)
            test_df = self._combine_text_column(test_df)
            logging.info("Train and test text columns combined successfully")

            train_df = self.shuffle_data(train_df)
            test_df = self.shuffle_data(test_df)
            logging.info("Train and test datasets shuffled successfully")

            preprocessor , train_arr , test_arr = self._tokenise_and_pad_text(train_df=train_df , test_df=test_df)

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            logging.info("Saving transformation object and transformed files.")
            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformation_object_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
