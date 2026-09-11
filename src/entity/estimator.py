import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.constants import TEXT_COLUMN
from src.exception import MyException
from src.logger import logging


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Fitted tokenizer/padding pipeline
        :param trained_model_object: Trained Keras GRU model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Accepts a dataframe containing the raw text column, tokenizes/pads it
        via preprocessing_object, and returns predicted class labels.
        """
        try:
            logging.info("Starting prediction process.")

            transformed_feature = self.preprocessing_object.transform(dataframe[TEXT_COLUMN])
            logging.info("Using the trained model to get predictions")
            probabilities = self.trained_model_object.predict(transformed_feature)
            predictions = np.argmax(probabilities, axis=1)
            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"