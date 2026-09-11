import os
from datetime import date

from dotenv import load_dotenv
load_dotenv()

HF_REPO_NAME = "TextScope"
HF_USERNAME = "research07"
HF_TOKEN = os.getenv("HF_TOKEN")

PIPELINE_NAME: str = "TextScope"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME: str = "model.pkl"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join(
    "config",
    "schema.yaml"
)


# Data Ingestion related constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"


# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
MAX_WORDS = 30000
MAX_LENGTH = 100
# Model Trainer related constants

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"


# Application constants

APP_HOST = "0.0.0.0"
APP_PORT = 5000