import os
from datetime import date

from dotenv import load_dotenv
load_dotenv()

HF_REPO_NAME = "TextScope"
HF_USERNAME = "research07"
HF_TOKEN = os.getenv("HF_TOKEN")

HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

PIPELINE_NAME: str = "TextScope"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME: str = "model.h5"
PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

TEXT_COLUMN: str = "text"
TARGET_COLUMN: str = "label"

CURRENT_YEAR = date.today().year

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


SCHEMA_FILE_PATH = os.path.join(
    "config",
    "schema.yaml",
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
MAX_WORDS: int = 30000
MAX_LENGTH: int = 100


# Model Trainer related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.95
MODEL_TRAINER_VOCAB_SIZE: int = 30000
MODEL_TRAINER_MAX_LENGTH: int = 100
MODEL_TRAINER_EMBEDDING_DIM: int = 128
MODEL_TRAINER_GRU_UNITS: int = 128
MODEL_TRAINER_DENSE_UNITS: int = 64
MODEL_TRAINER_DROPOUT: float = 0.3
MODEL_TRAINER_NUM_CLASSES: int = 14
MODEL_TRAINER_BATCH_SIZE: int = 256
MODEL_TRAINER_EPOCHS: int = 10
MODEL_TRAINER_LEARNING_RATE: float = 1e-3
MODEL_TRAINER_VALIDATION_SPLIT: float = 0.2
MODEL_TRAINER_EARLY_STOPPING_PATIENCE: int = 2
MODEL_TRAINER_EARLY_STOPPING_MONITOR: str = "val_loss"


# Model Evaluation related constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.005


# Application constants
APP_HOST: str = "0.0.0.0"
APP_PORT: int = 5000