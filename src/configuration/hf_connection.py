import sys
from huggingface_hub import HfApi
from src.exception import MyException
from src.logger import logging
from src.constants import HF_TOKEN


class HuggingFaceClient:
    client = None
    def __init__(self) -> None:
        try:
            if HuggingFaceClient.client is None:
                if HF_TOKEN is None:
                    raise Exception("Environment variable 'HF_TOKEN' is not set.")
                HuggingFaceClient.client = HfApi(token=HF_TOKEN)
            self.api = HuggingFaceClient.client
            self.token = HF_TOKEN
            logging.info("Hugging Face connection successful.")

        except Exception as e:
            raise MyException(e, sys) from e
