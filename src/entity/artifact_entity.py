from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the information related to data ingestion process.
    """
    train_file_path: str
    test_file_path: str

