# TextScope: Deep Learning Based Text Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple.svg)](https://dvc.org/)
[![Hugging%20Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Storage-yellow.svg)](https://huggingface.co/)

An end-to-end **MLOps-based Deep Learning Text Classification system** that classifies text into **14 different categories**. The project implements a complete machine learning pipeline including data ingestion, validation, transformation, model training, evaluation, experiment reproducibility using DVC, and model storage using Hugging Face Hub.

---

## 🚀 Project Overview

TextScope is designed as a production-oriented machine learning project rather than just a model training notebook.

The complete pipeline automates:

* 📥 Data Ingestion
* ✅ Data Validation
* 🔄 Data Transformation
* 🧠 Deep Learning Model Training
* 📊 Model Evaluation
* 🤗 Model Deployment to Hugging Face Hub
* 🔁 Pipeline Reproducibility using DVC

The project follows a modular and scalable architecture suitable for real-world MLOps workflows.

---

## 🏗️ Project Architecture

```text
                        +----------------+
                        | Data Ingestion |
                        +----------------+
                                 |
                                 |
                        +----------------+
                        | Data Validation|
                        +----------------+
                                 |
                                 |
                    +------------------------+
                    | Data Transformation    |
                    | Tokenization + Padding |
                    +------------------------+
                                 |
                                 |
                        +----------------+
                        | Model Trainer  |
                        +----------------+
                                 |
                                 |
                     +----------------------+
                     | Model Evaluation     |
                     +----------------------+
                                 |
                                 |
                        +----------------+
                        | Model Pusher   |
                        | Hugging Face   |
                        +----------------+
```

---

# 🔄 DVC Pipeline

The project uses **DVC (Data Version Control)** to manage and reproduce the complete machine learning pipeline.

```text
         +----------------+
         | data_ingestion |
         +----------------+
                 |
                 |
         +-----------------+
         | data_validation |
         +-----------------+
                 |
                 |
      +---------------------+
      | data_transformation |
      +---------------------+
                 |
                 |
          +---------------+
          | model_trainer |
          +---------------+
                 |
                 |
        +------------------+
        | model_evaluation |
        +------------------+
                 |
                 |
            +--------------+
            | model_pusher |
            +--------------+
```

Run the complete pipeline using:

```bash
dvc repro
```

DVC automatically detects which stages have changed and executes only the required stages.

---

# 📂 Project Structure

```text
TextScope-Deep-Learning-Based-Text-Classification/
│
├── config/
│   └── schema.yaml
│
├── scripts/
│   ├── run_data_ingestion.py
│   ├── run_data_validation.py
│   ├── run_data_transformation.py
│   ├── run_model_trainer.py
│   ├── run_model_evaluation.py
│   └── run_model_pusher.py
│
├── src/
│   │
│   ├── cloud_storage/
│   │   └── hf_storage.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   └── hf_connection.py
│   │
│   ├── entity/
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   └── hf_estimator.py
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── utils/
│   │   ├── TextTokenizer.py
│   │   └── main_utils.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── constants.py
│
├── artifact/
│
├── dvc.yaml
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The project uses the **DBPedia 14 dataset** for multi-class text classification.

The dataset contains text samples belonging to **14 different categories**.

Each record contains:

* `title`
* `content`
* `label`

During the data transformation stage, the `title` and `content` columns are combined into a single text feature.

```text
text = title + content
```

The final dataset is used for multi-class text classification.

---

# 🔄 Data Pipeline

## 1️⃣ Data Ingestion

The Data Ingestion component is responsible for:

* Fetching the dataset
* Creating train and test datasets
* Storing the datasets locally

Output:

```text
artifact/
└── data_ingestion/
    └── ingested/
        ├── train.csv
        └── test.csv
```

---

## 2️⃣ Data Validation

The Data Validation component verifies the dataset structure.

Validation includes checking:

* Required columns
* Dataset schema
* Data availability

A validation report is generated:

```text
artifact/
└── data_validation/
    └── report.yaml
```

---

## 3️⃣ Data Transformation

The Data Transformation component performs the following operations:

### Text Combination

The dataset originally contains:

```text
title
content
```

These columns are combined:

```text
title + content → text
```

### Data Shuffling

The training and testing datasets are shuffled using a fixed random state.

### Tokenization

The text is converted into integer sequences using a tokenizer.

Configuration:

```text
Maximum Vocabulary Size: 30,000
Maximum Sequence Length: 100
```

### Padding

Sequences are padded to ensure a fixed input length.

The transformed arrays contain:

```text
[tokenized_text_features + label]
```

Outputs:

```text
artifact/
└── data_transformation/
    │
    ├── transformed/
    │   ├── train.npy
    │   └── test.npy
    │
    └── preprocessing.pkl
```

The preprocessing object is saved so the same tokenizer and preprocessing pipeline can be used during model evaluation and inference.

---

# 🧠 Model Training

The project uses a Deep Learning architecture for multi-class text classification.

The model is trained using TensorFlow/Keras.

### Training Configuration

| Parameter               | Value  |
| ----------------------- | ------ |
| Maximum Vocabulary      | 30,000 |
| Maximum Sequence Length | 100    |
| Embedding Dimension     | 128    |
| GRU Units               | 128    |
| Dense Units             | 64     |
| Dropout                 | 0.3    |
| Number of Classes       | 14     |
| Batch Size              | 256    |
| Epochs                  | 10     |
| Learning Rate           | 0.001  |
| Validation Split        | 0.2    |

---

## 🛑 Early Stopping

Early stopping is used to prevent unnecessary training.

Configuration:

```text
Monitor: val_loss
Patience: 2
```

The best model is restored automatically based on validation loss.

---

## 📏 Model Metrics

The model performance is evaluated using classification metrics.

The project tracks metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

The primary comparison metric used during model evaluation is:

```text
Macro F1 Score
```

Macro F1 Score is useful for evaluating performance across multiple classes.

---

# 📊 Model Evaluation

The Model Evaluation component compares the newly trained model with the existing production model stored in Hugging Face Hub.

The workflow is:

```text
New Trained Model
        |
        |
        ▼
Calculate Macro F1 Score
        |
        |
        ▼
Production Model Available?
      /         \
    Yes          No
     |            |
     ▼            ▼
Compare Scores  Accept Model
     |
     ▼
Threshold Check
     |
     ▼
Accept / Reject Model
```

The configured score improvement threshold is:

```text
0.005
```

A model is accepted when its performance improvement satisfies the configured threshold.

---

# 🤗 Hugging Face Integration

The project uses Hugging Face Hub for model storage and retrieval.

The following artifacts can be stored:

* Trained Deep Learning Model
* Preprocessing Object

This allows the project to maintain a production model outside the local machine.

The Hugging Face integration is implemented through:

```text
HuggingFaceClient
        |
        ▼
HuggingFaceStorage
        |
        ▼
HFModelEstimator
        |
        ▼
Hugging Face Hub
```

---

# 🚀 Model Pusher

The Model Pusher component uploads the accepted model to Hugging Face Hub.

Pipeline:

```text
Model Evaluation
        |
        ▼
Model Accepted?
        |
        ▼
Model Pusher
        |
        ▼
Hugging Face Hub
```

The model and associated preprocessing object can then be retrieved for future evaluation or inference.

---

# 🔐 Environment Configuration

The project uses a Hugging Face access token for authentication with Hugging Face Hub.

The token is accessed through an environment variable:

```python
HF_TOKEN = os.getenv("HF_TOKEN")
```

## Required Environment Variable

```text
HF_TOKEN=your_huggingface_access_token
```

> **Note:** This project does not require a `.env` file. The `HF_TOKEN` can be configured directly as a system environment variable or as a deployment secret.

---

## Windows PowerShell

For temporary use:

```powershell
$env:HF_TOKEN="your_huggingface_access_token"
```

To verify:

```powershell
echo $env:HF_TOKEN
```

---

## Deployment

When deploying the application or pipeline, configure:

| Variable   | Description                                       |
| ---------- | ------------------------------------------------- |
| `HF_TOKEN` | Hugging Face access token used for authentication |

The token should be configured using the deployment platform's **Environment Variables** or **Secrets** section.

⚠️ **Never hardcode your Hugging Face token inside the source code.**

⚠️ **Never commit your token to GitHub.**

---

# 🔧 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/NikhilGarg21/TextScope-Deep-Learning-Based-Text-Classification.git
```

Move into the project directory:

```bash
cd TextScope-Deep-Learning-Based-Text-Classification
```

---

## 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

### Windows

```powershell
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Pipeline

The entire machine learning pipeline can be reproduced using DVC.

```bash
dvc repro
```

DVC executes the following stages:

```text
data_ingestion
        ↓
data_validation
        ↓
data_transformation
        ↓
model_trainer
        ↓
model_evaluation
        ↓
model_pusher
```

DVC detects unchanged stages and skips unnecessary computations.

For example:

```text
Stage 'model_trainer' didn't change, skipping
```

This improves reproducibility and avoids unnecessary model retraining.

---

# 🔁 Running Individual Components

Each component can also be executed independently.

## Data Ingestion

```bash
python scripts/run_data_ingestion.py
```

## Data Validation

```bash
python scripts/run_data_validation.py
```

## Data Transformation

```bash
python scripts/run_data_transformation.py
```

## Model Training

```bash
python scripts/run_model_trainer.py
```

## Model Evaluation

```bash
python scripts/run_model_evaluation.py
```

## Model Pusher

```bash
python scripts/run_model_pusher.py
```

---

# 📦 DVC Pipeline Stages

The pipeline consists of six stages.

```text
data_ingestion
        ↓
data_validation
        ↓
data_transformation
        ↓
model_trainer
        ↓
model_evaluation
        ↓
model_pusher
```

Each stage is defined in:

```text
dvc.yaml
```

DVC tracks:

* Dependencies
* Pipeline stages
* Generated artifacts
* Pipeline changes

---

# 🧩 MLOps Concepts Implemented

This project demonstrates several important MLOps concepts.

### ✔ Modular Architecture

Each machine learning stage is implemented as an independent component.

### ✔ Configuration Management

Configuration entities are used to manage pipeline paths and settings.

### ✔ Artifact Management

Each pipeline stage generates an artifact object.

Examples:

```text
DataIngestionArtifact
DataValidationArtifact
DataTransformationArtifact
ModelTrainerArtifact
ModelEvaluationArtifact
ModelPusherArtifact
```

### ✔ Data Version Control

DVC is used to reproduce and manage the machine learning pipeline.

### ✔ Experiment Reproducibility

Pipeline stages can be reproduced using:

```bash
dvc repro
```

### ✔ Model Evaluation

New models are compared with the existing production model.

### ✔ Model Registry / Storage

Hugging Face Hub is used to store trained models and preprocessing artifacts.

### ✔ Logging

The project implements logging to track pipeline execution.

### ✔ Custom Exception Handling

Custom exceptions provide detailed error information during pipeline failures.

---

# 🛠️ Technology Stack

| Category               | Technologies                      |
| ---------------------- | --------------------------------- |
| Programming Language   | Python                            |
| Deep Learning          | TensorFlow / Keras                |
| Data Processing        | Pandas, NumPy                     |
| Machine Learning       | Scikit-learn                      |
| Text Processing        | Tokenization and Sequence Padding |
| Pipeline Management    | DVC                               |
| Model Storage          | Hugging Face Hub                  |
| Configuration          | YAML                              |
| Environment Management | Python Virtual Environment        |

---

# 📁 Generated Artifacts

The pipeline generates artifacts during execution.

```text
artifact/
│
├── data_ingestion/
│   └── ingested/
│       ├── train.csv
│       └── test.csv
│
├── data_validation/
│   └── report.yaml
│
├── data_transformation/
│   ├── transformed/
│   │   ├── train.npy
│   │   └── test.npy
│   │
│   └── preprocessing.pkl
│
├── model_trainer/
│   └── trained_model/
│       └── model.h5
│
└── dvc_meta/
```
---

# 🎯 Key Features

* ✅ End-to-End Deep Learning Pipeline
* ✅ Multi-Class Text Classification
* ✅ 14 Text Categories
* ✅ Automated Data Ingestion
* ✅ Dataset Validation
* ✅ Text Preprocessing
* ✅ Tokenization and Padding
* ✅ Deep Learning Model Training
* ✅ Early Stopping
* ✅ Model Performance Tracking
* ✅ Macro F1 Score Evaluation
* ✅ Production Model Comparison
* ✅ Hugging Face Model Storage
* ✅ DVC Pipeline Reproducibility
* ✅ Modular Project Architecture
* ✅ Custom Logging
* ✅ Custom Exception Handling

---

# 🔮 Future Improvements

Potential improvements for the project include:

* [ ] FastAPI Prediction API
* [ ] Docker Containerization
* [ ] CI/CD Pipeline
* [ ] Automated Testing
* [ ] Model Monitoring
* [ ] Drift Detection
* [ ] MLflow Experiment Tracking
* [ ] Cloud Deployment
* [ ] Kubernetes Deployment
* [ ] Automated Model Retraining

---

# 📈 Learning Outcomes

Through this project, the following concepts are implemented and demonstrated:

* Deep Learning for NLP
* Multi-Class Text Classification
* Text Tokenization
* Sequence Padding
* GRU-based Neural Networks
* Machine Learning Pipeline Design
* MLOps Architecture
* Data Version Control with DVC
* Model Evaluation Strategies
* Model Versioning
* Hugging Face Hub Integration
* Artifact Management
* Configuration Management
* Reproducible Machine Learning Pipelines

---

# 👨‍💻 Author
**Nikhil Garg**
---

