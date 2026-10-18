# Multimodal AI Medical Triage System

AI & Big Data Program — UIR  
Integrated Project 2025/2026

> **Academic project.** This system is a demonstration of an AI-assisted triage workflow. It is not a medical device and must not be used for real clinical decisions.

## Project Overview

This project implements an AI-based medical triage dashboard for chest X-ray analysis. It combines:

- a PyTorch CNN for image classification
- a rule-based clinical guidance module
- a CrewAI pipeline of three LLM agents
- a medical reporting layer
- a human-in-the-loop validation step
- a Streamlit interface for practical use

The system classifies chest X-rays into:

- NORMAL
- PNEUMONIA

It then generates a structured triage report that includes:

- patient ID
- timestamp
- radiology finding
- triage tier
- recommended action
- guideline citation

## Team

- Lina Harti
- Amine Lassri
- Aymane Aboulhouda
- Maryam Boulminate

## Architecture

```text
Chest X-ray ──► CNN (PyTorch) ──► Clinical guidance (rules) ──► Report ──► Human validation ──► CSV audit log
                                          │
                                          └── CrewAI pipeline (3 agents, Gemini)
                                              1. Medical Image Classifier Agent
                                              2. Clinical Retrieval Agent
                                              3. Report Generation Agent
```

- **Vision module** (`src/vision_module.py`): CNN with three convolution blocks (32, 64, 128 filters), dropout and a 2-class output. Input images are resized to 128x128 and converted to grayscale.
- **Clinical retrieval module** (`src/retrieval_module.py`): maps the predicted class to a triage tier and a recommended action. This is a rule-based lookup (not a vector-database RAG): `PNEUMONIA` → Tier 2 (Urgent), `NORMAL` → Tier 3 (Routine).
- **CrewAI pipeline** (`src/crew_triage.py`): three sequential agents that interpret the CNN result, decide the triage priority, and write the final report.
- **Reporting layer** (`src/reporter_module.py`): produces the structured JSON report, through an LLM when an API key is available, or deterministically otherwise.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney (Kaggle): 5,863 JPEG images in two classes (NORMAL, PNEUMONIA), split into `train`, `val` and `test` folders. The data originates from the study published in *Cell* (Kermany et al., 2018). See the dataset page for license terms. The images are not included in this repository.

## Model Training

The CNN was trained in Google Colab (see [`notebooks/Chest_XRay_training.ipynb`](notebooks/Chest_XRay_training.ipynb)):

| Parameter | Value |
|---|---|
| Epochs | 8 |
| Optimizer | Adam (learning rate 0.0005) |
| Batch size | 32 |
| Loss | Cross-entropy with class weights `[2.0, 1.0]` (NORMAL, PNEUMONIA) |
| Input | 128x128, grayscale, normalized |

To retrain locally (from the project root, with the dataset downloaded into `data/chest_xray/`):

```bash
python -m src.train_vision --data-dir data/chest_xray
```

The script fixes the random seed, prints the test metrics and saves the weights to `weights/cnn_pneumonia_model.pth`, along with a `_test_metrics.json` file.

## Results

Test set (624 images), results of the original Colab run:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | 97 | 137 |
| **Actual PNEUMONIA** | 5 | 385 |

- Accuracy: **77.2%**
- Recall on PNEUMONIA: **98.7%** (5 missed cases out of 390)
- Precision on PNEUMONIA: 73.8%
- Specificity on NORMAL: 41.5%

The model favors recall, which is the priority in triage (missing a sick patient is worse than raising a false alarm). The cost is a high false-positive rate: 137 of the 234 normal images are flagged as pneumonia. The original notebook did not fix a random seed, so a new training run gives slightly different numbers.

### Known limitations

- High false-positive rate on NORMAL images.
- The training loss becomes very low while test accuracy stays around 77%, a sign of overfitting and of a distribution gap between the training and test sets of this dataset.
- The validation split of the dataset has only 16 images, so it is used for monitoring only, not for model selection.
- The clinical guidance is a simple rule table, not a retrieval over real clinical documents.

## Technologies Used

- Python
- PyTorch, Torchvision
- scikit-learn (evaluation)
- Streamlit
- CrewAI, LangChain + Google Generative AI (Gemini)
- Google Colab (training)
- CSV logging

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

> If the API key is missing, the application still works in fallback mode and generates a deterministic clinical report.

## Run the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
Medical-Image-Triage/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── Chest_XRay_training.ipynb
├── weights/
│   └── cnn_pneumonia_model.pth
├── src/
│   ├── vision_module.py
│   ├── retrieval_module.py
│   ├── reporter_module.py
│   ├── crew_triage.py
│   └── train_vision.py
└── tests/
    └── test_triage_logic.py
```

`.env` and `triage_audit_log.csv` are generated locally and ignored by Git.

## Workflow

1. Upload a chest X-ray image
2. CNN predicts NORMAL or PNEUMONIA
3. Clinical guidance is retrieved
4. Structured report is generated
5. Human validates the report
6. Result is saved to a CSV audit log

## Human-in-the-Loop

The dashboard includes a manual validation step:

- Approve report
- Reject report

This keeps the system aligned with clinical safety workflows.

## Output Log

Each analysis is saved in:

```text
triage_audit_log.csv
```

The log includes:

- timestamp
- patient ID
- prediction
- triage level
- recommended action

## Tests

```bash
python -m pytest tests/
```

## Future Improvements

- reduce false positives (transfer learning, data augmentation, threshold tuning)
- use the validation set for model selection, or build a larger validation split
- integrate a real retrieval layer over clinical documents
- deploy to a web or cloud platform
- add advanced medical visualization tools

## Conclusion

This project demonstrates a functional AI-assisted medical triage pipeline combining deep learning, rule-based guidance, LLM agents, reporting, and human validation in a single interface. It is suitable for academic demonstration and further extension into a more complete clinical decision-support system.
