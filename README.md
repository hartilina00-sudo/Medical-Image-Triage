# Multimodal AI Medical Triage System

AI & Big Data Program — UIR  
Integrated Project 2025/2026

## Project Overview

This project implements an AI-based medical triage dashboard for chest X-ray analysis. It combines:

- a PyTorch CNN for image classification
- a clinical retrieval module
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

## Features

### Deep Learning Model
- Custom CNN implemented with PyTorch
- Binary classification for pneumonia detection
- Model weights stored in the weights directory

### Clinical Decision Support
- Retrieval of clinical guidance based on the predicted class
- Urgent vs routine triage logic
- Action recommendation for human review

### Report Generation
- Structured medical JSON report
- Human validation checkpoint
- CSV audit logging for each triage event

### Interface
- Upload chest X-ray image
- Run AI analysis
- View prediction and confidence
- Review the final report in the dashboard

## Project Structure

```bash
Medical-Image-Triage-main/
├── app.py
├── requirements.txt
├── README.md
├── triage_audit_log.csv
├── .env
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

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- LangChain + Google Generative AI
- CrewAI
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

## Future Improvements

- improve model accuracy with larger datasets
- fine-tune with transfer learning
- integrate larger clinical document retrieval
- deploy to a web or cloud platform
- add advanced medical visualization tools

## Conclusion

This project demonstrates a functional AI-assisted medical triage pipeline combining deep learning, retrieval, reporting, and human validation in a single interface. It is suitable for academic demonstration and further extension into a more complete clinical decision-support system.
