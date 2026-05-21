Multimodal AI Medical Triage System

AI & Big Data Program — UIR
Integrated Project 2025/2026

Project Overview

This project implements a Multi-Agent Medical Image Triage System using:

Deep Learning (PyTorch CNN)
CrewAI-inspired multi-agent orchestration
Streamlit dashboard
Clinical guideline retrieval
Structured medical report generation
Human-in-the-loop validation

The system analyzes chest X-ray images and classifies them into:

NORMAL
PNEUMONIA

It then generates an AI-assisted medical triage report.

Features
Deep Learning CNN
Custom PyTorch CNN trained on chest X-ray images
Binary classification:
NORMAL
PNEUMONIA
Multi-Agent Architecture

Specialized AI agents:

Orchestrator Agent
Medical Vision Agent
Clinical Retrieval Agent
Report Generation Agent
Human-in-the-Loop
Manual validation checkpoint
Report approval/rejection
Logging
CSV audit logging
Timestamped triage history
Streamlit Dashboard
Upload medical scans
Run AI triage pipeline
Display predictions and reports

Project Structure
multimodal-ai-medical-triage-main/
│
├── data/
├── notebooks/
├── src/
│   ├── vision_module.py
│   ├── retrieval_module.py
│   ├── reporter_module.py
│   ├── crew_triage.py
│   └── cnn_classifier_tool.py
│
├── weights/
│   └── cnn_pneumonia_model.pth
│
├── app.py
├── requirements.txt
├── triage_audit_log.csv
└── README.md

Dataset

Dataset used:

Chest X-Ray Images (Pneumonia)
Kaggle Dataset

Classes:

NORMAL
PNEUMONIA
Technologies Used
| Component             | Technology   |
| --------------------- | ------------ |
| Programming Language  | Python       |
| Deep Learning         | PyTorch      |
| Multi-Agent Framework | CrewAI       |
| Interface             | Streamlit    |
| LLM Backend           | Gemini API   |
| Image Processing      | Torchvision  |
| IDE                   | VS Code      |
| Training Environment  | Google Colab |
Model Training

The CNN model was trained using:

CrossEntropyLoss
Adam Optimizer
Torchvision transforms
DataLoaders

Evaluation metrics:

Accuracy
Confusion Matrix

Approximate accuracy:

~78%
Installation
Clone Repository
git clone <your_repo_link>
cd multimodal-ai-medical-triage-main
Install Dependencies
pip install -r requirements.txt
Environment Variables

Create a .env file:

GOOGLE_API_KEY=your_gemini_api_key
Run the Application
streamlit run app.py

Application runs on:

http://localhost:8501
Workflow
Upload chest X-ray image
CNN predicts NORMAL or PNEUMONIA
Clinical retrieval module selects guideline
Report agent generates structured report
Human validates report
Result logged into audit file
Human-in-the-Loop

The system includes a manual validation checkpoint where the generated medical report can be:

Approved
Rejected

This ensures safe AI-assisted decision support.

Logging

Every AI triage action is logged with:

Timestamp
Patient ID
Prediction
Triage level
Recommended action

Stored in:

triage_audit_log.csv
Future Improvements
Improve CNN accuracy with transfer learning
Add larger medical datasets
Deploy cloud version
Add multilingual support
Integrate full RAG vector database
Add authentication/security
Authors

UIR — AI & Big Data Program
Integrated Project 2025/2026

References
PyTorch Documentation
CrewAI Documentation
Streamlit Documentation
Gemini API Documentation
Kaggle Pneumonia Dataset