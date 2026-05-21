#  Multimodal AI Medical Triage System

AI & Big Data Program — UIR  
Integrated Project 2025/2026

---
#  Members
Lina Harti / Amine Lassri/ Aymane Aboulhouda / Maryam Boulminate 

#  Project Overview

This project implements a **Multi-Agent Medical Image Triage System** using:

- Deep Learning (PyTorch CNN)
- CrewAI-inspired multi-agent orchestration
- Streamlit dashboard
- Clinical guideline retrieval
- Structured medical report generation
- Human-in-the-loop validation

The system analyzes chest X-ray images and classifies them into:

- NORMAL
- PNEUMONIA

It then generates an AI-assisted medical triage report.

---

#  Features

## Deep Learning CNN
- Custom PyTorch CNN trained on chest X-ray images
- Binary classification:
  - NORMAL
  - PNEUMONIA

## Multi-Agent Architecture

Specialized AI agents:
- Orchestrator Agent
- Medical Vision Agent
- Clinical Retrieval Agent
- Report Generation Agent

## Human-in-the-Loop
- Manual validation checkpoint
- Report approval/rejection

## Logging
- CSV audit logging
- Timestamped triage history

## Streamlit Dashboard
- Upload medical scans
- Run AI triage pipeline
- Display predictions and reports

---

#  Project Structure

```bash
multimodal-ai-medical-triage-main/
│
├── src/
│   ├── vision_module.py
│   ├── retrieval_module.py
│   ├── reporter_module.py
│   ├── crew_triage.py
│   └── train_vision.py
│
├── weights/
│   └── cnn_pneumonia_model.pth
│
├── app.py
├── requirements.txt
└── README.md
```

---

#  Technologies Used

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Multi-Agent System | CrewAI |
| Interface | Streamlit |
| LLM Backend | Gemini API |
| Image Processing | Torchvision |

---

#  Dataset

Dataset used:
- Chest X-Ray Images (Pneumonia Dataset)

Classes:
- NORMAL
- PNEUMONIA

---

#  Installation

## Install dependencies

```bash
pip install -r requirements.txt
```

## Add environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

---

#  Run Application

```bash
streamlit run app.py
```

Application runs on:

```text
http://localhost:8501
```

---

#  Workflow

1. Upload chest X-ray image
2. CNN predicts NORMAL or PNEUMONIA
3. Clinical retrieval module selects guideline
4. Report agent generates structured report
5. Human validates report
6. Result logged into audit file

---

#  Human-in-the-Loop

The system includes a manual validation checkpoint where the generated medical report can be:

- Approved
- Rejected

This ensures safe AI-assisted decision support.

---

#  Logging

Every AI triage action is logged with:
- Timestamp
- Patient ID
- Prediction
- Triage level
- Recommended action

Stored in:

```text
triage_audit_log.csv
```

---

#  Future Improvements

- Improve CNN accuracy with transfer learning
- Add larger medical datasets
- Add multilingual support
- Integrate full RAG vector database
- Deploy cloud version

---

#  Authors

UIR — AI & Big Data Program  
Integrated Project 2025/2026

---

- Streamlit Documentation
- Gemini API Documentation
- Kaggle Pneumonia Dataset
