import os
import tempfile
import streamlit as st
import json
import csv
from datetime import datetime
import base64
from dotenv import load_dotenv
from src.crew_triage import run_crewai_triage

# Import your existing backend modules
from src.vision_module import MedicalVisionModule
from src.retrieval_module import ClinicalRetrievalModule
from src.reporter_module import ClinicalReporter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Medical Triage", page_icon="⚕️", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #0d2540 0%, #08192b 26%, #07131f 100%);
        color: #eef5ff;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1 {
        color: #f1f7ff !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em;
        text-shadow: 0 4px 16px rgba(123, 202, 255, 0.24);
    }
    .header-box {
        background: linear-gradient(135deg, rgba(23, 76, 130, 0.92), rgba(12, 29, 52, 0.92));
        border: 1px solid rgba(151, 209, 255, 0.24);
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        color: #ecf8ff;
    }
    .section-card {
        background: rgba(18, 31, 46, 0.78);
        border: 1px solid rgba(144, 195, 255, 0.18);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(8px);
    }
    .prediction-pill {
        font-size: 1.35rem;
        font-weight: 800;
        padding: 0.8rem 1.2rem;
        border-radius: 999px;
        display: inline-block;
        margin: 0.7rem 0 1rem 0;
        letter-spacing: 0.02em;
    }
    .normal {
        background: rgba(35, 196, 117, 0.16);
        color: #9ce8bb;
        border: 1px solid rgba(35,196,117,0.35);
    }
    .pneumonia {
        background: rgba(255, 99, 107, 0.14);
        color: #ffb1b6;
        border: 1px solid rgba(255,99,107,0.35);
    }
    .stFileUploader > div > div {
        border: 1px dashed rgba(144, 195, 255, 0.45);
        border-radius: 18px;
        background: rgba(20, 31, 46, 0.8);
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #67d3ff, #2c7ef5 50%, #1d4fad);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 800;
        padding: 0.8rem 1.4rem;
        box-shadow: 0 12px 28px rgba(52, 130, 239, 0.4);
    }
    div[data-testid="stButton"] > button:hover {
        filter: brightness(1.08);
    }
    .report-box {
        background: rgba(14, 27, 39, 0.85);
        border-left: 5px solid #6ad0ff;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 0.75rem;
        box-shadow: inset 0 0 0 1px rgba(125, 170, 255, 0.1);
    }
    .stAlert {
        border-radius: 16px;
    }
    .stJson {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚕️ Multimodal AI Medical Triage Dashboard")
st.markdown("""
<div class="header-box">
    <strong>Clinical workflow:</strong> upload a chest X-ray, classify it as NORMAL or PNEUMONIA, and generate a structured triage report for human review.
</div>
""", unsafe_allow_html=True)

# --- Initialization (Cached so it doesn't reload every click) ---
@st.cache_resource
def load_ai_modules():
    load_dotenv()

    if "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"]:
        primary_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        robust_llm = primary_llm.with_fallbacks([fallback_llm])
    else:
        robust_llm = None
        st.warning("⚠️ GOOGLE_API_KEY not found. The app will use a deterministic clinical report fallback so the classification still displays correctly.")

    weights_path = "./weights/cnn_pneumonia_model.pth"
    vision = MedicalVisionModule(model_weights_path=weights_path if os.path.exists(weights_path) else None)
    retriever = ClinicalRetrievalModule()
    reporter = ClinicalReporter(llm=robust_llm)

    return vision, retriever, reporter, robust_llm

# Load the brains
vision_module, retrieval_module, reporter_module, llm = load_ai_modules()

def extract_pathology(image_path, modality):
    """Uses Gemini Vision to extract the actual medical finding from the image."""
    if llm is None:
        return modality

    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    prompt_text = (
        f"You are a radiologist. This is an image classified as '{modality}'. "
        "Identify any major pathological finding. If normal, state 'Normal'. "
        "Keep your response strictly under 10 words."
    )
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ])
    return llm.invoke([message]).content.strip()

def log_triage_result(patient_id, modality, finding, tier, action):
    """Appends the AI triage results to a local CSV audit log."""
    log_file = "triage_audit_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write headers if the file is brand new
        if not file_exists:
            writer.writerow(["Timestamp", "Patient_ID", "Predicted_Modality", "AI_Finding", "Triage_Tier", "Recommended_Action"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, patient_id, modality, finding, tier, action])

# --- The User Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-card"><h3 style="margin-top:0; color:#0f3d67;">1. Patient Scan</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Medical Image (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Scan", width="stretch")

with col2:
    st.markdown('<div class="section-card"><h3 style="margin-top:0; color:#0f3d67;">2. Triage Analysis</h3></div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("Run AI Triage Pipeline", type="primary"):
            with st.spinner("Analyzing scan and consulting clinical guidelines..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    vision_res = vision_module.extract_finding(tmp_path)
                    pathology = vision_res["modality_finding"]
                    confidence = vision_res["confidence_score"]
                    modality = "Chest X-Ray"

                    guideline = retrieval_module.retrieve_guideline(pathology)
                    patient_id = f"PT-{os.urandom(2).hex().upper()}"

                    try:
                        if llm is not None:
                            crew_result = run_crewai_triage(
                                patient_id=patient_id,
                                prediction=pathology,
                                confidence=confidence,
                                guideline=guideline,
                            )
                            st.subheader("CrewAI Multi-Agent Reasoning")
                            st.write(crew_result)
                    except Exception as e:
                        st.warning(f"⚠️ CrewAI unavailable ({e}). Continuing with the deterministic medical report.")

                    report = reporter_module.generate_report(
                        patient_id=patient_id,
                        finding=pathology,
                        confidence=confidence,
                        guideline=guideline,
                    )

                    st.success("Triage Complete!")
                    st.markdown(f'<div class="prediction-pill {"normal" if pathology == "NORMAL" else "pneumonia"}">{pathology}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {confidence * 100:.1f}%")

                    tier = report.get("triage_tier", "Unknown Tier")
                    action = report.get("recommended_action", "No action specified")

                    if "1" in tier or "CRITICAL" in tier.upper():
                        st.error(f"🚨 {tier}")
                    elif "2" in tier or "URGENT" in tier.upper():
                        st.warning(f"⚠️ {tier}")
                    else:
                        st.info(f"✅ {tier}")

                    st.subheader("Medical Report")
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.json(report)
                    st.markdown('</div>', unsafe_allow_html=True)

                    log_triage_result(patient_id, modality, pathology, tier, action)

                    approve = st.radio(
                        "Human Validation Required",
                        ["Approve Report", "Reject Report"],
                        horizontal=True,
                    )

                    if approve == "Approve Report":
                        st.success("✅ Final report approved by human reviewer.")
                    else:
                        st.error("❌ Report rejected. Further clinical review required.")

                    st.toast("✅ Report saved to triage_audit_log.csv")

                finally:
                    os.remove(tmp_path)
    else:
        st.info("Awaiting image upload...")