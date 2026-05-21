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
st.title("⚕️ Multimodal AI Medical Triage Dashboard")
st.markdown("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA using a PyTorch CNN, then generate an AI-assisted triage report.")

# --- Initialization (Cached so it doesn't reload every click) ---
@st.cache_resource
def load_ai_modules():
    # Load hidden API keys from the local .env file safely
    load_dotenv()
    
    # Fail-safe check to make sure the environment key is present locally
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("🚨 Missing GOOGLE_API_KEY! Please ensure your key is added to the local .env file.")
        st.stop()
        
    primary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    fallback_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    robust_llm = primary_llm.with_fallbacks([fallback_llm])
    
    # Load custom modules
    weights_path = "./weights/cnn_pneumonia_model.pth"
    vision = MedicalVisionModule(model_weights_path=weights_path if os.path.exists(weights_path) else None)
    retriever = ClinicalRetrievalModule()
    reporter = ClinicalReporter(llm=robust_llm)
    
    return vision, retriever, reporter, robust_llm

# Load the brains
vision_module, retrieval_module, reporter_module, llm = load_ai_modules()

def extract_pathology(image_path, modality):
    """Uses Gemini Vision to extract the actual medical finding from the image."""
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
    st.subheader("1. Patient Scan")
    uploaded_file = st.file_uploader("Upload Medical Image (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)

with col2:
    st.subheader("2. Triage Analysis")
    
    if uploaded_file is not None:
        if st.button("Run AI Triage Pipeline", type="primary"):
            with st.spinner("Analyzing scan and consulting clinical guidelines..."):
                
                # Save uploaded file temporarily for the CNN to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Step A: CNN Pneumonia Classification
                    vision_res = vision_module.extract_finding(tmp_path)
                    pathology = vision_res["modality_finding"]   # NORMAL or PNEUMONIA
                    confidence = vision_res["confidence_score"]
                    modality = "Chest X-Ray"
                    
                    # Step C: RAG Retrieval
                    guideline = retrieval_module.retrieve_guideline(pathology)
                    # Generate a unique patient ID for the audit log
                    patient_id = f"PT-{os.urandom(2).hex().upper()}" 
                    crew_result = run_crewai_triage(
                    patient_id=patient_id,
                    prediction=pathology,
                    confidence=confidence,
                    guideline=guideline)

                    st.subheader("CrewAI Multi-Agent Reasoning")
                    st.write(crew_result)
                    # Step D: JSON Generation
                    
                    report = reporter_module.generate_report(
                        patient_id=patient_id,
                        finding=pathology,
                        confidence=confidence,
                        guideline=guideline
                    )
                    
                    # --- Display Results ---
                    st.success("Triage Complete!")
                    
                    # Visual Badges based on Tier
                    tier = report.get("triage_tier", "Unknown Tier")
                    action = report.get("recommended_action", "No action specified")
                    
                    if "1" in tier or "CRITICAL" in tier.upper():
                        st.error(f"🚨 {tier}")
                    elif "2" in tier or "URGENT" in tier.upper():
                        st.warning(f"⚠️ {tier}")
                    else:
                        st.info(f"✅ {tier}")
                        
                    # Show the JSON
                    st.json(report)
                    
                    # --- Save to Audit Log ---
                    log_triage_result(patient_id, modality, pathology, tier, action)

                    # Show the JSON
                    st.json(report)

                    # --- Human in the Loop ---
                    approve = st.radio(
                       "Human Validation Required",
                       ["Approve Report", "Reject Report"])

                    if approve == "Approve Report":
                     st.success("✅ Final report approved by human reviewer.")
                    else:
                     st.error("❌ Report rejected. Further clinical review required.")

                    # --- Save to Audit Log ---
                    log_triage_result(patient_id, modality, pathology, tier, action)
                    st.toast("✅ Report saved to triage_audit_log.csv")

                finally:
                    # Clean up the temp file
                    os.remove(tmp_path)
    else:
        st.info("Awaiting image upload...")