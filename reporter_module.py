import json
from datetime import datetime
from langchain_core.prompts import PromptTemplate

class ClinicalReporter:
    def __init__(self, llm=None):
        self.llm = llm
        
        self.prompt_template = PromptTemplate(
            input_variables=["patient_id", "date", "finding", "confidence", "guideline"],
            template="""
You are an expert AI clinical triage assistant. Generate a structured medical triage report based on the following inputs.

[INPUTS]
Patient ID: {patient_id}
Date: {date}
Modality & Finding: {finding} (Confidence: {confidence})
Retrieved Clinical Guideline: {guideline}

[INSTRUCTIONS]
Output the report strictly in the following JSON format. Do not include any extra conversational text, markdown formatting blocks, or greetings. Just raw JSON.
{{
    "patient_id": "{patient_id}",
    "timestamp": "{date}",
    "radiology_finding": "...",
    "triage_tier": "...",
    "recommended_action": "...",
    "guideline_citation": "..."
}}
"""
        )

    def generate_report(self, patient_id, finding, confidence, guideline):
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence_str = f"{confidence * 100:.1f}%"
        
        if self.llm:
            print("[INFO] Generating report via LLM...")
            prompt = self.prompt_template.format(
                patient_id=patient_id,
                date=date_str,
                finding=finding,
                confidence=confidence_str,
                guideline=guideline
            )
            response = self.llm.invoke(prompt)
            
            # Extract the actual text string from the LangChain AIMessage object
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Clean up potential markdown formatting blocks safely
            clean_json = response_text.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
            
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError:
                print("[ERROR] LLM failed to return valid JSON. Returning raw text log.")
                return {"error": "Invalid JSON format", "raw_text": response_text}
        else:
            print("[INFO] No LLM provided. Generating deterministic mock report...")
            
            tier = "Tier 3 (Routine)"
            if "Tier 1" in guideline: 
                tier = "Tier 1 (CRITICAL)"
            elif "Tier 2" in guideline: 
                tier = "Tier 2 (Urgent)"
            
            action = guideline.split("Action: ")[-1] if "Action: " in guideline else "Follow standard clinical protocol."
            citation = guideline.split("- Finding:")[0].strip() if "- Finding:" in guideline else "General Protocol"
            
            report = {
                "patient_id": patient_id,
                "timestamp": date_str,
                "radiology_finding": f"{finding} detected with {confidence_str} confidence.",
                "triage_tier": tier,
                "recommended_action": action,
                "guideline_citation": citation
            }
            return report

if __name__ == "__main__":
    reporter = ClinicalReporter(llm=None)
    
    test_report = reporter.generate_report(
        patient_id="PT-99823",
        finding="Cardiomegaly / Enlarged Heart",
        confidence=0.942,
        guideline="PROTOCOL: Chest X-Ray (CXR) Triage Protocol\n- Finding: Cardiomegaly / Enlarged Heart\n  Action: Priority Tier 2 (Urgent Evaluation)."
    )
    
    print("\n--- GENERATED JSON REPORT ---")
    print(json.dumps(test_report, indent=4))