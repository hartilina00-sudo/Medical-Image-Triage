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

    @staticmethod
    def _extract_tier(guideline):
        if "Tier 1" in guideline:
            return "Tier 1 (CRITICAL)"
        if "Tier 2" in guideline:
            return "Tier 2 (Urgent)"
        if "Tier 3" in guideline:
            return "Tier 3 (Routine)"
        return "Tier 3 (Routine)"

    @staticmethod
    def _extract_action(guideline):
        if "Action:" in guideline:
            return guideline.split("Action:")[-1].strip()
        return "Follow standard clinical protocol."

    @staticmethod
    def _extract_citation(guideline):
        if "- Finding:" in guideline:
            return guideline.split("- Finding:")[0].strip()
        return "General Protocol"

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
            
            content = response.content if hasattr(response, "content") else response
            if isinstance(content, list):
                response_text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                response_text = str(content)
            
            clean_json = response_text.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
            
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError:
                print("[ERROR] LLM failed to return valid JSON. Returning raw text log.")
                return {"error": "Invalid JSON format", "raw_text": response_text}
        else:
            print("[INFO] No LLM provided. Generating deterministic report...")
            
            tier = self._extract_tier(guideline)
            action = self._extract_action(guideline)
            citation = self._extract_citation(guideline)
            
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