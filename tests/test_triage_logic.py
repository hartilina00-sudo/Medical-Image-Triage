from src.retrieval_module import ClinicalRetrievalModule
from src.reporter_module import ClinicalReporter


def test_retrieval_module_returns_explicit_tier_for_pneumonia():
    module = ClinicalRetrievalModule()
    guideline = module.retrieve_guideline("PNEUMONIA")

    assert "Tier" in guideline
    assert "PNEUMONIA" in guideline.upper()
    assert "Action:" in guideline


def test_reporter_generates_structured_report_without_llm():
    reporter = ClinicalReporter(llm=None)
    report = reporter.generate_report(
        patient_id="PT-001",
        finding="NORMAL",
        confidence=0.923,
        guideline="PROTOCOL: Standard chest X-ray review\n- Finding: NORMAL\n- Tier: Tier 3 (Routine)\n- Action: Continue standard clinical observation and review if symptoms persist."
    )

    assert report["patient_id"] == "PT-001"
    assert "NORMAL" in report["radiology_finding"].upper()
    assert report["triage_tier"] == "Tier 3 (Routine)"
    assert "observation" in report["recommended_action"].lower()
