class ClinicalRetrievalModule:
    def __init__(self):
        self.guidelines = {
            "NORMAL": {
                "tier": "Tier 3 (Routine)",
                "message": "No pneumonia detected by the CNN.",
                "action": "Continue standard clinical observation and review if symptoms persist."
            },
            "PNEUMONIA": {
                "tier": "Tier 2 (Urgent)",
                "message": "Possible pneumonia detected by the CNN.",
                "action": "Urgent clinical evaluation is recommended, including physical examination and further tests if needed."
            }
        }

    def retrieve_guideline(self, finding):
        finding = finding.upper()

        if "PNEUMONIA" in finding:
            guideline = self.guidelines["PNEUMONIA"]
            return (
                "PROTOCOL: Chest X-ray triage\n"
                f"- Finding: {finding}\n"
                f"- Tier: {guideline['tier']}\n"
                f"- Action: {guideline['action']}"
            )

        if "NORMAL" in finding:
            guideline = self.guidelines["NORMAL"]
            return (
                "PROTOCOL: Chest X-ray triage\n"
                f"- Finding: {finding}\n"
                f"- Tier: {guideline['tier']}\n"
                f"- Action: {guideline['action']}"
            )

        return (
            "PROTOCOL: Chest X-ray triage\n"
            "- Finding: UNKNOWN\n"
            "- Tier: Tier 3 (Routine)\n"
            "- Action: Human clinical review is required."
        )