class ClinicalRetrievalModule:
    def __init__(self):
        self.guidelines = {
            "NORMAL": "No pneumonia detected by the CNN. Non-urgent review recommended. Continue standard clinical observation if symptoms persist.",
            "PNEUMONIA": "Possible pneumonia detected by the CNN. Urgent clinical evaluation is recommended, including physical examination and further tests if needed."
        }

    def retrieve_guideline(self, finding):
        finding = finding.upper()

        if "PNEUMONIA" in finding:
            return self.guidelines["PNEUMONIA"]

        if "NORMAL" in finding:
            return self.guidelines["NORMAL"]

        return "No specific guideline found. Human clinical review is required."