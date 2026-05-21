from crewai import Agent, Task, Crew, Process
from crewai import LLM
import os 
def run_crewai_triage(patient_id, prediction, confidence, guideline):
    llm = LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.environ["GOOGLE_API_KEY"]
)
    classifier_agent = Agent(
        role="Medical Image Classifier Agent",
        goal="Interpret the CNN prediction for chest X-ray triage.",
        backstory="You are responsible for validating the output of a PyTorch CNN trained to classify chest X-rays as NORMAL or PNEUMONIA.",
        llm=llm,
        verbose=True,
        allow_delegation=False
        
    )

    clinical_agent = Agent(
        role="Clinical Retrieval Agent",
        goal="Use the retrieved clinical guideline to decide the triage priority.",
        backstory="You analyze clinical guidance and convert AI findings into safe triage recommendations.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    report_agent = Agent(
        role="Report Generation Agent",
        goal="Generate a structured medical triage report.",
        backstory="You write clear JSON-style medical triage reports for human review.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    task1 = Task(
        description=f"""
        Patient ID: {patient_id}
        CNN prediction: {prediction}
        Confidence: {confidence}

        Explain the CNN result briefly.
        """,
        expected_output="A short explanation of the CNN prediction.",
        agent=classifier_agent
    )

    task2 = Task(
        description=f"""
        Prediction: {prediction}
        Clinical guideline: {guideline}

        Determine triage priority and recommended action.
        """,
        expected_output="Triage tier and recommended action.",
        agent=clinical_agent
    )

    task3 = Task(
        description=f"""
        Create a final structured report using:
        Patient ID: {patient_id}
        Prediction: {prediction}
        Confidence: {confidence}
        Guideline: {guideline}

        The report must mention that human validation is required.
        """,
        expected_output="Final structured medical triage report.",
        agent=report_agent
    )

    crew = Crew(
        agents=[classifier_agent, clinical_agent, report_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    return result