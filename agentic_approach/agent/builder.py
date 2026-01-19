from crewai import Agent
from crewai.llm import LLM
from agentic_approach.config import MODEL_NAME, SYSTEM_INSTRUCTION
from agentic_approach.tools.policy_search import search_policy_documents
from agentic_approach.tools.email_search import search_emails

def create_agent():
    gemini_llm = LLM(
        model=MODEL_NAME,
        temperature=0.3
    )

    return Agent(
        role="Lending Compliance Assistant",
        goal="Provide compliant answers using RBI policies and recent emails.",
        backstory=SYSTEM_INSTRUCTION,
        tools=[
            search_policy_documents,
            search_emails
        ],
        llm=gemini_llm,
        memory=False,        # CRITICAL: do not enable
        verbose=True
    )

agent = create_agent()
