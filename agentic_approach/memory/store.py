from langchain_classic.memory import ConversationSummaryBufferMemory
from agentic_approach.utils.llm_client import get_gemini_client
from agentic_approach.memory.gemini_summary import GeminiSummaryLLM

# Policy-only conversation memory
def create_memory():
    client = get_gemini_client()
    return ConversationSummaryBufferMemory(
        llm=GeminiSummaryLLM(client=client),
        max_token_limit=1000,
        memory_key="history",
        input_key="input",
    )

memory = create_memory()

def build_context(mem, user_query: str) -> str:
    history = mem.load_memory_variables({}).get("history", "").strip()
    if not history:
        return user_query

    return f"""
Conversation context (policy discussions only):
{history}

Current user question:
{user_query}
""".strip()

def should_store_in_memory(answer: str) -> bool:
    if not isinstance(answer, str):
        return False

    email_indicators = [
        "email", "mail", "thread", "from:", "subject:", "approved"
    ]

    text = answer.lower()
    return not any(indicator in text for indicator in email_indicators)
