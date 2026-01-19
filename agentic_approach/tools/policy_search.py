from google.genai import types
from crewai_tools import tool
from agentic_approach.utils.llm_client import get_gemini_client
from agentic_approach.config import PROJECT_ID, LOCATION, MODEL_NAME

client = get_gemini_client()

# Vertex AI RAG configuration
rag_tool_config = types.Tool(
    retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(
                    rag_corpus=(
                        f"projects/{PROJECT_ID}/"
                        f"locations/{LOCATION}/"
                        "ragCorpora/6917529027641081856"
                    )
                )
            ]
        )
    )
)

@tool("search_policy_documents")
def search_policy_documents(query: str) -> dict:
    """
    Search RBI policy and compliance documents.
    Use ONLY for regulatory, eligibility, and rule-based questions.
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[rag_tool_config],
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    chunks = (
        response.candidates[0]
        .grounding_metadata
        .grounding_chunks or []
    )

    return {
        "answer": response.text,
        "sources": [
            {
                "uri": c.retrieved_context.uri,
                "snippet": c.retrieved_context.text[:200]
            }
            for c in chunks
        ]
    }
