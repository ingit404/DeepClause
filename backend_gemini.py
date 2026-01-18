
from typing import Optional, List
from google import genai
from google.genai import types
from langchain_core.language_models.llms import LLM

# -------------------------
# 1. Gemini client
# -------------------------
client = genai.Client(
    vertexai=True,
    project="lending-partner",
    location="asia-south1"
)


# -------------------------
# 2. Vertex RAG tool
# -------------------------
rag_tool = types.Tool(
    retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(
                    rag_corpus=(
                        "projects/lending-partner/"
                        "locations/asia-south1/"
                        "ragCorpora/6917529027641081856"
                    )
                )
            ]
        )
    )
)


# -------------------------
# 3. Gemini wrapper for LangChain memory summarization
# -------------------------
class GeminiLLM(LLM):
    client: genai.Client = client
    model_name: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 256,
            },
        )
        return response.text
