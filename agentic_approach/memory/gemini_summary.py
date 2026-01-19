from google import genai
from langchain_core.language_models.llms import LLM as LangChainLLM
from agentic_approach.config import MODEL_NAME

class GeminiSummaryLLM(LangChainLLM):
    client: genai.Client

    @property
    def _llm_type(self) -> str:
        return "gemini-summary"

    def _call(self, prompt: str, stop=None) -> str:
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 256,
            },
        )
        return response.text
