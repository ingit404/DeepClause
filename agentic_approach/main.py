from agentic_approach.agent.builder import agent
from agentic_approach.memory.store import memory, build_context, should_store_in_memory

def extract_text(response) -> str:
    if hasattr(response, "output"):
        return response.output
    if isinstance(response, str):
        return response
    return str(response)

def chat(user_query: str) -> str:
    contextual_query = build_context(memory, user_query)

    raw_response = agent.run(contextual_query)
    answer = extract_text(raw_response)

    if should_store_in_memory(answer):
        memory.save_context(
            {"input": user_query},
            {"output": answer}
        )

    return answer

if __name__ == "__main__":
    print("Agentic RAG + Email Chat ready.\n")

    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            print("\nAssistant:")
            print(chat(q))
            print("-" * 60)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
