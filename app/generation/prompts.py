"""Prompt templates for grounded generation and RAG reasoning."""

RAG_SYSTEM_PROMPT = """You are an accurate, truthful AI assistant.
Answer the user's question using ONLY the provided context snippets.
If the answer cannot be determined from the context, state clearly: "I don't know based on the provided documents."
Always include citations in the format [Chunk: <chunk_id>] when making claims supported by the text.
"""

RAG_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


def build_rag_prompt(question: str, context_chunks: list[str]) -> tuple[str, str]:
    """Construct system and user messages for grounded generation."""
    formatted_context = "\n\n".join(
        f"[{idx+1}] {chunk}" for idx, chunk in enumerate(context_chunks)
    )
    user_prompt = RAG_USER_TEMPLATE.format(context=formatted_context, question=question)
    return RAG_SYSTEM_PROMPT, user_prompt
