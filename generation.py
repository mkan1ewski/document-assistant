import re
from dataclasses import dataclass

import ollama
from langchain_core.documents import Document

from vector_store import (
    get_embedding_model,
    get_vector_store,
    search_similar_chunks,
)


DEFAULT_MODEL = "qwen3:30b-a3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"

SYSTEM_PROMPT = """\
Jesteś pomocnym asystentem firmowym, który odpowiada na pytania WYŁĄCZNIE
na podstawie dostarczonego kontekstu.

Zasady:
- Odpowiadaj TYLKO na podstawie informacji zawartych w kontekście poniżej.
- Jeśli kontekst nie zawiera odpowiedzi na pytanie, powiedz wprost:
  "Nie znalazłem odpowiedzi na to pytanie w dostępnych dokumentach."
- NIE wymyślaj informacji, które nie wynikają z kontekstu.
- Odpowiadaj zwięźle i konkretnie, po polsku.
- Jeśli to możliwe, wskaż z której strony dokumentu pochodzi informacja.
"""


@dataclass
class RAGResponse:
    answer: str
    source_chunks: list[tuple[Document, float]]
    model: str


def _format_context(chunks: list[tuple[Document, float]]) -> str:
    context_parts: list[str] = []

    for rank, (doc, score) in enumerate(chunks, start=1):
        page_index = doc.metadata.get("page", -1)
        page_display = int(page_index) + 1 if page_index != -1 else "?"

        header = f"[Źródło: strona {page_display} | trafność: {score:.4f}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n---\n".join(context_parts)


def _strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def generate_answer(
    query: str,
    context_chunks: list[tuple[Document, float]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> RAGResponse:
    context_text = _format_context(context_chunks)

    user_message = f"Kontekst:\n{context_text}\n\nPytanie: {query}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message + "\n/nothink"},
        ],
        options={
            "temperature": temperature,
        },
    )

    answer_text: str = response["message"]["content"]
    answer_text = _strip_thinking_tags(answer_text)

    return RAGResponse(
        answer=answer_text,
        source_chunks=context_chunks,
        model=model,
    )


def ask(
    query: str,
    vector_store,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
) -> RAGResponse:
    retrieved_chunks = search_similar_chunks(
        vector_store=vector_store, query=query, k=top_k
    )

    response = generate_answer(
        query=query,
        context_chunks=retrieved_chunks,
        model=model,
        temperature=temperature,
    )

    return response


def display_response(response: RAGResponse) -> None:
    print(f"\n{'=' * 60}")
    print("ODPOWIEDŹ:")
    print(f"{'=' * 60}")
    print(response.answer)

    print(f"\n{'─' * 60}")
    print(f"Model: {response.model}")
    print("Źródła:")
    for doc, score in response.source_chunks:
        page_index = doc.metadata.get("page", -1)
        page_display = int(page_index) + 1 if page_index != -1 else "?"
        print(f"  • strona {page_display} (dystans: {score:.4f})")
    print(f"{'=' * 60}")



def main() -> None:
    print("Ładowanie modelu embeddingowego...")
    embedding_model = get_embedding_model(EMBEDDING_MODEL)

    print("Łączenie z bazą wektorową...")
    vector_db = get_vector_store(
        embedding_model=embedding_model,
        collection_name="general",
    )

    print(f"\nModel generatywny: {DEFAULT_MODEL}")
    print("Wpisz pytanie (lub 'q' aby zakończyć):\n")

    while True:
        query = input(">>> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("Do widzenia!")
            break
        if not query:
            continue

        try:
            response = ask(query=query, vector_store=vector_db)
            display_response(response)
        except Exception as error:
            print(f"\nBłąd: {error}")


if __name__ == "__main__":
    main()