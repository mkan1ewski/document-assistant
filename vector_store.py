from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

from config import EMBEDDING_MODEL_NAME, DB_DIRECTORY, COLLECTION_NAME


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """Downloads and returns the embedding model"""
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embedding_model


def get_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    db_directory: str = DB_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Connects to an existing Chroma database or creates a new empty one
    if it doesn't exist at the specified directory.
    """
    vector_store: Chroma = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )
    return vector_store


def add_chunks_to_database(vector_store: Chroma, chunks: list[Document]) -> None:
    """
    Adds new document chunks to the existing vector database
    prevents duplicates with hashing.
    """
    chunk_ids: list[str] = []
    for chunk in chunks:
        source: str = chunk.metadata.get("source", "unknown")
        page: str = str(chunk.metadata.get("page", "unknown"))
        content: str = chunk.page_content
        content_to_hash: str = f"{source}_{page}_{content}"
        chunk_hash: str = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
        chunk_ids.append(chunk_hash)

    vector_store.add_documents(documents=chunks, ids=chunk_ids)


def search_similar_chunks(
    vector_store: Chroma, query: str, k: int = 3
) -> list[tuple[Document, float]]:
    """
    Searches the database for the top 'k' chunks most similar to the query.
    Returns a list of tuples: (Document, distance_score).
    """
    print(f"\nSearching for top {k} matches for query: '{query}'...")

    results: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query=query, k=k
    )

    return results
