import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from rag.config import (
    EMBEDDING_MODEL_NAME,
    DB_DIRECTORY,
    COLLECTION_NAME,
    TOP_K,
    RERANK_INITIAL_K,
    RERANK_MODEL_NAME,
)


class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        db_directory: str = DB_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        rerank_model_name: str = RERANK_MODEL_NAME,
    ) -> None:
        """
        Creates or connects to existing database
        """
        self._embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self._chroma = Chroma(
            persist_directory=db_directory,
            embedding_function=self._embedding_model,
            collection_name=collection_name,
        )

        self._reranker = CrossEncoder(rerank_model_name)

    def add_chunks(self, chunks: list[Document]) -> None:
        """
        Adds chunks to the database.
        Skips duplicates.
        """
        candidate_ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        existing_ids: set[str] = set(self._chroma.get()["ids"])

        new_chunks: list[Document] = []
        new_ids: list[str] = []
        for chunk, chunk_id in zip(chunks, candidate_ids):
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if new_chunks:
            self._chroma.add_documents(documents=new_chunks, ids=new_ids)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
        """
        Searches for k most similar chunks.
        """
        return self._chroma.similarity_search_with_score(query=query, k=k)

    def search_with_rerank(
        self,
        query: str,
        initial_k: int = RERANK_INITIAL_K,
        final_k: int = TOP_K,
    ) -> list[tuple[Document, float]]:
        """
        Two step searching, initial k chunks from database,
        then reranking with cross encoder.
        """
        candidates = self._chroma.similarity_search_with_score(query=query, k=initial_k)

        if not candidates:
            return []

        docs = [doc for doc, _ in candidates]
        pairs = [[query, doc.page_content] for doc in docs]

        scores = self._reranker.predict(pairs)

        scored_results = list(zip(docs, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for doc, score in scored_results[:final_k]]

    def count(self) -> int:
        """Returns count of chunks in database."""
        return len(self._chroma.get()["ids"])

    @staticmethod
    def _generate_chunk_id(chunk: Document) -> str:
        """Generates hash ID for a chunk."""
        source = chunk.metadata.get("source", "unknown")
        page = str(chunk.metadata.get("page", "unknown"))
        content = chunk.page_content

        content_to_hash = f"{source}_{page}_{content}"
        return hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()


_default_store: VectorStore | None = None


def get_store() -> VectorStore:
    """
    Returns the default instance of a vector store.
    """
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
