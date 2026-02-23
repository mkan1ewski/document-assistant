from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_store


def load_and_chunk_pdf(
    file_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Loads a PDF file and splits it into smaller pieces"""
    loader: PyMuPDFLoader = PyMuPDFLoader(file_path)
    documents: list[Document] = loader.load()

    clean_filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"] = clean_filename

    splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks: list[Document] = splitter.split_documents(documents)
    return chunks


def ingest_pdf(file_path: str, store=None) -> int:
    """Ingests pdf into database"""
    if store is None:
        store = get_store()

    chunks: list[Document] = load_and_chunk_pdf(file_path)
    added_count = store.add_chunks(chunks)

    return added_count
