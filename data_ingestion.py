from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(
    file_path: str, chunk_size: int = 400, chunk_overlap: int = 50
) -> list[Document]:
    """Loads a PDF file and splits it into smaller pieces"""
    loader: PyMuPDFLoader = PyMuPDFLoader(file_path)
    documents: list[Document] = loader.load()

    splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks: list[Document] = splitter.split_documents(documents)
    return chunks
