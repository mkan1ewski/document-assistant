from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embedding_model(model_name: str) ->  HuggingFaceEmbeddings:
    """Downloads and returns the embedding model"""
    print(f"Loading embedding model: {model_name}...")
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model

def get_vector_store(embedding_model: HuggingFaceEmbeddings, db_directory: str = "./chroma_db", collection_name = "general") -> Chroma:
    """
    Connects to an existing Chroma database or creates a new empty one
    if it doesn't exist at the specified directory.
    """
    vector_store: Chroma = Chroma(persist_directory=db_directory, embedding_function=embedding_model, collection_name=collection_name)
    return vector_store

def add_chunks_to_database(vector_store: Chroma, chunks: list[Document]) -> None:
    """Adds new document chunks to the existing vector database."""
    vector_store.add_documents(documents=chunks)

