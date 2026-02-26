import os


EMBEDDING_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large"

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:30b-a3b")
LLM_TEMPERATURE = 0.1

DB_DIRECTORY = os.getenv("DB_DIRECTORY", "../chroma_db")
COLLECTION_NAME = "general"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_INITIAL_K = 50

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
