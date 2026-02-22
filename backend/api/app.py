import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag.generation import ask, RAGResponse
from rag.data_ingestion import ingest_pdf
from rag.vector_store import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Ładowanie modelu RAG do pamięci...")
    get_store()
    print("[API] Serwer gotowy do pracy.")
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    """Information about one chunk used for answer"""

    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


@app.post("/ask", response_model=AskResponse)
def handle_ask(request: AskRequest):
    """
    Looks for proper chunks in database and generates answer.
    """
    response: RAGResponse = ask(query=request.query)

    sources = []
    for doc, score in response.source_chunks:
        page_index = doc.metadata.get("page", -1)
        page_num = int(page_index) + 1 if page_index != -1 else -1
        sources.append(SourceInfo(page=page_num, score=round(score, 4)))

    return AskResponse(answer=response.answer, sources=sources)


@app.post("/upload", response_model=UploadResponse)
async def handle_upload(file: UploadFile):
    """
    Uploads a PDF file to database.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_added = ingest_pdf(file_path=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        filename=file.filename or "unknown.pdf",
        chunks_added=chunks_added,
    )
