from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from ollama import ResponseError
from pydantic import BaseModel, Field

from app.rag import RAGService

app = FastAPI(title="RAG Document Chatbot", version="1.0.0")
rag = RAGService()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict[str, str]] = Field(default_factory=list)


class SummaryRequest(BaseModel):
    source: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("default"),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="Supported file types are PDF, DOCX, TXT, and Markdown.",
        )

    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temporary_path = Path(temporary.name)
        return rag.ingest(temporary_path, file.filename or "document", collection)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/documents")
def list_documents() -> list[dict]:
    return rag.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    try:
        return rag.delete_document(document_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/documents/{document_id}/reindex")
def reindex_document(document_id: str) -> dict:
    try:
        return rag.reindex_document(document_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)


@app.post("/chat")
def chat(request: QuestionRequest) -> dict:
    try:
        return rag.answer(request.question, request.top_k, request.history)
    except ResponseError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The local Ollama model could not start. Try a smaller model or "
                f"check Ollama: {error}"
            ),
        ) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/summarize")
def summarize(request: SummaryRequest) -> dict[str, str]:
    try:
        return {"summary": rag.summarize(request.source)}
    except ResponseError as error:
        raise HTTPException(
            status_code=503,
            detail=f"The local Ollama model could not start: {error}",
        ) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
