import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in WORD_PATTERN.findall(text)]


class RAGService:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.index_path = Path(settings.index_path)
        self.chunks: list[dict[str, Any]] = []
        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=settings.ollama_chat_model,
            temperature=0,
            base_url=settings.ollama_base_url,
        )
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            stored_chunks = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.chunks = [
                chunk for chunk in stored_chunks if "embedding" in chunk
            ]

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_document(path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(str(path)).load()
        if path.suffix.lower() == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader

            return Docx2txtLoader(str(path)).load()
        if path.suffix.lower() in {".txt", ".md"}:
            return [Document(page_content=path.read_text(encoding="utf-8"))]
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def ingest(self, path: Path, source_name: str | None = None) -> dict[str, Any]:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported file types are PDF, DOCX, TXT, and Markdown.")

        document_id = str(uuid4())
        display_name = source_name or path.name
        documents = self._load_document(path)
        for document in documents:
            document.metadata.update(
                {"document_id": document_id, "source": display_name}
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)
        vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
        self.chunks.extend(
            {
                "text": chunk.page_content,
                "source": display_name,
                "page": chunk.metadata.get("page"),
                "document_id": document_id,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        )
        self._save_index()
        return {
            "document_id": document_id,
            "filename": display_name,
            "chunks": len(chunks),
        }

    def _search(self, question: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        if not self.chunks:
            return []

        query_vector = self.embeddings.embed_query(question)
        query_norm = sum(value * value for value in query_vector) ** 0.5
        query_terms = set(_tokens(question))
        scored: list[tuple[dict[str, Any], float]] = []

        for chunk in self.chunks:
            vector = chunk["embedding"]
            denominator = query_norm * (sum(value * value for value in vector) ** 0.5)
            semantic_score = (
                sum(left * right for left, right in zip(query_vector, vector)) / denominator
                if denominator
                else 0
            )
            lexical_score = len(query_terms & set(_tokens(chunk["text"]))) / max(len(query_terms), 1)
            score = semantic_score * 0.8 + lexical_score * 0.2
            scored.append((chunk, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[: max(top_k, 8)]

    def answer(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        matches = self._search(question, top_k)
        if not matches or matches[0][1] == 0:
            return {
                "answer": "I could not find an answer in the uploaded documents.",
                "sources": [],
            }

        matches = matches[:top_k]
        sources: list[dict[str, Any]] = []
        context_parts: list[str] = []
        for index, (chunk, score) in enumerate(matches, start=1):
            source = chunk["source"]
            page = chunk.get("page")
            label = f"{source}, page {page + 1}" if page is not None else source
            sources.append({"label": label, "score": round(score, 4)})
            context_parts.append(f"[Source {index}: {label}]\n{chunk['text']}")

        recent_history = (history or [])[-6:]
        history_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in recent_history
        )
        context = "\n\n".join(context_parts)
        prompt = (
            "Answer using only the document context. Resolve follow-up questions "
            "using the conversation when helpful. If the context does not contain "
            "the answer, say so. Cite every factual claim with [Source N].\n\n"
            f"Conversation:\n{history_text or '(none)'}\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content, "sources": sources}

    def summarize(self, source: str) -> str:
        chunks = [chunk["text"] for chunk in self.chunks if chunk["source"] == source]
        if not chunks:
            raise ValueError(f"No indexed document named {source}.")
        response = self.llm.invoke(
            "Summarize this document in concise bullet points. Include its main "
            "purpose, key facts, and important conclusions.\n\n" + "\n\n".join(chunks[:12])
        )
        return str(response.content)
