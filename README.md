<div align="center">

# 💬 RAG Document Chatbot

### Chat with your documents. Keep your data yours.

An intelligent, private document assistant that runs entirely on your
computer. Upload files, ask questions in natural language, and get grounded
answers with citations — without sending your documents to the cloud.

<p>
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/LLM-Ollama-black?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/Privacy-100%25%20local-22C55E?style=for-the-badge" alt="100 percent local">
  <img src="https://img.shields.io/badge/License-To%20be%20added-gray?style=for-the-badge" alt="License">
</p>

</div>

---

## ✨ Why this project?

Most chatbots make you choose between convenience and privacy. This one is
designed to give you both:

> **Your files stay on your machine. Your answers are grounded in your files.**

No Claude API. No Mistral API key. No hosted vector database. Just a local
FastAPI backend, a clean Streamlit interface, and Ollama-powered intelligence.

## 🚀 What it can do

| Capability | What you get |
|---|---|
| 📚 Multi-format uploads | PDF, DOCX, TXT, and Markdown |
| 🗂️ Document collections | Group uploads by collection |
| 🧹 Document management | Re-index or delete indexed documents |
| 🧠 Semantic retrieval | Finds meaning, not only exact keywords |
| 🔀 Hybrid retrieval | Combines semantic similarity with lexical matching |
| 💬 Conversation memory | Ask natural follow-up questions |
| 🔎 Grounded answers | Responses are based on retrieved document context |
| 📌 Citations | See which document and page supported the answer |
| 📝 Summaries | Generate a concise summary for an indexed document |
| 🔒 Local-first privacy | No document leaves your computer |

## 🧭 How it works

```text
  Upload files
       │
       ▼
  Split into smart chunks ──► Create local embeddings
                                      │
                                      ▼
                         Hybrid retrieval
                                      │
                                      ▼
                         Ollama generates an answer
                                      │
                                      ▼
                         Answer + source citations
```

## 🛠️ Technology

- **Python 3.14** — application runtime
- **FastAPI** — backend API
- **Streamlit** — interactive web interface
- **LangChain** — document loading and chunking
- **Ollama** — local model runtime
- **`llama3.2:3b`** — local chat model
- **`nomic-embed-text`** — local embedding model

## ⚡ Quick start

### 1. Install the project

Open PowerShell:

```powershell
cd "C:\Users\marni\Desktop\RAG Document Chatbot"

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Install Ollama and download the models

Install Ollama for Windows:

<https://ollama.com/download/windows>

Then run:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Ollama usually starts automatically. If it does not, open the Ollama
application before continuing.

### 3. Start the backend

Use the first PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

Verify the backend at <http://127.0.0.1:8000/health>. You should see:

```json
{"status":"ok"}
```

### 4. Start the interface

Open a second PowerShell window:

```powershell
cd "C:\Users\marni\Desktop\RAG Document Chatbot"
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

Open <http://127.0.0.1:8501> in your browser.

## 🎯 Your first conversation

1. Upload one or more files.
2. Choose a collection name in the sidebar.
3. Click **Index documents**.
4. Ask a question in the chat box.
5. Read the answer and expand **Sources** to inspect citations.
6. Use the sidebar to **Re-index** or **Delete** a document.
7. Select a document and click **Create summary**.
8. Ask a follow-up question using the same conversation.

For the best results, upload clear, text-based documents and ask focused
questions such as:

```text
What are the three main conclusions in this report?
```

```text
What does the document say about the project timeline?
```

## 📁 Project structure

```text
RAG Document Chatbot/
├── app/
│   ├── config.py       # Local application settings
│   ├── main.py         # FastAPI endpoints
│   └── rag.py          # Loading, indexing, retrieval, and answers
├── streamlit_app.py    # Web interface and drag-and-drop collections
├── evaluation/         # Ground-truth questions and RAG metrics
├── requirements.txt    # Python dependencies
├── .env.example        # Safe configuration template
└── README.md
```

## 🔐 Privacy and local data

This project is local-first:

- Documents are processed on your machine.
- Ollama models run locally.
- No cloud API key is required.
- `local_index.json` stores the indexed text and embeddings.
- `uploads/` stores local copies so documents can be re-indexed later.
- `.env`, `local_index.json`, and virtual environments are ignored by Git.

To delete the local index and start fresh:

```powershell
Remove-Item .\local_index.json
```

Never commit `.env` or private documents to a public repository.

## 🧯 Troubleshooting

### Ollama model not found

```powershell
ollama list
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### CUDA out-of-memory

Use a smaller model in `.env`:

```env
OLLAMA_CHAT_MODEL=llama3.2:1b
```

Then download it:

```powershell
ollama pull llama3.2:1b
```

### Port already in use

Start the backend on another port:

```powershell
python -m uvicorn app.main:app --reload --port 8001
```

Then update `.env`:

```env
API_URL=http://localhost:8001
```

### Streamlit cannot connect

Confirm that the backend terminal is still running and that
<http://127.0.0.1:8000/health> returns `{"status":"ok"}`.

## 📊 Evaluate retrieval quality

The repository includes a lightweight evaluation harness in `evaluation/`.
Replace the sample rows in `evaluation/questions.json` with questions,
expected source filenames, and expected answer terms from your documents.

Run:

```powershell
python evaluation\run_evaluation.py
```

The report includes source recall@5, answer-term coverage, retrieved sources,
and the generated answer for each question. Use it as a regression check after
changing chunking, embeddings, retrieval weights, or the local model.

## 🗺️ Possible next steps

- Add chat export
- Add support for scanned PDFs with local OCR


<div align="center">

Built for private, grounded conversations with your own knowledge.

</div>
