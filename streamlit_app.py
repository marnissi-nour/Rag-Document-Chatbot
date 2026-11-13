import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")


def post_json(path: str, **kwargs: object) -> requests.Response | None:
    try:
        return requests.post(f"{API_URL}{path}", **kwargs)
    except requests.ConnectionError:
        st.error(
            "The backend is not running. Start it in another terminal with: "
            "`python -m uvicorn app.main:app --reload --port 8000`"
        )
    except requests.Timeout:
        st.error("The backend took too long to respond. Please try again.")
    return None

st.set_page_config(page_title="RAG Document Chatbot", page_icon="💬")
st.title("RAG Document Chatbot")
st.caption("Upload your documents and ask questions grounded in their content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []

st.sidebar.header("Document collections")
collection = st.sidebar.text_input("Collection name", value="default").strip() or "default"
refresh_documents = st.sidebar.button("Refresh documents")

uploaded_files = st.file_uploader(
    "Drag and drop documents here",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Index documents", disabled=not uploaded_files):
    for uploaded_file in uploaded_files:
        response = post_json(
            "/documents",
            data={"collection": collection},
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=120,
        )
        if response is not None and response.ok:
            data = response.json()
            if data["filename"] not in st.session_state.indexed_sources:
                st.session_state.indexed_sources.append(data["filename"])
            st.success(f"{data['filename']} indexed ({data['chunks']} chunks).")
        elif response is not None:
            st.error(response.text)

try:
    documents_response = requests.get(f"{API_URL}/documents", timeout=30)
except requests.ConnectionError:
    documents_response = None
    st.sidebar.error("Backend is not running.")
except requests.Timeout:
    documents_response = None
    st.sidebar.error("Backend took too long to respond.")
if documents_response is not None and documents_response.ok:
    documents = documents_response.json()
    if documents:
        st.sidebar.subheader("Indexed documents")
        for document in documents:
            label = f"{document['filename']} · {document['collection']}"
            st.sidebar.caption(f"{label} ({document['chunks']} chunks)")
            action_col1, action_col2 = st.sidebar.columns(2)
            if action_col1.button("Re-index", key=f"reindex-{document['document_id']}"):
                response = post_json(
                    f"/documents/{document['document_id']}/reindex",
                    timeout=180,
                )
                if response is not None and response.ok:
                    st.sidebar.success("Re-indexed.")
                    st.rerun()
                elif response is not None:
                    st.sidebar.error(response.text)
            if action_col2.button("Delete", key=f"delete-{document['document_id']}"):
                response = requests.delete(
                    f"{API_URL}/documents/{document['document_id']}",
                    timeout=30,
                )
                if response.ok:
                    st.sidebar.success("Deleted.")
                    st.rerun()
                else:
                    st.sidebar.error(response.text)

if st.session_state.get("indexed_sources"):
    source = st.selectbox("Summarize an indexed document", st.session_state.indexed_sources)
    if st.button("Create summary"):
        response = post_json("/summarize", json={"source": source}, timeout=180)
        if response is not None and response.ok:
            st.info(response.json()["summary"])
        elif response is not None:
            st.error(response.text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    response = post_json(
        "/chat",
        json={
            "question": question,
            "history": st.session_state.messages[:-1],
        },
        timeout=120,
    )
    with st.chat_message("assistant"):
        if response is not None and response.ok:
            data = response.json()
            st.markdown(data["answer"])
            if data["sources"]:
                with st.expander("Sources"):
                    for source in data["sources"]:
                        st.caption(f"{source['label']} — similarity {source['score']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"]}
            )
        elif response is not None:
            st.error(response.text)
