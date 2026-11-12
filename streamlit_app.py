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

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Index documents", disabled=not uploaded_files):
    for uploaded_file in uploaded_files:
        response = post_json(
            "/documents",
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
