"""
FinSight — Streamlit frontend.

Four pages accessible from the sidebar:
  Chat      — conversational interface; each answer shows citations, RAGAS
              scores and fraud risk inline.
  Upload    — drag-and-drop PDF/TXT ingestion with progress feedback.
  Library   — view all ingested documents; delete with one click.
  Dashboard — points to MLflow for historical query quality tracking.
"""

import requests
import streamlit as st

API = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📊 FinSight")
st.sidebar.caption("Financial document RAG chatbot")
page = st.sidebar.radio("Navigate", ["Chat", "Upload", "Library", "Dashboard"])


# ── Helper: render answer metadata ────────────────────────────────────────────
def _render_meta(data: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness",  f"{data['faithfulness_score']:.0%}")
    c2.metric("Relevancy",     f"{data['answer_relevancy_score']:.0%}")
    fraud = data.get("fraud_risk_score")
    c3.metric("Fraud Risk",    f"{fraud:.0%}" if fraud is not None else "N/A")
    c4.metric("Latency",       f"{data['latency_ms']} ms")

    if data["fraud_flags"]:
        st.warning("🚨 Red flags: " + "  ·  ".join(data["fraud_flags"]))

    if data["sources"]:
        with st.expander(f"📄 {len(data['sources'])} source(s)"):
            for s in data["sources"]:
                st.markdown(f"**{s['filename']}** — Page {s['page']}")
                st.caption(s["excerpt"])


# ── Page: Chat ─────────────────────────────────────────────────────────────────
if page == "Chat":
    st.title("Ask FinSight")
    st.caption("Ask any question about your uploaded financial documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                _render_meta(msg["meta"])

    if question := st.chat_input("e.g. What was the revenue growth in Q3?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    resp = requests.post(f"{API}/query", json={"question": question}, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            st.markdown(data["answer"])
            _render_meta(data)

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "meta": data,
            })


# ── Page: Upload ───────────────────────────────────────────────────────────────
elif page == "Upload":
    st.title("Upload Financial Documents")
    st.caption("Supported formats: PDF, TXT. Max 50 MB. Duplicate uploads are automatically skipped.")

    files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if files and st.button("Ingest Documents", type="primary"):
        for f in files:
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    resp = requests.post(
                        f"{API}/upload",
                        files={"file": (f.name, f.read(), f.type)},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"❌ {f.name} — {e}")
                    continue

            if result["status"] == "already_exists":
                st.warning(f"⚠️ {f.name} already ingested ({result['chunks_created']} chunks exist)")
            else:
                st.success(f"✅ {f.name} — {result['chunks_created']} chunks created")


# ── Page: Library ──────────────────────────────────────────────────────────────
elif page == "Library":
    st.title("Document Library")

    try:
        resp = requests.get(f"{API}/documents", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Could not reach API: {e}")
        st.stop()

    st.caption(f"{data['total']} document(s) in the vector store")

    for doc in data["documents"]:
        col_name, col_chunks, col_del = st.columns([4, 1, 1])
        col_name.markdown(f"**{doc['filename']}**  \n`{doc['doc_id']}`  \n_{doc['ingested_at'][:19]}_")
        col_chunks.metric("Chunks", doc["chunk_count"])
        if col_del.button("🗑 Delete", key=doc["doc_id"]):
            try:
                del_resp = requests.delete(f"{API}/documents/{doc['doc_id']}", timeout=30)
                del_resp.raise_for_status()
                st.success(f"Deleted {doc['filename']}")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")


# ── Page: Dashboard ────────────────────────────────────────────────────────────
elif page == "Dashboard":
    st.title("Evaluation Dashboard")
    st.info("📈 MLflow tracks every query's faithfulness, relevancy, and latency.")
    st.markdown("""
    **Start the MLflow UI:**
    ```bash
    mlflow ui --port 5000
    # or via Docker Compose — it's already running at http://localhost:5000
    ```

    **Metrics logged per query:**
    | Metric | Description |
    |---|---|
    | `faithfulness` | Fraction of answer claims supported by retrieved context |
    | `answer_relevancy` | How well the answer addresses the question |
    | `question_len` | Character length of the question |
    | `num_sources` | Number of chunks retrieved |
    """)
    st.link_button("Open MLflow UI", "http://localhost:5000")
