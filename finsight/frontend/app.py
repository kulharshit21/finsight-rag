"""
FinSight Streamlit Frontend

Pages:
  1. Upload Documents — drag-and-drop PDFs
  2. Ask Questions — chat interface with source citations
  3. Document Library — list + delete ingested docs
  4. Evaluation Dashboard — faithfulness/relevancy scores over time
"""

import streamlit as st
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 FinSight")
st.sidebar.caption("RAG chatbot for financial documents")
page = st.sidebar.radio("Navigate", ["Chat", "Upload", "Library", "Dashboard"])

# ── Page: Upload ──────────────────────────────────────────────────────────────
if page == "Upload":
    st.title("Upload Financial Documents")
    st.caption("Supports PDF and TXT. Annual reports, 10-Ks, earnings transcripts.")

    uploaded = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest Documents", type="primary"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}..."):
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (f.name, f.read(), f.type)},
                )
            if resp.status_code == 200:
                data = resp.json()
                if data["status"] == "already_exists":
                    st.warning(f"⚠️ {f.name} already ingested ({data['chunks_created']} chunks)")
                else:
                    st.success(f"✅ {f.name} — {data['chunks_created']} chunks created")
            else:
                st.error(f"❌ {f.name} failed: {resp.text}")

# ── Page: Chat ────────────────────────────────────────────────────────────────
elif page == "Chat":
    st.title("Ask FinSight")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Faithfulness", f"{meta['faithfulness_score']:.0%}")
                col2.metric("Relevancy", f"{meta['answer_relevancy_score']:.0%}")
                col3.metric("Fraud Risk", f"{meta['fraud_risk_score']:.0%}" if meta['fraud_risk_score'] is not None else "N/A")
                col4.metric("Latency", f"{meta['latency_ms']}ms")

                if meta["fraud_flags"]:
                    st.warning("🚨 Fraud flags detected: " + " · ".join(meta["fraud_flags"]))

                with st.expander(f"📄 {len(meta['sources'])} sources"):
                    for s in meta["sources"]:
                        st.markdown(f"**{s['filename']}** · Page {s['page']}")
                        st.caption(s["excerpt"])

    # Input
    question = st.chat_input("Ask about your financial documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question},
                )
            if resp.status_code == 200:
                data = resp.json()
                st.markdown(data["answer"])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Faithfulness", f"{data['faithfulness_score']:.0%}")
                col2.metric("Relevancy", f"{data['answer_relevancy_score']:.0%}")
                col3.metric("Fraud Risk", f"{data['fraud_risk_score']:.0%}" if data['fraud_risk_score'] is not None else "N/A")
                col4.metric("Latency", f"{data['latency_ms']}ms")

                if data["fraud_flags"]:
                    st.warning("🚨 " + " · ".join(data["fraud_flags"]))

                with st.expander(f"📄 {len(data['sources'])} sources"):
                    for s in data["sources"]:
                        st.markdown(f"**{s['filename']}** · Page {s['page']}")
                        st.caption(s["excerpt"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": data,
                })
            else:
                st.error(f"API error: {resp.text}")

# ── Page: Library ─────────────────────────────────────────────────────────────
elif page == "Library":
    st.title("Document Library")
    resp = requests.get(f"{API_BASE}/documents")
    if resp.status_code == 200:
        data = resp.json()
        st.caption(f"{data['total']} documents ingested")
        for doc in data["documents"]:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**{doc['filename']}**  \n`{doc['doc_id']}`")
            col2.markdown(f"{doc['chunk_count']} chunks")
            if col3.button("Delete", key=doc["doc_id"]):
                del_resp = requests.delete(f"{API_BASE}/documents/{doc['doc_id']}")
                if del_resp.status_code == 200:
                    st.success("Deleted")
                    st.rerun()
    else:
        st.error("Could not fetch documents")

# ── Page: Dashboard ───────────────────────────────────────────────────────────
elif page == "Dashboard":
    st.title("Evaluation Dashboard")
    st.info("Connect MLflow at http://localhost:5000 to see query-level metrics over time.")
    st.markdown("""
    **Metrics tracked per query:**
    - `faithfulness` — is the answer grounded in retrieved context?
    - `answer_relevancy` — does the answer address the question?
    - `fraud_risk_score` — XGBoost fraud signal 0–1
    - `latency_ms` — end-to-end response time

    Run `mlflow ui` in your terminal to explore the full experiment history.
    """)
