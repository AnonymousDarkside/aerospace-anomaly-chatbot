"""
Streamlit Frontend — Aerospace Anomaly & Telemetry Assistant

Dark-themed chat interface with sidebar configuration for LLM
provider selection and API key input. Communicates with the
FastAPI backend over HTTP.

Usage:
    streamlit run app/ui.py
"""

import os

import streamlit as st
import requests
from pathlib import Path

# ── Page Config ──
st.set_page_config(
    page_title="Aerospace Telemetry Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Custom Dark Theme CSS ──
st.markdown(
    """
    <style>
    /* Dark background overrides */
    .stApp {
        background-color: #0e1117;
    }
    /* Chat message styling */
    .context-card {
        background-color: #1a1d24;
        border: 1px solid #2d333b;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    .context-header {
        color: #58a6ff;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .context-snippet {
        color: #8b949e;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .score-badge {
        display: inline-block;
        background-color: #1f6feb33;
        color: #58a6ff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-bar {
        background-color: #161b22;
        border: 1px solid #2d333b;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 16px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar Configuration ──
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    provider = st.selectbox(
        "LLM Provider",
        options=["openrouter", "google"],
        format_func=lambda x: "OpenRouter" if x == "openrouter" else "Google AI Studio",
    )

    st.markdown("#### API Keys")
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get a key at openrouter.ai/keys",
    )
    google_key = st.text_input(
        "Google AI Studio API Key",
        type="password",
        placeholder="AIza...",
        help="Get a key at aistudio.google.com/apikey",
    )

    st.markdown("---")
    st.markdown("#### Advanced")
    model_override = st.text_input(
        "Model Override (optional)",
        placeholder="e.g. anthropic/claude-sonnet-4",
        help="Leave blank to use provider default.",
    )

    st.markdown("---")

    # Health check
    if st.button("Check Backend Status"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            data = resp.json()
            vlm = "✅" if data.get("vlm_loaded") else "❌"
            qdrant = "✅" if data.get("qdrant_connected") else "❌"
            st.markdown(
                f'<div class="status-bar">'
                f"VLM: {vlm} &nbsp;|&nbsp; Qdrant: {qdrant} &nbsp;|&nbsp; "
                f'Status: <strong>{data.get("status", "unknown")}</strong>'
                f"</div>",
                unsafe_allow_html=True,
            )
        except requests.ConnectionError:
            st.error("Cannot reach backend. Is `uvicorn app.main:app` running?")

    st.markdown(
        "<small style='color:#8b949e'>"
        "API keys are sent per-request and never stored on the server."
        "</small>",
        unsafe_allow_html=True,
    )


# ── Helper ──
def get_active_api_key() -> str | None:
    """Return the API key for the currently selected provider."""
    if provider == "openrouter":
        return openrouter_key or None
    return google_key or None


# ── Main Chat Interface ──
st.markdown("# 🛰️ Aerospace Anomaly & Telemetry Assistant")
st.caption("Ask questions about aerospace systems engineering, fault management, and telemetry analysis.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("contexts"):
            _render_contexts(msg["contexts"]) if False else None  # rendered below


def render_contexts(contexts: list[dict]):
    """Display retrieved document contexts with images."""
    if not contexts:
        return
    st.markdown("**Retrieved References:**")
    cols = st.columns(min(len(contexts), 3))
    for i, ctx in enumerate(contexts):
        with cols[i % 3]:
            st.markdown(
                f'<div class="context-card">'
                f'<div class="context-header">{ctx["source_name"]}</div>'
                f'Page {ctx["page_number"]}/{ctx["total_pages"]} '
                f'<span class="score-badge">{ctx["score"]:.3f}</span>'
                f'<div class="context-snippet">{ctx["text_snippet"][:200]}...</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            # Display the page image if available
            image_path = ctx.get("image_path", "")
            if image_path and Path(image_path).exists():
                st.image(image_path, use_container_width=True)
            elif image_path:
                # Try fetching from the backend image endpoint
                parts = Path(image_path).parts
                if len(parts) >= 2:
                    source_id = parts[-2]
                    filename = parts[-1]
                    st.image(
                        f"{BACKEND_URL}/image/{source_id}/{filename}",
                        use_container_width=True,
                    )


# Re-render history contexts
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and msg.get("contexts"):
        render_contexts(msg["contexts"])

# Chat input
if prompt := st.chat_input("Ask about aerospace telemetry, anomalies, or systems engineering..."):
    api_key = get_active_api_key()
    if not api_key:
        st.error(f"Please enter your {'OpenRouter' if provider == 'openrouter' else 'Google AI Studio'} API key in the sidebar.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching aerospace documents and generating response..."):
            try:
                payload = {
                    "query": prompt,
                    "provider": provider,
                    "api_key": api_key,
                }
                if model_override:
                    payload["model"] = model_override

                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=payload,
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    contexts = data.get("contexts", [])
                    model_used = data.get("model_used", "unknown")

                    st.markdown(answer)
                    st.caption(f"Model: `{model_used}` via `{provider}`")
                    render_contexts(contexts)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "contexts": contexts,
                    })
                elif resp.status_code == 502:
                    detail = resp.json().get("detail", "LLM provider error")
                    st.error(f"LLM Error: {detail}")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the backend. "
                    "Make sure the FastAPI server is running:\n\n"
                    "`uvicorn app.main:app --host 0.0.0.0 --port 8000`"
                )
            except requests.Timeout:
                st.error("Request timed out. The backend may be processing a large query.")
