import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, Document  # type: ignore
# ChatMemoryBuffer location changed across llama_index versions; try both
try:
    from llama_index.memory.chat_memory import ChatMemoryBuffer  # type: ignore
except Exception:
    try:
        from llama_index.memory import ChatMemoryBuffer  # type: ignore
    except Exception:
        # Minimal fallback to avoid import errors in environments without llama_index memory module
        class ChatMemoryBuffer:
            def __init__(self, token_limit: int = 3000):
                self.token_limit = token_limit

            @classmethod
            def from_defaults(cls, token_limit: int = 3000):
                return cls(token_limit=token_limit)

try:
    from llama_index.llms.openai import OpenAI as LlamaOpenAI  # type: ignore
except Exception:
    from llama_index.llm_predictor.openai import OpenAI as LlamaOpenAI  # type: ignore
try:
    from llama_index.core.callbacks import CallbackManager  # type: ignore
except Exception:
    from llama_index.callbacks import CallbackManager  # type: ignore
try:
    from llama_index.core.node_parser import SentenceSplitter  # type: ignore
except Exception:
    from llama_index.node_parser import SentenceSplitter  # type: ignore
try:
    from llama_index.core.storage.storage_context import StorageContext  # type: ignore
except Exception:
    from llama_index.storage.storage_context import StorageContext  # type: ignore
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot mit LlamaIndex")
st.caption("Lade eigene Dokumente hoch und stelle Fragen. Antworten enthalten Quellenhinweise.")

# API-Key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Bitte trage deinen OpenAI API Key ein, um fortzufahren.", icon="🗝️")
    st.stop()

# Model- und Parameterwahl
col1, col2, col3 = st.columns(3)
with col1:
    model_name = st.selectbox(
        "OpenAI Modell",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"],
        index=0
    )
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
with col3:
    top_k = st.slider("Top-K Dokumente", 1, 10, 4, 1)

# LlamaIndex Settings
Settings.llm = LlamaOpenAI(model=model_name, api_key=openai_api_key, temperature=temperature)
Settings.callback_manager = CallbackManager([])
Settings.embed_model = None  # Falls du ein eigenes Embedding-Modell setzen willst, hier konfigurieren
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

# Session State init
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "memory" not in st.session_state:
    # Begrenzt Kontextlänge, damit der Chat nicht explodiert
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("📄 Dokumente hochladen")
uploaded_files = st.file_uploader(
    "Unterstützt: PDF, TXT, MD",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

def build_index_from_uploads(files):
    tmpdir = tempfile.mkdtemp()
    paths = []
    for f in files:
        path = os.path.join(tmpdir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        paths.append(path)

    # Reader lädt automatisch PDFs/TXTs/MDs
    documents = SimpleDirectoryReader(tmpdir, recursive=True).load_data()
    # Optional: Metadaten anreichern
    for d in documents:
        d.metadata = d.metadata or {}
        d.metadata["source"] = d.metadata.get("file_name") or d.metadata.get("filename") or d.metadata.get("source") or "upload"

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("📚 Index erstellen/aktualisieren", type="primary", disabled=not uploaded_files):
        with st.spinner("Baue Index..."):
            st.session_state.index = build_index_from_uploads(uploaded_files)
            st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                chat_mode="context",
                memory=st.session_state.memory,
                system_prompt=(
                    "Du bist ein hilfreicher Assistent. Antworte präzise. "
                    "Wenn Informationen nicht im Kontext sind, sage das offen. "
                    "Gib, wenn möglich, knappe Quellenhinweise (Titel/Datei)."
                ),
                similarity_top_k=top_k,
            )
        st.success("Index fertig.")
with col_b:
    if st.button("🧹 Chat zurücksetzen"):
        st.session_state.messages = []
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        if st.session_state.index:
            st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                chat_mode="context",
                memory=st.session_state.memory,
                similarity_top_k=top_k,
            )
        st.rerun()

st.divider()

# Chatbereich
chat_disabled = st.session_state.chat_engine is None
if chat_disabled:
    st.info("Lade Dokumente hoch und klicke auf 'Index erstellen', um loszulegen.")
else:
    # Bisherige Nachrichten anzeigen
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Frage zu deinen Dokumenten stellen..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                # Antwort mit Quellen
                response = st.session_state.chat_engine.chat(prompt)
                answer_text = response.response

                # Quellen (citations) extrahieren
                sources = []
                try:
                    for src in response.source_nodes[:top_k]:
                        meta = src.node.metadata or {}
                        name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "Quelle"
                        sources.append(f"- {name}")
                except Exception:
                    pass

                if sources:
                    final = answer_text + "\n\n**Quellen:**\n" + "\n".join(sources)
                else:
                    final = answer_text

                st.markdown(final)
                st.session_state.messages.append({"role": "assistant", "content": final})
