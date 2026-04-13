"""RAG querying using LangChain."""

import json
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from openmic.session import SESSIONS_DIR, session_to_text, list_sessions, get_session_meta

INDEX_DIR = Path.home() / ".config" / "openmic" / "faiss_index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"


def get_embeddings():
    """Get embeddings model based on configured provider."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=embed_model, base_url=base_url)

    # All cloud providers use OpenAI embeddings (Anthropic has no embeddings API)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


def get_llm():
    """Get LLM based on configured provider."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    model = os.environ.get("LLM_MODEL")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini")
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash")
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "meta-llama/llama-3.3-70b-instruct",
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model or "llama3.2:3b", base_url=base_url)
    else:  # anthropic
        from langchain_anthropic import ChatAnthropic
        kwargs = {}
        if os.environ.get("LLM_EXTENDED_THINKING", "").lower() == "true":
            kwargs["model_kwargs"] = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20241022", **kwargs)


def _session_display_name(session_path: Path) -> str:
    """Return a human-readable display name for a session."""
    try:
        meta = get_session_meta(session_path)
        name = meta.get("name") or session_path.stem
    except Exception:
        name = session_path.stem
    return name.replace("_", " ").strip()


class TranscriptRAG:
    """RAG system for querying meeting transcripts."""

    def __init__(self) -> None:
        self._vectorstore: FAISS | None = None
        self._qa_chain = None
        self._chat_history: list = []

    def _load_manifest(self) -> dict:
        """Read manifest from disk. Returns empty dict if missing or corrupt."""
        if not MANIFEST_FILE.exists():
            return {}
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        """Write manifest to disk, creating directories as needed."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    def _build_manifest(self) -> dict:
        """Build a manifest from the current sessions directory."""
        if not SESSIONS_DIR.exists():
            return {"files": {}, "sessions_dir": str(SESSIONS_DIR)}
        files = {}
        for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
            stat = f.stat()
            files[f.name] = {"mtime": stat.st_mtime, "size": stat.st_size}
        return {"files": files, "sessions_dir": str(SESSIONS_DIR)}

    def _detect_changes(self, manifest: dict) -> tuple[list[Path], list[str]]:
        """Compare disk files vs manifest.

        Returns:
            (new_or_modified_files, deleted_filenames)
        """
        old_files = manifest.get("files", {})
        new_files = []
        deleted = []

        if not SESSIONS_DIR.exists():
            return [], list(old_files.keys())

        current = {}
        for f in SESSIONS_DIR.glob("*.jsonl"):
            current[f.name] = f

        for name, path in current.items():
            if name not in old_files:
                new_files.append(path)
            else:
                stat = path.stat()
                old = old_files[name]
                if stat.st_mtime != old.get("mtime") or stat.st_size != old.get("size"):
                    new_files.append(path)

        for name in old_files:
            if name not in current:
                deleted.append(name)

        return new_files, deleted

    def _load_documents(self, paths: list[Path] | None = None) -> list[Document]:
        """Load session documents as LangChain Documents.

        Each session JSONL becomes one Document whose page_content is the
        concatenated transcript text from all recordings in that session.
        """
        if paths is not None:
            target = paths
        elif SESSIONS_DIR.exists():
            target = list(SESSIONS_DIR.glob("*.jsonl"))
        else:
            return []

        docs = []
        for p in target:
            text = session_to_text(p)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(p), "session": p.stem},
                ))
        return docs

    def _split_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)

    def _build_vectorstore(self, documents: list | None = None) -> FAISS | None:
        """Build the vector store from documents (or all sessions if None)."""
        if documents is None:
            documents = self._load_documents()
        if not documents:
            return None

        splits = self._split_documents(documents)
        embeddings = get_embeddings()
        return FAISS.from_documents(splits, embeddings)

    def _load_or_build_vectorstore(self) -> FAISS | None:
        """Load persisted index or build/update as needed."""
        manifest = self._load_manifest()
        index_exists = (INDEX_DIR / "index.faiss").exists()

        # If sessions_dir changed, force full rebuild
        if manifest.get("sessions_dir") != str(SESSIONS_DIR):
            index_exists = False
            manifest = {}

        if index_exists:
            try:
                embeddings = get_embeddings()
                # SECURITY: allow_dangerous_deserialization is required by LangChain's
                # FAISS integration (it uses pickle). The index directory is protected
                # with 0o700 permissions to limit access to the current user only.
                store = FAISS.load_local(
                    str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                store = None
                manifest = {}
                index_exists = False

        if index_exists and store is not None:
            new_files, deleted = self._detect_changes(manifest)
            has_modified = any(
                f.name in manifest.get("files", {}) for f in new_files
            )

            if deleted or has_modified:
                store = self._build_vectorstore()
            elif new_files:
                new_docs = self._load_documents(new_files)
                if new_docs:
                    new_store = self._build_vectorstore(new_docs)
                    if new_store:
                        store.merge_from(new_store)
        else:
            store = self._build_vectorstore()

        if store is not None:
            INDEX_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            store.save_local(str(INDEX_DIR))
            self._save_manifest(self._build_manifest())

        return store

    def _build_chain(self):
        """Build the conversational retrieval chain."""
        if self._vectorstore is None:
            return None

        llm = get_llm()
        retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        )

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might "
             "reference context in the chat history, formulate a standalone "
             "question which can be understood without the chat history. "
             "Preserve specific names, dates, and details from the original question. "
             "Do NOT answer the question, just reformulate it if needed "
             "and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about meeting transcripts.\n\n"
             "Use the retrieved context below to answer. Follow these rules:\n"
             "1. If the answer is in the context, quote the relevant passage, then explain.\n"
             "2. If the context is partial, share what you found and note what's missing.\n"
             "3. If the context is insufficient, say \"I couldn't find that specifically —\" "
             "then describe what related information IS available, and suggest rephrasing.\n"
             "4. Never say just \"I don't know.\" Always offer next steps.\n"
             "5. Be specific about which session the information comes from.\n\n"
             "{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def refresh(self) -> None:
        """Refresh the vector store with latest sessions."""
        self._vectorstore = self._load_or_build_vectorstore()
        self._qa_chain = self._build_chain()

    def clear_chat_history(self) -> None:
        """Reset conversation history for a fresh chat."""
        self._chat_history = []

    def query(self, question: str) -> dict:
        """Query across all sessions with a question.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of session display names)
        """
        if self._vectorstore is None:
            self.refresh()

        if self._qa_chain is None:
            return {"answer": "No sessions available to query.", "sources": []}

        result = self._qa_chain.invoke({
            "input": question,
            "chat_history": self._chat_history,
        })
        answer = result.get("answer", "Unable to generate answer.")

        sources = []
        seen = set()
        for doc in result.get("context", []):
            source_path = doc.metadata.get("source", "")
            if source_path and source_path not in seen:
                seen.add(source_path)
                sources.append(_session_display_name(Path(source_path)))

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        return {"answer": answer, "sources": sources}

    def query_session(self, question: str, session_path: Path) -> str:
        """Query a single session JSONL file.

        Builds an in-memory FAISS from the session's transcript text and
        queries it directly without affecting the global vector store or
        chat history.

        Args:
            question: The question to answer.
            session_path: Path to the session JSONL file.

        Returns:
            The answer string from the LLM.
        """
        text = session_to_text(session_path)
        if not text.strip():
            return "This session has no transcript content yet."

        doc = Document(page_content=text, metadata={"source": str(session_path)})
        splits = self._split_documents([doc])
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        llm = get_llm()

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about meeting transcripts.\n\n"
             "Use the retrieved context below to answer. Follow these rules:\n"
             "1. If the answer is in the context, quote the relevant passage, then explain.\n"
             "2. If the context is partial, share what you found and note what's missing.\n"
             "3. If the context is insufficient, say \"I couldn't find that specifically —\" "
             "then describe what related information IS available, and suggest rephrasing.\n"
             "4. Never say just \"I don't know.\" Always offer next steps.\n\n"
             "{context}"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        )
        chain = create_retrieval_chain(retriever, question_answer_chain)
        result = chain.invoke({"input": question})
        return result.get("answer", "Unable to generate answer.")
