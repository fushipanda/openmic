"""RAG querying using LangChain."""

import json
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from openmic.storage import TRANSCRIPTS_DIR, format_transcript_title

INDEX_DIR = Path.home() / ".config" / "openmic" / "faiss_index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"


def get_embeddings():
    """Get embeddings model based on configured provider."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        # Default to OpenAI embeddings even with Anthropic LLM
        # since Anthropic doesn't have embeddings API
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()


def get_llm():
    """Get LLM based on configured provider."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022")


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
        """Build a manifest from the current transcripts directory."""
        if not TRANSCRIPTS_DIR.exists():
            return {"files": {}, "transcripts_dir": str(TRANSCRIPTS_DIR)}
        files = {}
        for f in sorted(TRANSCRIPTS_DIR.glob("*.md")):
            stat = f.stat()
            files[f.name] = {"mtime": stat.st_mtime, "size": stat.st_size}
        return {"files": files, "transcripts_dir": str(TRANSCRIPTS_DIR)}

    def _detect_changes(self, manifest: dict) -> tuple[list[Path], list[str]]:
        """Compare disk files vs manifest.

        Returns:
            (new_or_modified_files, deleted_filenames)
        """
        old_files = manifest.get("files", {})
        new_files = []
        deleted = []

        if not TRANSCRIPTS_DIR.exists():
            # All previously indexed files are effectively deleted
            return [], list(old_files.keys())

        current = {}
        for f in TRANSCRIPTS_DIR.glob("*.md"):
            current[f.name] = f

        # Find new or modified files
        for name, path in current.items():
            if name not in old_files:
                new_files.append(path)
            else:
                stat = path.stat()
                old = old_files[name]
                if stat.st_mtime != old.get("mtime") or stat.st_size != old.get("size"):
                    new_files.append(path)

        # Find deleted files
        for name in old_files:
            if name not in current:
                deleted.append(name)

        return new_files, deleted

    def _load_documents(self, paths: list[Path] | None = None) -> list:
        """Load transcript documents, optionally from specific paths."""
        if paths is not None:
            docs = []
            for p in paths:
                loader = TextLoader(str(p))
                docs.extend(loader.load())
            return docs

        if not TRANSCRIPTS_DIR.exists():
            return []

        loader = DirectoryLoader(
            str(TRANSCRIPTS_DIR),
            glob="*.md",
            loader_cls=TextLoader,
        )
        return loader.load()

    def _split_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)

    def _build_vectorstore(self, documents: list | None = None) -> FAISS | None:
        """Build the vector store from documents (or all transcripts if None)."""
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

        # If transcripts_dir changed, force full rebuild
        if manifest.get("transcripts_dir") != str(TRANSCRIPTS_DIR):
            index_exists = False
            manifest = {}

        if index_exists:
            try:
                embeddings = get_embeddings()
                store = FAISS.load_local(
                    str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                # Corrupted index — fall through to full rebuild
                store = None
                manifest = {}
                index_exists = False

        if index_exists and store is not None:
            new_files, deleted = self._detect_changes(manifest)
            has_modified = any(
                f.name in manifest.get("files", {}) for f in new_files
            )

            if deleted or has_modified:
                # Deletions or modifications require full rebuild
                store = self._build_vectorstore()
            elif new_files:
                # Only additions — embed just new files and merge
                new_docs = self._load_documents(new_files)
                if new_docs:
                    new_store = self._build_vectorstore(new_docs)
                    if new_store:
                        store.merge_from(new_store)
            # else: no changes — use loaded index as-is
        else:
            # No existing index — full build
            store = self._build_vectorstore()

        # Persist
        if store is not None:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            store.save_local(str(INDEX_DIR))
            self._save_manifest(self._build_manifest())

        return store

    def _build_chain(self):
        """Build the conversational retrieval chain."""
        if self._vectorstore is None:
            return None

        llm = get_llm()
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

        # Step 1: History-aware retriever — reformulates follow-up questions
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might "
             "reference context in the chat history, formulate a standalone "
             "question which can be understood without the chat history. "
             "Do NOT answer the question, just reformulate it if needed "
             "and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        # Step 2: QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that answers questions about "
             "meeting transcripts. Use the following retrieved context to "
             "answer the question. If you don't know the answer, say so. "
             "Keep answers concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def refresh(self) -> None:
        """Refresh the vector store with latest transcripts."""
        self._vectorstore = self._load_or_build_vectorstore()
        self._qa_chain = self._build_chain()

    def clear_chat_history(self) -> None:
        """Reset conversation history for a fresh chat."""
        self._chat_history = []

    def query(self, question: str) -> dict:
        """Query the transcripts with a question.

        Args:
            question: The question to answer

        Returns:
            Dict with 'answer' (str) and 'sources' (list of human-readable titles)
        """
        if self._vectorstore is None:
            self.refresh()

        if self._qa_chain is None:
            return {"answer": "No transcripts available to query.", "sources": []}

        result = self._qa_chain.invoke({
            "input": question,
            "chat_history": self._chat_history,
        })
        answer = result.get("answer", "Unable to generate answer.")

        # Extract unique source filenames and convert to readable titles
        sources = []
        seen = set()
        for doc in result.get("context", []):
            source_path = doc.metadata.get("source", "")
            if source_path and source_path not in seen:
                seen.add(source_path)
                stem = Path(source_path).stem
                ts = stem[:16]
                name = stem[17:] if len(stem) > 16 else None
                sources.append(format_transcript_title(ts, name))

        # Append to chat history
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        return {"answer": answer, "sources": sources}

    def query_file(self, question: str, transcript_path: Path) -> str:
        """Query a specific transcript file.

        Args:
            question: The question to answer
            transcript_path: Path to the specific transcript file

        Returns:
            The answer from the LLM based on the transcript content
        """
        loader = TextLoader(str(transcript_path))
        documents = loader.load()
        if not documents:
            return "Transcript is empty."

        splits = self._split_documents(documents)
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        llm = get_llm()

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that answers questions about "
             "meeting transcripts. Use the following retrieved context to "
             "answer the question. If you don't know the answer, say so. "
             "Keep answers concise.\n\n{context}"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        chain = create_retrieval_chain(retriever, question_answer_chain)

        result = chain.invoke({"input": question})
        return result.get("answer", "Unable to generate answer.")
