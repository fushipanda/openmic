"""RAG querying using LangChain."""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

from openmic.storage import TRANSCRIPTS_DIR


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

    def _load_documents(self) -> list:
        """Load all transcript documents."""
        if not TRANSCRIPTS_DIR.exists():
            return []

        loader = DirectoryLoader(
            str(TRANSCRIPTS_DIR),
            glob="*.md",
            loader_cls=TextLoader,
        )
        return loader.load()

    def _build_vectorstore(self) -> FAISS | None:
        """Build or rebuild the vector store from transcripts."""
        documents = self._load_documents()
        if not documents:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)

        embeddings = get_embeddings()
        return FAISS.from_documents(splits, embeddings)

    def refresh(self) -> None:
        """Refresh the vector store with latest transcripts."""
        self._vectorstore = self._build_vectorstore()
        if self._vectorstore:
            llm = get_llm()
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self._vectorstore.as_retriever(search_kwargs={"k": 4}),
            )

    def query(self, question: str) -> str:
        """Query the transcripts with a question.

        Args:
            question: The question to answer

        Returns:
            The answer from the LLM based on relevant transcript chunks
        """
        if self._vectorstore is None:
            self.refresh()

        if self._qa_chain is None:
            return "No transcripts available to query."

        result = self._qa_chain.invoke({"query": question})
        return result.get("result", "Unable to generate answer.")

    def query_file(self, question: str, transcript_path: Path) -> str:
        """Query a specific transcript file.

        Args:
            question: The question to answer
            transcript_path: Path to the specific transcript file

        Returns:
            The answer from the LLM based on the transcript content
        """
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(transcript_path))
        documents = loader.load()
        if not documents:
            return "Transcript is empty."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        llm = get_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        )
        result = qa_chain.invoke({"query": question})
        return result.get("result", "Unable to generate answer.")
