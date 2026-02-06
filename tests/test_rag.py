"""Integration test for RAG pipeline with mocked embeddings."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openmic.rag import TranscriptRAG, get_llm, get_embeddings


@pytest.fixture
def transcript_dir(tmp_path, monkeypatch):
    """Create a temp transcripts dir with sample data."""
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr("openmic.rag.TRANSCRIPTS_DIR", transcripts)

    # Write sample transcripts
    (transcripts / "2025-06-15_14-30.md").write_text(
        "# Meeting Transcript - 2025-06-15_14-30\n\n"
        "**Speaker 1:** We need to migrate the database to PostgreSQL by end of Q3.\n\n"
        "**Speaker 2:** I'll handle the schema migration. The deadline is September 30th.\n\n"
    )
    (transcripts / "2025-06-16_10-00.md").write_text(
        "# Meeting Transcript - 2025-06-16_10-00\n\n"
        "**Speaker 1:** The new API endpoint for user profiles is ready for review.\n\n"
        "**Speaker 2:** Great, I'll review it this afternoon.\n\n"
    )
    return transcripts


class FakeEmbeddings:
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 64
        for i, ch in enumerate(text[:64]):
            vec[i % 64] += ord(ch) / 1000.0
        magnitude = sum(v * v for v in vec) ** 0.5
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec


class TestTranscriptRAG:
    def test_query_returns_answer(self, transcript_dir):
        """Full RAG pipeline: load docs, build vectorstore, query with mocked chain."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"result": "The deadline is September 30th."}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.RetrievalQA") as mock_qa_cls:
            mock_qa_cls.from_chain_type.return_value = mock_chain

            rag = TranscriptRAG()
            rag.refresh()

            assert rag._vectorstore is not None
            assert rag._qa_chain is mock_chain

            result = rag.query("What is the database migration deadline?")
            assert result == "The deadline is September 30th."
            mock_chain.invoke.assert_called_once()

    def test_query_triggers_refresh(self, transcript_dir):
        """First query auto-refreshes the vectorstore."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"result": "Answer."}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.RetrievalQA") as mock_qa_cls:
            mock_qa_cls.from_chain_type.return_value = mock_chain

            rag = TranscriptRAG()
            assert rag._vectorstore is None

            result = rag.query("test?")
            # Should have auto-refreshed
            assert rag._vectorstore is not None
            assert result == "Answer."

    def test_query_no_transcripts(self, tmp_path, monkeypatch):
        """Query with no transcripts returns informative message."""
        empty_dir = tmp_path / "transcripts"
        empty_dir.mkdir()
        monkeypatch.setattr("openmic.rag.TRANSCRIPTS_DIR", empty_dir)

        fake_embeddings = FakeEmbeddings()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm"):
            rag = TranscriptRAG()
            result = rag.query("anything")
            assert result == "No transcripts available to query."

    def test_refresh_rebuilds(self, transcript_dir):
        """Calling refresh rebuilds the vectorstore."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.RetrievalQA") as mock_qa_cls:
            mock_qa_cls.from_chain_type.return_value = mock_chain

            rag = TranscriptRAG()
            rag.refresh()
            first_store = rag._vectorstore

            # Add another transcript
            (transcript_dir / "2025-07-01_09-00.md").write_text(
                "# Meeting Transcript\n\n**Speaker 1:** New content.\n\n"
            )
            rag.refresh()
            assert rag._vectorstore is not first_store

    def test_nonexistent_transcripts_dir(self, tmp_path, monkeypatch):
        """If transcripts dir doesn't exist, returns empty message."""
        monkeypatch.setattr("openmic.rag.TRANSCRIPTS_DIR", tmp_path / "nonexistent")

        with patch("openmic.rag.get_embeddings"), \
             patch("openmic.rag.get_llm"):
            rag = TranscriptRAG()
            result = rag.query("anything")
            assert result == "No transcripts available to query."

    def test_query_missing_result_key(self, transcript_dir):
        """Query handles missing 'result' key in chain output."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {}  # no "result" key

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.RetrievalQA") as mock_qa_cls:
            mock_qa_cls.from_chain_type.return_value = mock_chain

            rag = TranscriptRAG()
            rag.refresh()
            result = rag.query("test?")
            assert result == "Unable to generate answer."

    def test_vectorstore_built_with_documents(self, transcript_dir):
        """Vectorstore is built from transcript documents."""
        fake_embeddings = FakeEmbeddings()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.RetrievalQA"):
            rag = TranscriptRAG()
            docs = rag._load_documents()
            assert len(docs) == 2  # two transcript files

            vs = rag._build_vectorstore()
            assert vs is not None


class TestGetLLM:
    def test_anthropic_provider(self, monkeypatch):
        """LLM_PROVIDER=anthropic returns ChatAnthropic."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        llm = get_llm()

        from langchain_anthropic import ChatAnthropic
        assert isinstance(llm, ChatAnthropic)

    def test_openai_provider(self, monkeypatch):
        """LLM_PROVIDER=openai returns ChatOpenAI."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        llm = get_llm()

        from langchain_openai import ChatOpenAI
        assert isinstance(llm, ChatOpenAI)

    def test_default_provider(self, monkeypatch):
        """Default provider (no env var) returns ChatAnthropic."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        llm = get_llm()

        from langchain_anthropic import ChatAnthropic
        assert isinstance(llm, ChatAnthropic)


class TestGetEmbeddings:
    def test_always_returns_openai_embeddings(self, monkeypatch):
        """Embeddings always use OpenAI regardless of LLM_PROVIDER."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        for provider in ["anthropic", "openai"]:
            monkeypatch.setenv("LLM_PROVIDER", provider)
            emb = get_embeddings()

            from langchain_openai import OpenAIEmbeddings
            assert isinstance(emb, OpenAIEmbeddings)
