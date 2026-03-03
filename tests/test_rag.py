"""Integration test for RAG pipeline with mocked embeddings."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openmic.rag import TranscriptRAG, get_llm, get_embeddings, INDEX_DIR, MANIFEST_FILE


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


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Use a temp directory for the FAISS index."""
    idx = tmp_path / "faiss_index"
    idx.mkdir()
    monkeypatch.setattr("openmic.rag.INDEX_DIR", idx)
    monkeypatch.setattr("openmic.rag.MANIFEST_FILE", idx / "manifest.json")
    return idx


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
    def test_query_returns_answer(self, transcript_dir, index_dir):
        """Full RAG pipeline: load docs, build vectorstore, query with mocked chain."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "The deadline is September 30th.", "context": []}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()

            assert rag._vectorstore is not None
            assert rag._qa_chain is mock_chain

            result = rag.query("What is the database migration deadline?")
            assert result["answer"] == "The deadline is September 30th."
            assert result["sources"] == []
            mock_chain.invoke.assert_called_once()

    def test_query_returns_sources(self, transcript_dir, index_dir):
        """Query returns human-readable source titles from context documents."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"source": str(transcript_dir / "2025-06-15_14-30.md")}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"source": str(transcript_dir / "2025-06-16_10-00.md")}
        mock_chain.invoke.return_value = {
            "answer": "The answer.",
            "context": [mock_doc1, mock_doc2, mock_doc1],  # duplicate
        }

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()
            result = rag.query("test?")
            assert result["answer"] == "The answer."
            assert len(result["sources"]) == 2  # deduped
            assert "Meeting Transcript" in result["sources"][0]

    def test_query_triggers_refresh(self, transcript_dir, index_dir):
        """First query auto-refreshes the vectorstore."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Answer.", "context": []}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            assert rag._vectorstore is None

            result = rag.query("test?")
            # Should have auto-refreshed
            assert rag._vectorstore is not None
            assert result["answer"] == "Answer."

    def test_query_no_transcripts(self, tmp_path, monkeypatch, index_dir):
        """Query with no transcripts returns informative message."""
        empty_dir = tmp_path / "transcripts"
        empty_dir.mkdir()
        monkeypatch.setattr("openmic.rag.TRANSCRIPTS_DIR", empty_dir)

        fake_embeddings = FakeEmbeddings()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm"):
            rag = TranscriptRAG()
            result = rag.query("anything")
            assert result["answer"] == "No transcripts available to query."
            assert result["sources"] == []

    def test_refresh_rebuilds(self, transcript_dir, index_dir):
        """Calling refresh rebuilds the vectorstore."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()
            first_store = rag._vectorstore

            # Add another transcript
            (transcript_dir / "2025-07-01_09-00.md").write_text(
                "# Meeting Transcript\n\n**Speaker 1:** New content.\n\n"
            )
            rag.refresh()
            # New file triggers incremental merge, but store object changes
            assert rag._vectorstore is not None

    def test_nonexistent_transcripts_dir(self, tmp_path, monkeypatch, index_dir):
        """If transcripts dir doesn't exist, returns empty message."""
        monkeypatch.setattr("openmic.rag.TRANSCRIPTS_DIR", tmp_path / "nonexistent")

        with patch("openmic.rag.get_embeddings"), \
             patch("openmic.rag.get_llm"):
            rag = TranscriptRAG()
            result = rag.query("anything")
            assert result["answer"] == "No transcripts available to query."
            assert result["sources"] == []

    def test_query_missing_answer_key(self, transcript_dir, index_dir):
        """Query handles missing 'answer' key in chain output."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {}  # no "answer" key

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()
            result = rag.query("test?")
            assert result["answer"] == "Unable to generate answer."
            assert result["sources"] == []

    def test_vectorstore_built_with_documents(self, transcript_dir, index_dir):
        """Vectorstore is built from transcript documents."""
        fake_embeddings = FakeEmbeddings()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain"):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            docs = rag._load_documents()
            assert len(docs) == 2  # two transcript files

            vs = rag._build_vectorstore(docs)
            assert vs is not None


class TestChatHistory:
    def test_chat_history_accumulates(self, transcript_dir, index_dir):
        """Chat history grows with each query."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Yes.", "context": []}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()
            rag.query("First question?")
            assert len(rag._chat_history) == 2  # HumanMessage + AIMessage

            rag.query("Follow-up?")
            assert len(rag._chat_history) == 4

            # Verify chat_history was passed to chain
            last_call = mock_chain.invoke.call_args
            assert "chat_history" in last_call[0][0]

    def test_clear_chat_history(self, transcript_dir, index_dir):
        """clear_chat_history resets the conversation."""
        fake_embeddings = FakeEmbeddings()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Yes.", "context": []}

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain", return_value=mock_chain):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()
            rag.query("Question?")
            assert len(rag._chat_history) == 2

            rag.clear_chat_history()
            assert len(rag._chat_history) == 0


class TestPersistentIndex:
    def test_manifest_save_and_load(self, index_dir):
        """Manifest can be saved and loaded."""
        rag = TranscriptRAG()
        manifest = {"files": {"test.md": {"mtime": 123.0, "size": 100}}, "transcripts_dir": "/tmp"}
        rag._save_manifest(manifest)

        loaded = rag._load_manifest()
        assert loaded == manifest

    def test_manifest_corrupt_returns_empty(self, index_dir):
        """Corrupt manifest returns empty dict."""
        manifest_path = index_dir / "manifest.json"
        manifest_path.write_text("not json")

        rag = TranscriptRAG()
        assert rag._load_manifest() == {}

    def test_detect_changes_new_files(self, transcript_dir, index_dir):
        """Detects new files not in manifest."""
        rag = TranscriptRAG()
        manifest = {"files": {}, "transcripts_dir": str(transcript_dir)}

        new_files, deleted = rag._detect_changes(manifest)
        assert len(new_files) == 2
        assert len(deleted) == 0

    def test_detect_changes_deleted_files(self, transcript_dir, index_dir):
        """Detects files in manifest that no longer exist on disk."""
        rag = TranscriptRAG()
        manifest = {
            "files": {
                "2025-06-15_14-30.md": {"mtime": 0, "size": 0},
                "deleted.md": {"mtime": 0, "size": 0},
            },
            "transcripts_dir": str(transcript_dir),
        }

        new_files, deleted = rag._detect_changes(manifest)
        assert "deleted.md" in deleted
        # 2025-06-15_14-30.md has different mtime/size, so it's "modified" (in new_files)
        # 2025-06-16_10-00.md is truly new
        new_names = [f.name for f in new_files]
        assert "2025-06-16_10-00.md" in new_names

    def test_detect_changes_no_changes(self, transcript_dir, index_dir):
        """No changes detected when manifest matches disk."""
        rag = TranscriptRAG()
        manifest = rag._build_manifest()

        new_files, deleted = rag._detect_changes(manifest)
        assert len(new_files) == 0
        assert len(deleted) == 0

    def test_index_persisted_to_disk(self, transcript_dir, index_dir):
        """After refresh, index files are saved to disk."""
        fake_embeddings = FakeEmbeddings()

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain"):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()

            assert (index_dir / "index.faiss").exists()
            assert (index_dir / "index.pkl").exists()
            assert (index_dir / "manifest.json").exists()

            manifest = json.loads((index_dir / "manifest.json").read_text())
            assert "2025-06-15_14-30.md" in manifest["files"]
            assert "2025-06-16_10-00.md" in manifest["files"]

    def test_incremental_update(self, transcript_dir, index_dir):
        """Adding a new file only embeds the new file (incremental merge)."""
        fake_embeddings = FakeEmbeddings()
        embed_call_count = [0]
        original_embed_docs = fake_embeddings.embed_documents

        def counting_embed(texts):
            embed_call_count[0] += len(texts)
            return original_embed_docs(texts)

        fake_embeddings.embed_documents = counting_embed

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain"):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()  # Full build
            first_count = embed_call_count[0]

            # Add a new transcript
            (transcript_dir / "2025-07-01_09-00.md").write_text(
                "# Meeting Transcript\n\n**Speaker 1:** New content.\n\n"
            )

            embed_call_count[0] = 0
            rag.refresh()  # Should only embed the new file
            # The new file is small (1 chunk), so embed count should be much less than full rebuild
            assert embed_call_count[0] < first_count

    def test_loaded_index_reused_no_changes(self, transcript_dir, index_dir):
        """When nothing changed, loaded index is reused without re-embedding."""
        fake_embeddings = FakeEmbeddings()
        embed_call_count = [0]
        original_embed_docs = fake_embeddings.embed_documents

        def counting_embed(texts):
            embed_call_count[0] += len(texts)
            return original_embed_docs(texts)

        fake_embeddings.embed_documents = counting_embed

        with patch("openmic.rag.get_embeddings", return_value=fake_embeddings), \
             patch("openmic.rag.get_llm") as mock_get_llm, \
             patch("openmic.rag.create_retrieval_chain"):
            mock_get_llm.return_value = MagicMock()

            rag = TranscriptRAG()
            rag.refresh()  # Full build
            first_count = embed_call_count[0]
            assert first_count > 0

            embed_call_count[0] = 0
            rag2 = TranscriptRAG()
            rag2.refresh()  # Should load from disk, zero embedding calls
            assert embed_call_count[0] == 0


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
