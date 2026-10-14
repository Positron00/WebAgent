"""IMP-091: LDA skips huge documents and caps training passes."""
from unittest.mock import patch

from app.agents.document_extraction_agent import DocumentExtractionAgent
from app.core.config import settings


def test_lda_skips_when_document_too_large(monkeypatch):
    monkeypatch.setattr(settings, "LDA_MAX_CHARS", 100)
    agent = DocumentExtractionAgent()
    text = "word " * 200

    result = agent.extract_with_lda(text)

    assert result["topics_skipped"] == "document too large"
    assert result["topics"] == []


def test_lda_passes_capped_at_five():
    agent = DocumentExtractionAgent()
    agent.lda_params["passes"] = 15
    tiny = "short document about finance and markets. " * 3

    with patch.object(
        agent,
        "_preprocess_text",
        return_value=(["one sentence."], [["one", "sentence"]]),
    ):
        with patch("app.agents.document_extraction_agent.LdaModel") as lda_cls:
            lda_cls.return_value = lda_cls.return_value
            with patch("app.agents.document_extraction_agent.CoherenceModel") as coh_cls:
                coh_cls.return_value.get_coherence.return_value = 0.0
                lda_cls.return_value.print_topics.return_value = []
                lda_cls.return_value.get_document_topics.return_value = []

                agent.extract_with_lda(tiny)

    assert lda_cls.call_args.kwargs["passes"] == 5
