"""Internal research keeps only relevant chunks and seeds sample documents only when asked."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.agents import internal_research
from app.agents.internal_research import InternalResearchAgent, RAGRetriever, filter_by_relevance
from app.models.task import WorkflowState

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _doc(text, score=None, source="sample_docs/ai_ethics_guidelines.txt"):
    doc = {"page_content": text, "metadata": {"source": source}, "id": text}
    if score is not None:
        doc["score"] = score
    return doc


def test_filter_by_relevance_keeps_scores_at_or_above_the_cutoff():
    docs = [_doc("a", 4.0), _doc("b", -5.0), _doc("c", -7.6), _doc("unscored")]
    assert [d["id"] for d in filter_by_relevance(docs, -5.0)] == ["a", "b"]


@pytest.mark.parametrize("env, seed", [("dev", True), ("uat", False), ("prod", False)])
def test_sample_documents_are_seeded_only_in_dev(env, seed):
    cfg = yaml.safe_load((CONFIG / f"{env}.yaml").read_text())["agents"]["internal_research"]
    assert cfg["seed_sample_documents"] is seed
    assert cfg["min_relevance"] == -5.0


def _agents_config(seed=False, min_relevance=-5.0):
    return SimpleNamespace(internal_research=SimpleNamespace(
        seed_sample_documents=seed, min_relevance=min_relevance, max_documents=20))


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [False, True])
async def test_empty_knowledge_base_is_seeded_only_when_configured(monkeypatch, seed):
    calls = []

    class FakeIngestion:
        async def run_document_ingestion(self):
            calls.append("ingest")
            return {"status": "error", "error": "test"}

    monkeypatch.setattr(internal_research, "get_agents_config", lambda: _agents_config(seed=seed))
    monkeypatch.setattr("app.services.document_ingestion_service.get_document_ingestion_service",
                        lambda: FakeIngestion())
    retriever = object.__new__(RAGRetriever)
    retriever.vector_db_document_count = 0  # known empty
    assert await retriever.ensure_vector_db_has_documents() is False
    assert calls == (["ingest"] if seed else [])


class _Retriever:
    def __init__(self, docs):
        self.docs = docs

    async def retrieve(self, query, k=20):
        return [dict(d) for d in self.docs]


class _Reranker:
    """Scores by a fixed table; the real one is the ms-marco cross-encoder."""

    initialized = True

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    async def rerank(self, query, documents, top_n=5):
        self.calls += 1
        for doc in documents:
            doc["score"] = self.scores[doc["id"]]
        return sorted(documents, key=lambda d: d["score"], reverse=True)[:top_n]


def _agent(docs, scores):
    agent = object.__new__(InternalResearchAgent)
    agent.config = {}
    agent.max_documents = 20
    agent.retriever = _Retriever(docs)
    agent.reranker = _Reranker(scores)
    return agent


def _state():
    return WorkflowState(query="Who won the 1998 World Cup?", context={"task_id": "task-relevance"}, inputs={})


@pytest.mark.asyncio
async def test_unrelated_question_gets_no_documents_and_no_guarded_text(monkeypatch):
    """Two demo chunks, fewer than rerank_top_n: they are still scored, and both fall below the cutoff."""
    monkeypatch.setattr(internal_research, "get_agents_config", lambda: _agents_config())
    agent = _agent([_doc("ethics chunk"), _doc("finance chunk")], {"ethics chunk": -11.2, "finance chunk": -8.6})
    prompts = []

    async def generate(self, prompt_inputs, state):
        prompts.append(prompt_inputs)
        return "findings"

    monkeypatch.setattr(InternalResearchAgent, "_generate_findings_with_retry", generate)
    state = await agent.run(_state())
    assert agent.reranker.calls == 1
    assert prompts == []  # nothing reached the prompt, so checkpoint 1 had nothing to decline
    assert state.agent_outputs["internal_research"]["documents"] == []


@pytest.mark.asyncio
async def test_relevant_chunks_reach_the_prompt_and_irrelevant_ones_do_not(monkeypatch):
    monkeypatch.setattr(internal_research, "get_agents_config", lambda: _agents_config())
    agent = _agent([_doc("relevant chunk", source="report.txt"), _doc("demo chunk")],
                   {"relevant chunk": 4.1, "demo chunk": -9.0})
    prompts = []

    async def generate(self, prompt_inputs, state):
        prompts.append(prompt_inputs)
        return "findings"

    monkeypatch.setattr(InternalResearchAgent, "_generate_findings_with_retry", generate)
    await agent.run(_state())
    assert len(prompts) == 1
    assert "relevant chunk" in prompts[0]["documents"]
    assert "demo chunk" not in prompts[0]["documents"]
