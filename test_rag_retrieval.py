from app import rag


class FakeCollection:
    def __init__(self) -> None:
        self.docs = [
            {
                "document": "Use STAR with a measurable result and clear personal ownership.",
                "metadata": {
                    "source": "data/rag/frameworks/answer_frameworks.md",
                    "doc_type": "framework",
                    "profession": "software_engineer",
                    "focus_area": "behavioral",
                },
                "distance": 0.12,
            },
            {
                "document": "Backend candidates should explain APIs, data models, reliability, and validation.",
                "metadata": {
                    "source": "data/kb/backend_developer.md",
                    "doc_type": "role",
                    "profession": "backend_developer",
                    "focus_area": "technical",
                },
                "distance": 0.2,
            },
            {
                "document": "Company rubrics reward structured tradeoffs and customer impact.",
                "metadata": {
                    "source": "data/rag/rubrics/company_rubrics.md",
                    "doc_type": "rubric",
                    "profession": "",
                    "company": "google",
                },
                "distance": 0.18,
            },
        ]

    def query(self, query_texts, n_results, where=None):
        rows = self.docs
        if where:
            rows = [row for row in rows if all(row["metadata"].get(key) == value for key, value in where.items())]
            if not rows:
                raise RuntimeError("no filtered rows")
        rows = rows[:n_results]
        return {
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[row["distance"] for row in rows]],
        }


def test_retrieve_for_hint_returns_quality_and_evidence(monkeypatch):
    requested_collections: list[str] = []

    def fake_get_collection(name="knowledge_base"):
        requested_collections.append(name)
        return FakeCollection()

    monkeypatch.setattr(rag, "get_collection", fake_get_collection)

    result = rag.retrieve_for_hint(
        profession="Software Engineer",
        question="Tell me about a project where you made impact.",
        config={"focus_area": "Behavioral", "target_company": "Google"},
    )

    assert result.context
    assert result.evidence
    assert result.quality["label"] in {"low", "medium", "high"}
    assert result.quality["layer_count"] >= 1
    assert result.quality["collection_count"] >= 1
    assert result.evidence[0]["layer"]
    assert result.evidence[0]["collection"]
    assert result.evidence[0]["doc_type"] in {"framework", "rubric", "role"}
    assert "RAG support" in result.summary
    assert "answer_kb" in requested_collections
    assert "knowledge_base" in requested_collections


def test_legacy_retrieve_context_still_returns_tuple(monkeypatch):
    monkeypatch.setattr(rag, "get_collection", lambda name="knowledge_base": FakeCollection())

    context, evidence = rag.retrieve_context("Backend Developer", "api reliability validation", return_evidence=True)

    assert "Backend candidates" in context
    assert evidence[0]["profession"] == "backend_developer"
    assert "retrieval_quality" not in evidence[0]


def test_story_candidate_rerank_adds_private_match_metadata():
    stories = [
        {"id": 1, "title": "Payment API outage", "question": "Tell me about impact", "answer": "Reduced latency 30%."},
        {"id": 2, "title": "Design critique", "question": "Tell me about design", "answer": "Improved prototype clarity."},
    ]

    ranked = rag.rank_story_candidates("api latency impact", stories)

    assert ranked[0]["id"] == 1
    assert ranked[0]["retrieval_match"]["hybrid_score"] > 0


def test_user_memory_and_graph_are_retrieval_sources(monkeypatch):
    monkeypatch.setattr(rag, "get_collection", lambda name="knowledge_base": FakeCollection())

    result = rag.retrieve(
        rag.RetrievalQuery(
            purpose="question_generation",
            profession="Backend Developer",
            query="Ask about API reliability and metrics",
            user_memory=[
                {
                    "id": 10,
                    "memory_type": "skill_gap",
                    "content": "User should add measurable metrics and validation details.",
                    "score": 0.8,
                }
            ],
        )
    )

    layers = {item["layer"] for item in result.evidence}
    assert "user_memory_kb" in layers
    assert any(item.get("doc_type") == "graph_edge" for item in result.evidence)


def test_retrieval_evaluation_and_citations_are_explainable():
    evidence = [
        {
            "source": "framework.md",
            "layer": "answer_kb",
            "doc_type": "framework",
            "hybrid_score": 0.8,
            "relevance_label": "high",
            "content": "STAR answers need measurable impact and clear ownership.",
            "preview": "STAR answers need measurable impact and clear ownership.",
        }
    ]

    citations = rag.build_citations(evidence)
    evaluation = rag.evaluate_retrieval(evidence, answer_text="I owned the work", feedback_text="Add measurable impact.")

    assert citations[0]["id"] == "C1"
    assert evaluation["retrieval_precision_proxy"] == 1.0
    assert evaluation["low_confidence"] is False
