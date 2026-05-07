from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple, Optional, Iterable
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from .config import settings


DEFAULT_COLLECTION = "knowledge_base"
LEGACY_COLLECTION = DEFAULT_COLLECTION
RAG_LAYERS: dict[str, list[str]] = {
    "role_kb": ["role", "knowledge"],
    "company_kb": ["company", "rubric"],
    "question_kb": ["question_seed"],
    "answer_kb": ["framework", "anti_pattern", "story"],
    "cv_kb": ["cv_signal"],
    "evaluation_kb": ["rubric", "anti_pattern"],
    "roadmap_kb": ["drill"],
    "user_memory_kb": ["user_memory"],
}
LAYER_COLLECTIONS: dict[str, str] = {layer: layer for layer in RAG_LAYERS}
LAYER_COLLECTIONS["graph_kb"] = "graph_kb"
LAYER_WEIGHTS: dict[str, float] = {
    "role_kb": 0.08,
    "company_kb": 0.07,
    "question_kb": 0.07,
    "answer_kb": 0.09,
    "cv_kb": 0.08,
    "evaluation_kb": 0.1,
    "roadmap_kb": 0.06,
    "user_memory_kb": 0.12,
    "graph_kb": 0.05,
    "general_kb": 0.0,
}
DOC_TYPE_TO_LAYER = {
    doc_type: layer
    for layer, doc_types in RAG_LAYERS.items()
    for doc_type in doc_types
}
PURPOSE_DOC_TYPES: dict[str, list[str]] = {
    "evaluation": ["role", "rubric", "framework", "anti_pattern", "company", "user_memory", "knowledge"],
    "hint": ["framework", "rubric", "anti_pattern", "role", "company", "user_memory", "knowledge"],
    "question_generation": ["question_seed", "rubric", "company", "role", "framework", "user_memory", "cv_signal", "knowledge"],
    "cv_screening": ["cv_signal", "role", "rubric", "user_memory", "knowledge"],
    "roadmap": ["drill", "framework", "rubric", "role", "company", "knowledge"],
    "story_search": ["story", "framework", "rubric", "knowledge"],
}
PURPOSE_LAYERS: dict[str, list[str]] = {
    "evaluation": ["evaluation_kb", "role_kb", "answer_kb", "company_kb", "user_memory_kb"],
    "hint": ["answer_kb", "evaluation_kb", "role_kb", "company_kb", "user_memory_kb"],
    "question_generation": ["question_kb", "role_kb", "company_kb", "answer_kb", "cv_kb", "user_memory_kb"],
    "cv_screening": ["cv_kb", "role_kb", "evaluation_kb", "user_memory_kb"],
    "roadmap": ["roadmap_kb", "answer_kb", "evaluation_kb", "role_kb", "company_kb", "user_memory_kb"],
    "story_search": ["answer_kb", "evaluation_kb", "role_kb"],
}
GRAPH_EDGES: list[dict[str, str]] = [
    {"source": "backend_developer", "relation": "requires", "target": "api_design"},
    {"source": "backend_developer", "relation": "requires", "target": "data_modeling"},
    {"source": "backend_developer", "relation": "requires", "target": "reliability"},
    {"source": "software_engineer", "relation": "requires", "target": "testing"},
    {"source": "frontend_developer", "relation": "requires", "target": "accessibility"},
    {"source": "devops_engineer", "relation": "requires", "target": "observability"},
    {"source": "api_design", "relation": "evaluated_by", "target": "technical_depth"},
    {"source": "metrics", "relation": "evaluated_by", "target": "impact"},
    {"source": "star", "relation": "improves", "target": "structure"},
    {"source": "google", "relation": "values", "target": "structured_problem_solving"},
    {"source": "meta", "relation": "values", "target": "impact"},
    {"source": "amazon", "relation": "values", "target": "ownership"},
    {"source": "stripe", "relation": "values", "target": "precision"},
    {"source": "apple", "relation": "values", "target": "craft"},
]


@dataclass
class RetrievalQuery:
    purpose: str
    query: str
    profession: str = ""
    k: int = 4
    collection_name: str = DEFAULT_COLLECTION
    sector: str = ""
    company: str = ""
    focus_area: str = ""
    difficulty: str = ""
    user_memory: list[dict[str, Any]] = field(default_factory=list)
    cv_facts: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    extra_filters: dict[str, str] = field(default_factory=dict)
    route_collections: bool = True


@dataclass
class RetrievalResult:
    context: str
    evidence: list[dict[str, Any]]
    summary: str
    quality: dict[str, Any]


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)


def get_collection(name: str = DEFAULT_COLLECTION):
    chroma = get_chroma_client()

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )

    return chroma.get_or_create_collection(
        name=name,
        embedding_function=ef
    )


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def _profession_slug(profession: str) -> str:
    return _slug(profession)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def layer_for_metadata(meta: dict[str, Any]) -> str:
    explicit = str(meta.get("layer") or "").strip()
    if explicit:
        return explicit
    source = str(meta.get("source") or "").lower()
    doc_type = str(meta.get("doc_type") or "knowledge").lower().strip()
    if "company" in source or "rubric" in source and "evaluation" not in source:
        return "company_kb"
    if "question" in source:
        return "question_kb"
    if "cv_" in source or "cv_signal" in doc_type:
        return "cv_kb"
    if "evaluation" in source:
        return "evaluation_kb"
    if "drill" in source:
        return "roadmap_kb"
    if doc_type in {"framework", "anti_pattern", "story"}:
        return "answer_kb"
    return DOC_TYPE_TO_LAYER.get(doc_type, "general_kb")


def collection_for_layer(layer: str) -> str:
    return LAYER_COLLECTIONS.get(layer, LEGACY_COLLECTION)


def _keyword_overlap_score(query_tokens: set[str], doc: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(1, len(query_tokens))


def _generate_query_variants(profession: str, query: str) -> List[str]:
    cleaned = (query or "").strip()
    if not cleaned:
        return [f"Profession: {profession}"]

    variants = [
        f"Profession: {profession}\n{cleaned}",
        cleaned,
    ]

    # Extract focused terms from the query to improve retrieval recall.
    terms = [w for w in re.findall(r"[a-zA-Z0-9_]+", cleaned.lower()) if len(w) > 3]
    if terms:
        keywords = " ".join(list(dict.fromkeys(terms))[:12])
        variants.append(f"{profession} interview {keywords}")

    return list(dict.fromkeys(variants))


def _generate_advanced_query_variants(payload: RetrievalQuery) -> list[str]:
    cleaned = (payload.query or "").strip()
    parts = [
        f"Purpose: {payload.purpose}",
        f"Profession: {payload.profession}" if payload.profession else "",
        f"Difficulty: {payload.difficulty}" if payload.difficulty else "",
        f"Focus area: {payload.focus_area}" if payload.focus_area else "",
        f"Sector: {payload.sector}" if payload.sector else "",
        f"Company: {payload.company}" if payload.company else "",
        cleaned,
    ]
    rich = "\n".join(part for part in parts if part)
    variants = [rich or cleaned or f"Purpose: {payload.purpose}"]
    if cleaned:
        variants.append(cleaned)
    if payload.profession:
        variants.append(f"{payload.profession} {payload.focus_area} {payload.difficulty} interview {cleaned}".strip())
    terms = [w for w in re.findall(r"[a-zA-Z0-9_]+", rich.lower()) if len(w) > 3]
    if terms:
        variants.append(" ".join(list(dict.fromkeys(terms))[:18]))
    return list(dict.fromkeys(v for v in variants if v.strip()))


def _candidate_filters(payload: RetrievalQuery) -> list[Optional[dict[str, str]]]:
    filters: list[Optional[dict[str, str]]] = []
    profession = _profession_slug(payload.profession)
    company = _slug(payload.company)
    doc_types = payload.doc_types or PURPOSE_DOC_TYPES.get(payload.purpose, [])
    if profession:
        filters.append({"profession": profession})
    if doc_types:
        for doc_type in doc_types[:3]:
            filters.append({"doc_type": doc_type})
    if company:
        filters.append({"company": company})
    for key, value in payload.extra_filters.items():
        if value:
            filters.append({key: _slug(value)})
    filters.append(None)
    unique: list[Optional[dict[str, str]]] = []
    seen = set()
    for item in filters:
        marker = tuple(sorted(item.items())) if item else ()
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    return unique


def _layers_for_payload(payload: RetrievalQuery) -> list[str]:
    layers = PURPOSE_LAYERS.get(payload.purpose, [])
    if payload.doc_types:
        layers.extend(DOC_TYPE_TO_LAYER.get(doc_type, "general_kb") for doc_type in payload.doc_types)
    if payload.user_memory:
        layers.append("user_memory_kb")
    if payload.cv_facts:
        layers.append("cv_kb")
    return list(dict.fromkeys(layer for layer in layers if layer))


def _collection_routes(payload: RetrievalQuery) -> list[dict[str, Any]]:
    if not payload.route_collections or (payload.collection_name and payload.collection_name != LEGACY_COLLECTION):
        return [{"collection": payload.collection_name, "layer": "custom_kb", "weight": 0.0}]
    routes = [
        {
            "collection": collection_for_layer(layer),
            "layer": layer,
            "weight": LAYER_WEIGHTS.get(layer, 0.0),
        }
        for layer in _layers_for_payload(payload)
        if layer not in {"user_memory_kb", "graph_kb"}
    ]
    routes.append({"collection": LEGACY_COLLECTION, "layer": "legacy_kb", "weight": 0.0})
    unique: list[dict[str, Any]] = []
    seen = set()
    for route in routes:
        if route["collection"] not in seen:
            seen.add(route["collection"])
            unique.append(route)
    return unique


def _query_collection(collection: Any, query_text: str, n_results: int, where: Optional[dict[str, str]] = None) -> Dict[str, Any]:
    if where:
        try:
            return collection.query(query_texts=[query_text], n_results=n_results, where=where)
        except Exception:
            pass
    return collection.query(query_texts=[query_text], n_results=n_results)


def _query_collection_with_optional_filter(
    collection: Any,
    query_text: str,
    n_results: int,
    profession_slug: str,
) -> Dict[str, Any]:
    return _query_collection(collection, query_text, n_results, {"profession": profession_slug})


def _metadata_match_score(payload: RetrievalQuery, meta: dict[str, Any]) -> float:
    score = 0.0
    if payload.profession and str(meta.get("profession", "")).lower().strip() == _profession_slug(payload.profession):
        score += 0.16
    if payload.company and str(meta.get("company", "")).lower().strip() in {_slug(payload.company), payload.company.lower().strip()}:
        score += 0.08
    if payload.focus_area and str(meta.get("focus_area", "")).lower().strip() == _slug(payload.focus_area):
        score += 0.05
    if payload.difficulty and str(meta.get("difficulty", "")).lower().strip() == _slug(payload.difficulty):
        score += 0.04
    doc_types = payload.doc_types or PURPOSE_DOC_TYPES.get(payload.purpose, [])
    if doc_types and str(meta.get("doc_type", "")).lower().strip() in doc_types:
        score += 0.07
    layer = layer_for_metadata(meta)
    if layer in {"user_memory_kb", "cv_kb"} and (payload.user_memory or payload.cv_facts):
        score += 0.06
    return score


def _relevance_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.46:
        return "medium"
    return "low"


def _quality_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {
            "label": "none",
            "score": 0,
            "top_score": 0,
            "source_count": 0,
            "doc_type_count": 0,
            "layer_count": 0,
            "collection_count": 0,
            "evidence_count": 0,
            "coverage": 0,
        }
    top_score = float(evidence[0].get("hybrid_score", 0) or 0)
    avg_score = sum(float(item.get("hybrid_score", 0) or 0) for item in evidence) / len(evidence)
    source_count = len({str(item.get("source", "")) for item in evidence if item.get("source")})
    doc_type_count = len({str(item.get("doc_type", "")) for item in evidence if item.get("doc_type")})
    layer_count = len({str(item.get("layer", "")) for item in evidence if item.get("layer")})
    collection_count = len({str(item.get("collection", "")) for item in evidence if item.get("collection")})
    coverage_bonus = min(0.12, (source_count - 1) * 0.04 + (doc_type_count - 1) * 0.03)
    quality_score = max(0.0, min(1.0, (0.68 * top_score) + (0.32 * avg_score) + coverage_bonus))
    label = "high" if quality_score >= 0.68 else "medium" if quality_score >= 0.42 else "low"
    return {
        "label": label,
        "score": round(quality_score * 100),
        "top_score": round(top_score, 4),
        "source_count": source_count,
        "doc_type_count": doc_type_count,
        "layer_count": layer_count,
        "collection_count": collection_count,
        "evidence_count": len(evidence),
        "coverage": round(min(1.0, (source_count + doc_type_count + layer_count) / 9), 4),
    }


def _build_summary(payload: RetrievalQuery, quality: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return f"No retrieval evidence found for {payload.purpose}; fallback coaching logic was used."
    sources = ", ".join(list(dict.fromkeys(str(item.get("source", "unknown")) for item in evidence))[:3])
    collections = ", ".join(list(dict.fromkeys(str(item.get("collection", "")) for item in evidence if item.get("collection")))[:5])
    collection_text = f" across {quality.get('collection_count', 0)} collection(s)" if quality.get("collection_count") else ""
    collections_suffix = f". Collections: {collections}" if collections else ""
    return (
        f"{quality['label'].title()} RAG support for {payload.purpose}: "
        f"{len(evidence)} chunks from {quality['source_count']} source(s)"
        f"{collection_text}, top sources: {sources}{collections_suffix}."
    )


def _format_context(evidence: Iterable[dict[str, Any]]) -> str:
    chunks = []
    for item in evidence:
        chunks.append(
            "[Collection: {collection} | Source: {source} | Layer: {layer} | DocType: {doc_type} | HybridScore: {score} | Profession: {profession}]\n{content}".format(
                collection=item.get("collection", LEGACY_COLLECTION),
                source=item.get("source", "unknown"),
                layer=item.get("layer", "general_kb"),
                doc_type=item.get("doc_type", "knowledge"),
                score=item.get("hybrid_score", 0),
                profession=item.get("profession", ""),
                content=item.get("content", ""),
            )
        )
    return "\n\n---\n\n".join(chunks)


def graph_context_for_query(payload: RetrievalQuery) -> list[dict[str, Any]]:
    tokens = _tokenize(
        " ".join(
            [
                payload.query,
                payload.profession,
                payload.company,
                payload.focus_area,
                payload.difficulty,
                " ".join(payload.cv_facts),
                " ".join(str(item.get("memory_type", "")) + " " + str(item.get("content", "")) for item in payload.user_memory),
            ]
        )
    )
    hits: list[dict[str, Any]] = []
    for edge in GRAPH_EDGES:
        edge_text = " ".join(edge.values())
        edge_tokens = _tokenize(edge_text) | _tokenize(edge_text.replace("_", " "))
        overlap = len(tokens & edge_tokens)
        if overlap:
            hits.append({**edge, "overlap": overlap})
    hits.sort(key=lambda item: item["overlap"], reverse=True)
    return hits[:8]


def _memory_evidence(payload: RetrievalQuery, query_tokens: set[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, memory in enumerate(payload.user_memory[:8]):
        content = _clean_text(memory.get("content", ""))
        if not content:
            continue
        keyword_score = _keyword_overlap_score(query_tokens, content)
        signal_score = float(memory.get("score", 0.55) or 0.55)
        hybrid_score = min(1.0, (0.45 * keyword_score) + (0.45 * signal_score) + 0.35)
        items.append(
            {
                "source": f"user_memory:{memory.get('memory_type', 'signal')}",
                "collection": "user_memory_kb",
                "layer": "user_memory_kb",
                "doc_type": "user_memory",
                "profession": _profession_slug(payload.profession),
                "semantic_score": round(signal_score, 4),
                "keyword_score": round(keyword_score, 4),
                "metadata_score": 0.12,
                "hybrid_score": round(hybrid_score, 4),
                "content": content,
                "preview": content[:260],
                "keyword_hits": len(query_tokens & _tokenize(content)),
                "memory_id": memory.get("id"),
                "relevance_label": _relevance_label(hybrid_score),
            }
        )
    return items


def _cv_evidence(payload: RetrievalQuery, query_tokens: set[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, fact in enumerate(payload.cv_facts[:8]):
        content = _clean_text(fact)
        if not content:
            continue
        keyword_score = _keyword_overlap_score(query_tokens, content)
        hybrid_score = min(1.0, 0.42 + (0.45 * keyword_score))
        items.append(
            {
                "source": f"cv_fact:{idx + 1}",
                "collection": "cv_kb",
                "layer": "cv_kb",
                "doc_type": "cv_signal",
                "profession": _profession_slug(payload.profession),
                "semantic_score": round(hybrid_score, 4),
                "keyword_score": round(keyword_score, 4),
                "metadata_score": 0.1,
                "hybrid_score": round(hybrid_score, 4),
                "content": content,
                "preview": content[:260],
                "keyword_hits": len(query_tokens & _tokenize(content)),
                "relevance_label": _relevance_label(hybrid_score),
            }
        )
    return items


def build_citations(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for idx, item in enumerate(evidence[:6], start=1):
        doc_type = str(item.get("doc_type") or "knowledge")
        preview = str(item.get("preview") or "").strip()
        if not preview:
            continue
        citations.append(
            {
                "id": f"C{idx}",
                "source": item.get("source", "unknown"),
                "layer": item.get("layer", DOC_TYPE_TO_LAYER.get(doc_type, "general_kb")),
                "doc_type": doc_type,
                "claim": preview[:180],
                "score": item.get("hybrid_score", 0),
                "relevance_label": item.get("relevance_label", "low"),
            }
        )
    return citations


def evaluate_retrieval(evidence: list[dict[str, Any]], answer_text: str = "", feedback_text: str = "") -> dict[str, Any]:
    quality = _quality_summary(evidence)
    answer_tokens = _tokenize(answer_text)
    feedback_tokens = _tokenize(feedback_text)
    evidence_tokens = _tokenize(" ".join(str(item.get("content", "")) for item in evidence))
    precision_proxy = (
        sum(1 for item in evidence if item.get("relevance_label") in {"high", "medium"}) / max(1, len(evidence))
    )
    faithfulness_proxy = len(feedback_tokens & evidence_tokens) / max(1, len(feedback_tokens)) if feedback_tokens else 0.0
    answer_grounding_proxy = len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens)) if answer_tokens else 0.0
    low_confidence = quality["label"] in {"none", "low"} or precision_proxy < 0.35
    return {
        "retrieval_precision_proxy": round(precision_proxy, 4),
        "coverage": quality.get("coverage", 0),
        "faithfulness_proxy": round(faithfulness_proxy, 4),
        "answer_grounding_proxy": round(answer_grounding_proxy, 4),
        "low_confidence": low_confidence,
        "quality": quality,
    }


def retrieve(payload: RetrievalQuery) -> RetrievalResult:
    query_tokens = _tokenize(
        " ".join(
            [
                payload.query,
                payload.profession,
                payload.focus_area,
                payload.sector,
                payload.company,
                payload.difficulty,
                " ".join(payload.doc_types),
            ]
        )
    )
    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}
    query_variants = _generate_advanced_query_variants(payload)
    n_results = max(18, payload.k * 8)
    routes = _collection_routes(payload)
    successful_collections: list[str] = []

    for route in routes:
        try:
            collection = get_collection(route["collection"])
            successful_collections.append(route["collection"])
        except Exception:
            continue

        for variant in query_variants:
            for where in _candidate_filters(payload):
                try:
                    results = _query_collection(collection, variant, n_results, where=where)
                except Exception:
                    continue
                docs: list[str] = results.get("documents", [[]])[0] or []
                metas: list[dict[str, Any]] = results.get("metadatas", [[]])[0] or []
                distances: list[float] = results.get("distances", [[]])[0] or []

                for idx, doc in enumerate(docs):
                    meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                    source = str(meta.get("source", "unknown"))
                    distance = float(distances[idx]) if idx < len(distances) else 1.0
                    semantic_score = max(0.0, min(1.0, 1.0 - distance))
                    keyword_score = _keyword_overlap_score(query_tokens, doc)
                    metadata_score = _metadata_match_score(payload, meta)
                    layer = layer_for_metadata(meta)
                    layer_weight = max(float(route.get("weight", 0.0)), LAYER_WEIGHTS.get(layer, 0.0))
                    length_penalty = 0.04 if len(doc) > 2400 else 0.0
                    hybrid_score = (
                        (0.52 * semantic_score)
                        + (0.27 * keyword_score)
                        + metadata_score
                        + layer_weight
                        - length_penalty
                    )
                    key = (route["collection"], source, doc[:160])
                    candidate = {
                        "source": source,
                        "collection": route["collection"],
                        "collection_route_layer": route["layer"],
                        "doc_type": str(meta.get("doc_type", "knowledge")),
                        "layer": layer,
                        "profession": str(meta.get("profession", "")).lower().strip(),
                        "sector": str(meta.get("sector", "")).lower().strip(),
                        "company": str(meta.get("company", "")).lower().strip(),
                        "focus_area": str(meta.get("focus_area", "")).lower().strip(),
                        "difficulty": str(meta.get("difficulty", "")).lower().strip(),
                        "chunk_index": meta.get("chunk_index"),
                        "semantic_score": round(semantic_score, 4),
                        "keyword_score": round(keyword_score, 4),
                        "metadata_score": round(metadata_score, 4),
                        "layer_weight": round(layer_weight, 4),
                        "hybrid_score": round(max(0.0, hybrid_score), 4),
                        "content": doc,
                        "preview": _clean_text(doc)[:260],
                        "keyword_hits": len(query_tokens & _tokenize(doc)),
                        "matched_filter": where or {},
                    }
                    current = candidate_map.get(key)
                    if current is None or candidate["hybrid_score"] > current["hybrid_score"]:
                        candidate_map[key] = candidate

    if not successful_collections and not (payload.user_memory or payload.cv_facts):
        return RetrievalResult(
            context="",
            evidence=[],
            summary=f"No routed vector collections were available for {payload.purpose}; fallback coaching logic was used.",
            quality=_quality_summary([]),
        )

    ranked = sorted(candidate_map.values(), key=lambda x: x["hybrid_score"], reverse=True)
    ranked.extend(_memory_evidence(payload, query_tokens))
    ranked.extend(_cv_evidence(payload, query_tokens))
    graph_hits = graph_context_for_query(payload)
    for idx, edge in enumerate(graph_hits):
        content = f"Graph relation: {edge['source']} {edge['relation']} {edge['target']}."
        ranked.append(
            {
                "source": f"knowledge_graph:{idx + 1}",
                "collection": "graph_kb",
                "layer": "graph_kb",
                "doc_type": "graph_edge",
                "profession": _profession_slug(payload.profession),
                "semantic_score": 0.55,
                "keyword_score": round(min(1.0, edge.get("overlap", 0) / 3), 4),
                "metadata_score": 0.08,
                "hybrid_score": round(min(1.0, 0.48 + (0.08 * edge.get("overlap", 0))), 4),
                "content": content,
                "preview": content,
                "keyword_hits": edge.get("overlap", 0),
                "graph_edge": edge,
            }
        )
    ranked = sorted(ranked, key=lambda x: x["hybrid_score"], reverse=True)
    top: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    doc_type_counts: dict[str, int] = {}
    collection_counts: dict[str, int] = {}
    for item in ranked:
        source_count = source_counts.get(str(item["source"]), 0)
        doc_type_count = doc_type_counts.get(str(item["doc_type"]), 0)
        collection_count = collection_counts.get(str(item.get("collection", "")), 0)
        adjusted = float(item["hybrid_score"]) - (0.07 * source_count) - (0.035 * doc_type_count) - (0.025 * collection_count)
        picked = dict(item)
        picked["hybrid_score"] = round(max(0.0, adjusted), 4)
        picked["relevance_label"] = _relevance_label(picked["hybrid_score"])
        top.append(picked)
        source_counts[str(item["source"])] = source_count + 1
        doc_type_counts[str(item["doc_type"])] = doc_type_count + 1
        collection_counts[str(item.get("collection", ""))] = collection_count + 1
        if len(top) >= max(payload.k * 2, 8):
            break
    top.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top = top[: payload.k]
    quality = _quality_summary(top)
    summary = _build_summary(payload, quality, top)
    return RetrievalResult(context=_format_context(top), evidence=top, summary=summary, quality=quality)


def retrieve_for_evaluation(profession: str, question: str, answer_text: str, config: Optional[dict[str, Any]] = None) -> RetrievalResult:
    config = config or {}
    user_memory = config.get("user_memory") if isinstance(config.get("user_memory"), list) else []
    cv_facts = config.get("cv_facts") if isinstance(config.get("cv_facts"), list) else []
    return retrieve(
        RetrievalQuery(
            purpose="evaluation",
            profession=profession,
            query=f"Question: {question}\nCandidate answer: {answer_text}",
            difficulty=str(config.get("difficulty") or ""),
            focus_area=str(config.get("focus_area") or ""),
            sector=str(config.get("sector") or ""),
            company=str(config.get("target_company") or config.get("company_pack") or ""),
            user_memory=user_memory,
            cv_facts=[str(item) for item in cv_facts],
            k=5,
        )
    )


def retrieve_for_hint(profession: str, question: str, config: Optional[dict[str, Any]] = None) -> RetrievalResult:
    config = config or {}
    user_memory = config.get("user_memory") if isinstance(config.get("user_memory"), list) else []
    cv_facts = config.get("cv_facts") if isinstance(config.get("cv_facts"), list) else []
    return retrieve(
        RetrievalQuery(
            purpose="hint",
            profession=profession,
            query=f"Interview question: {question}\nNeed concise coaching hints, answer structure, rubric expectations, and anti-patterns.",
            difficulty=str(config.get("difficulty") or ""),
            focus_area=str(config.get("focus_area") or ""),
            sector=str(config.get("sector") or ""),
            company=str(config.get("target_company") or config.get("company_pack") or ""),
            user_memory=user_memory,
            cv_facts=[str(item) for item in cv_facts],
            k=4,
        )
    )


def retrieve_for_question_generation(
    profession: str,
    config: Optional[dict[str, Any]] = None,
    asked_topics: Optional[list[str]] = None,
    asked_questions: Optional[list[str]] = None,
) -> RetrievalResult:
    config = config or {}
    asked_topics = asked_topics or []
    asked_questions = asked_questions or []
    user_memory = config.get("user_memory") if isinstance(config.get("user_memory"), list) else []
    cv_facts = config.get("cv_facts") if isinstance(config.get("cv_facts"), list) else []
    return retrieve(
        RetrievalQuery(
            purpose="question_generation",
            profession=profession,
            query=(
                "Generate fresh interview question seeds. "
                f"Avoid topics: {', '.join(asked_topics[-12:])}. "
                f"Recent questions: {' | '.join(asked_questions[-8:])}."
            ),
            difficulty=str(config.get("difficulty") or ""),
            focus_area=str(config.get("focus_area") or ""),
            sector=str(config.get("sector") or ""),
            company=str(config.get("target_company") or config.get("company_pack") or ""),
            user_memory=user_memory,
            cv_facts=[str(item) for item in cv_facts],
            k=4,
        )
    )


def retrieve_for_cv_screening(cv_text: str, target_profession: str = "", k: int = 5) -> RetrievalResult:
    return retrieve(
        RetrievalQuery(
            purpose="cv_screening",
            profession=target_profession,
            query=(
                "CV screening signals, role fit evidence, skill gaps, interview prep priorities.\n"
                f"CV excerpt: {_clean_text(cv_text)[:3500]}"
            ),
            k=k,
        )
    )


def retrieve_for_roadmap(
    profession: str,
    target_company: str = "",
    focus_area: str = "",
    interview_date: str = "",
    user_memory: Optional[list[dict[str, Any]]] = None,
    cv_facts: Optional[list[str]] = None,
    k: int = 5,
) -> RetrievalResult:
    return retrieve(
        RetrievalQuery(
            purpose="roadmap",
            profession=profession,
            query=(
                "Build a preparation roadmap with drills, weak-skill practice, frameworks, and success criteria. "
                f"Interview date: {interview_date}"
            ),
            company=target_company,
            focus_area=focus_area,
            user_memory=user_memory or [],
            cv_facts=cv_facts or [],
            k=k,
        )
    )


def rank_story_candidates(query: str, stories: list[dict[str, Any]], k: int = 20) -> list[dict[str, Any]]:
    """Semantic-lite rerank for private Story Vault data without writing user text to Chroma."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return stories[:k]
    ranked: list[tuple[float, dict[str, Any]]] = []
    for story in stories:
        text = " ".join(
            [
                str(story.get("title") or ""),
                str(story.get("question") or ""),
                str(story.get("answer") or ""),
                " ".join(str(tag) for tag in story.get("tags") or []),
            ]
        )
        keyword_score = _keyword_overlap_score(query_tokens, text)
        title_boost = 0.12 if query.lower() in str(story.get("title") or "").lower() else 0.0
        metric_boost = 0.04 if any(ch.isdigit() for ch in text) else 0.0
        score = keyword_score + title_boost + metric_boost
        out = dict(story)
        out["retrieval_match"] = {
            "hybrid_score": round(score, 4),
            "keyword_score": round(keyword_score, 4),
            "relevance_label": _relevance_label(score),
        }
        ranked.append((score, out))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for score, item in ranked[:k] if score > 0]


def retrieve_context(
    profession: str,
    query: str,
    k: int = 4,
    return_evidence: bool = False,
) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    advanced = retrieve(
        RetrievalQuery(
            purpose="evaluation",
            profession=profession,
            query=query,
            k=k,
            collection_name=DEFAULT_COLLECTION,
            route_collections=False,
        )
    )
    if return_evidence:
        return advanced.context, advanced.evidence
    return advanced.context