"""Layer-2 tests (retrieval). Unit tests run offline; integration needs
BIFROST_INTEGRATION=1 and the Dockerized Neo4j/Qdrant.
"""
from __future__ import annotations

import os

import pytest

from infra.retrieval.bm25_index import BM25Index
from infra.retrieval.corpus import CONCEPT_REGISTRY, build_corpus


# ── unit ─────────────────────────────────────────────────────────────────
def test_corpus_shape_and_determinism():
    a, b = build_corpus(), build_corpus()
    assert len(a["documents"]) == 500
    assert len(a["registry"]) == 25
    assert len(a["queries"]) >= 40
    assert a["documents"] == b["documents"]
    assert a["queries"] == b["queries"]


def test_corpus_families_map_to_concepts():
    data = build_corpus()
    for fam, ids in data["registry"].items():
        assert len(ids) == 20, fam
        doc = data["documents"][int(ids[0].split("-")[1])]
        assert doc["family"] == fam
        for c in doc["concepts"]:
            assert c in CONCEPT_REGISTRY


def test_bm25_finds_keyword_docs():
    data = build_corpus()
    bm25 = BM25Index(data["documents"])
    hits = bm25.search("kubernetes helm cluster deployment", k=5)
    assert hits, "bm25 returned nothing"
    # top hit must be from the kubernetes-deployment family
    top_id = hits[0][0]
    top_doc = next(d for d in data["documents"] if d["doc_id"] == top_id)
    assert top_doc["family"] == "kubernetes-deployment"


def test_eval_queries_have_valid_families():
    data = build_corpus()
    for eq in data["queries"]:
        assert eq["family"] in data["registry"]
        assert len(eq["query"]) > 10


def test_precision_math():
    from scripts.check_layer2_results import precision_at_5

    assert precision_at_5(["A", "B", "C", "D", "E"], {"A", "C"}) == 0.4
    assert precision_at_5(["A", "B", "C", "D", "E"], {"Z"}) == 0.0


# ── integration (live Neo4j/Qdrant) ─────────────────────────────────────
@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_neo4j_ingested():
    from infra.retrieval.graph import KnowledgeGraph

    g = KnowledgeGraph()
    stats = g.stats()
    g.close()
    assert stats["documents"] == 500
    assert stats["concepts"] > 0


@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_qdrant_collection():
    import os

    from qdrant_client import QdrantClient

    client = QdrantClient(host=os.getenv("QDRANT_HOST", "127.0.0.1"),
                          port=int(os.getenv("QDRANT_PORT", "6333")))
    assert client.collection_exists(os.getenv("QDRANT_COLLECTION", "bifrost_docs"))