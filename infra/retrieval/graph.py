"""Neo4j knowledge graph — Query / Document / Concept and the retrieval
paths the plan calls out:

  Query -(ASKS_ABOUT)-> Concept <-(HAS_CONCEPT)- Document      (direct)
  Query -(ASKS_ABOUT)-> Concept -(RELATED_TO)-> Concept2
                          <-(HAS_CONCEPT)- Document            (one hop)

Concepts are matched from a registry of aliases (word-boundary), so natural
language like "I locked myself out" maps to the `password` concept node.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infra.retrieval.corpus import CONCEPT_REGISTRY  # noqa: E402

load_dotenv()

BOLT_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bifrost_neo4j_pw")


class KnowledgeGraph:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            BOLT_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="WARNING",  # silence INFO/perf spam
        )
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    # ── schema ───────────────────────────────────────────────────────────
    def init_schema(self) -> None:
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT document_id IF NOT EXISTS "
                  "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            s.run("CREATE CONSTRAINT concept_name IF NOT EXISTS "
                  "FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            s.run("CREATE CONSTRAINT query_id IF NOT EXISTS "
                  "FOR (q:Query) REQUIRE q.qid IS UNIQUE")

    def reset(self) -> None:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # ── ingestion ────────────────────────────────────────────────────────
    def ingest(self, documents: List[Dict[str, Any]]) -> int:
        self.init_schema()
        with self.driver.session() as s:
            for d in documents:
                s.run(
                    "MERGE (doc:Document {doc_id: $doc_id}) "
                    "SET doc.title = $title, doc.body = $body, doc.family = $family, "
                    "    doc.source = $source ",
                    doc_id=d["doc_id"], title=d["title"], body=d["body"],
                    family=d["family"], source=d["source"],
                )
                concepts = d.get("concepts", [])
                for cname in concepts:
                    s.run("MERGE (c:Concept {name: $name})", name=cname)
                    s.run(
                        "MATCH (doc:Document {doc_id: $doc_id}), (c:Concept {name: $name}) "
                        "MERGE (doc)-[:HAS_CONCEPT]->(c)",
                        doc_id=d["doc_id"], name=cname,
                    )
                # concepts sharing a document are related
                for i, a in enumerate(concepts):
                    for b in concepts[i + 1:]:
                        s.run(
                            "MATCH (a:Concept {name: $a}), (b:Concept {name: $b}) "
                            "MERGE (a)-[:RELATED_TO]->(b) MERGE (b)-[:RELATED_TO]->(a)",
                            a=a, b=b,
                        )
        return len(documents)

    def stats(self) -> Dict[str, int]:
        with self.driver.session() as s:
            counts = {}
            for label in ("Document", "Concept", "Query"):
                counts[label.lower() + "s"] = s.run(
                    f"MATCH (n:{label}) RETURN count(n) AS n"
                ).single()["n"]
            counts["rel_count"] = s.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
            return counts

    # ── query-side ───────────────────────────────────────────────────────
    def _match_concepts(self, query: str) -> List[str]:
        lowered = query.lower()
        hits = []
        for name, aliases in CONCEPT_REGISTRY.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", lowered) or alias in lowered:
                    hits.append(name)
                    break
        return hits

    def query(self, query_text: str, limit: int = 10) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Returns ([(doc_id, strength)], matched_concepts)."""
        qid = hashlib.sha256(query_text.encode()).hexdigest()[:16]
        concepts = self._match_concepts(query_text)
        with self.driver.session() as s:
            s.run("MERGE (q:Query {qid: $qid}) SET q.text = $text",
                  qid=qid, text=query_text)
            for c in concepts:
                s.run(
                    "MATCH (q:Query {qid: $qid}), (c:Concept {name: $name}) "
                    "MERGE (q)-[:ASKS_ABOUT]->(c)",
                    qid=qid, name=c,
                )
            if not concepts:
                return [], []
            # direct + one-hop related concept documents, deduped
            rows = s.run(
                "MATCH (q:Query {qid: $qid})-[:ASKS_ABOUT]->(c:Concept) "
                "OPTIONAL MATCH (c)<-[:HAS_CONCEPT]-(direct:Document) "
                "OPTIONAL MATCH (c)-[:RELATED_TO]->(c2:Concept)<-[:HAS_CONCEPT]-(rel:Document) "
                "WITH COLLECT(DISTINCT direct) AS d, COLLECT(DISTINCT rel) AS r "
                "UNWIND (d + r) AS doc "
                "WITH doc WHERE doc IS NOT NULL "
                "RETURN DISTINCT doc.doc_id AS id LIMIT $limit",
                qid=qid, limit=limit,
            )
            results = [(r["id"], 1.0) for r in rows]
            return results, concepts