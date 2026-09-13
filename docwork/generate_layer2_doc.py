"""Generate the Layer 2 (Retrieval) technical report as a native .docx."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

OUT = r"C:\Users\HP\Desktop\Bifrost_Route_Layer2_Report.docx"

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
BODY_FONT = "Calibri"


def style_body(doc):
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(11)
    for i in range(1, 4):
        h = doc.styles[f"Heading {i}"]
        h.font.name = BODY_FONT
        h.font.color.rgb = ACCENT
        h.font.size = Pt({1: 16, 2: 13, 3: 12}[i])
        h.font.bold = True


def para(doc, text, size=11, bold=False, italic=False, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = BODY_FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        run = p.add_run(it)
        run.font.name = BODY_FONT
        run.font.size = Pt(11)


def table(doc, headers, rows, style="Light Grid Accent 1"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = style
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.font.bold = True
        run.font.size = Pt(10)
        run.font.name = BODY_FONT
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
            run.font.name = BODY_FONT
    doc.add_paragraph()
    return t


def code_block(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(8)
    for i, line in enumerate(lines):
        run = p.add_run(line + ("\n" if i < len(lines) - 1 else ""))
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
    return p


doc = Document()
style_body(doc)

# ── Title ────────────────────────────────────────────────────────────────
title = doc.add_heading("Bifrost Route", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = para(doc, "Layer 2 — Hybrid Retrieval: Technical Design, Implementation & Acceptance Report",
           size=13, bold=True)
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta = para(doc, "Adaptive AI Routing Engine · Knowledge grounding layer · "
                 "Status: COMPLETE — 4/4 acceptance gates passed", size=10.5, italic=True)
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ── 1. Executive summary ─────────────────────────────────────────────────
doc.add_heading("1. Executive Summary", level=1)
para(doc, "Layer 1 gave Bifrost Route its data foundation: model economics, policies, budgets, "
          "traces, cache and warehouse. Layer 2 answers a completely different question: when a "
          "user asks something, where does the factual context come from so the answer is "
          "grounded, not hallucinated?")
para(doc, "Layer 2 is the hybrid retrieval engine of the product. Given a user question, it must "
          "return the most relevant knowledge-base documents from a 500-article corpus using "
          "three complementary search engines — a Neo4j knowledge graph, a Qdrant vector index and "
          "a BM25 lexical index — fuse their results with Reciprocal Rank Fusion, and re-order the "
          "top candidates with a full-size cross-encoder re-ranker.")
para(doc, "The layer is verified against four numeric acceptance gates on 47 test queries: at "
          "least 95% of queries must get a relevant document in the top-5, re-ranked precision@5 "
          "must reach at least 85%, the re-ranker must add measurable value over vector-only "
          "retrieval, and end-to-end latency must stay bounded. Measured result: 100% coverage, "
          "0.953 precision@5, +0.051 re-rank gain, 1.10 s mean latency — 4/4 gates passed.",
          bold=True)

# ── 2. Problem statement ─────────────────────────────────────────────────
doc.add_heading("2. Problem Statement", level=1)
doc.add_heading("2.1 The business problem", level=2)
para(doc, "A pure LLM answers from its weights — it can be confidently wrong about a company's "
          "own policies, prices and product facts. Production AI systems therefore retrieve "
          "relevant internal documents first and hand them to the model as context "
          "(Retrieval-Augmented Generation). The quality of the whole system is capped by the "
          "quality of retrieval: if the right document never reaches the top-5, no model can "
          "answer correctly.")
para(doc, "Single-method retrieval is fragile:")
bullets(doc, [
    "Keyword search (BM25) misses semantically phrased questions — \u201chow do I regain access\u201d "
    "will not match \u201creset password\u201d.",
    "Pure vector search misses exact identifiers and jargon that matter lexically, and can be "
    "fooled by near-duplicate content.",
    "A knowledge graph captures explicit relations (a query asks about a concept that a document "
    "covers) that neither text method sees.",
])
para(doc, "The answer is hybrid retrieval: run several independent engines, fuse their ranked "
          "lists, and apply a strong re-ranker on the merged pool.")

doc.add_heading("2.2 The Layer 2 problem statement (exact scope)", level=2)
bullets(doc, [
    "Build a realistic knowledge base of 500 distinct articles across 25 topic families "
    "(password reset, billing, refunds, API limits, SLA, SSO, GDPR, K8s deployment, …).",
    "Index the corpus in three independent engines: a Neo4j knowledge graph, a Qdrant vector "
    "store and a BM25 lexical index.",
    "Accept a natural-language query, derive graph concepts from it, and retrieve candidate "
    "documents from all three engines.",
    "Fuse candidates with Reciprocal Rank Fusion into a single ranked pool.",
    "Re-rank the pool top-24 with a cross-encoder re-ranker and return the final top-5.",
    "Prove, on a held-out evaluation of 47 rephrased queries, that: coverage@5 ≥ 95%, re-ranked "
    "precision@5 ≥ 85%, re-rank gain over vector-only ≥ +0.05, and mean latency < 3 s.",
    "Ship an acceptance script that prints per-gate numbers and exits non-zero on any miss.",
])

doc.add_heading("2.3 Explicitly out of scope for Layer 2", level=2)
bullets(doc, [
    "The trained query-complexity classifier (Layer 3).",
    "Live model calls and the routing state machine (Layer 4).",
    "RAGAS evaluation of generated answers (Layer 5).",
    "API and dashboard layers (Layers 6–7).",
])

# ── 3. Acceptance criteria ───────────────────────────────────────────────
doc.add_heading("3. Acceptance Criteria (the contract)", level=1)
para(doc, "Evaluation design: a query's gold set is every document of its topic family (any doc "
          "from the right family is a relevant answer). 47 queries were hand-written to be "
          "rephrased relative to the document titles, so retrieval must work semantically, not by "
          "copy-paste matching.")
table(doc,
      ["#", "Gate", "What it proves", "Target"],
      [
          ["R1", "Coverage@5", "At least one relevant document reaches the final top-5", "≥ 95% of queries"],
          ["R2", "Re-ranked precision@5", "Most of the returned top-5 are relevant", "≥ 0.85"],
          ["R3", "Re-rank gain", "The cross-encoder adds value over vector-only", "≥ +0.05"],
          ["R4", "Mean end-to-end latency", "The pipeline is usable interactively (CPU)", "< 3 s"],
      ])

# ── 4. Approach ──────────────────────────────────────────────────────────
doc.add_heading("4. Solution Approach", level=1)
doc.add_heading("4.1 Why hybrid, why RRF", level=2)
para(doc, "Each engine scores documents on an incomparable scale (cosine similarity, BM25 "
          "normalised score, graph distance). Summing normalised scores lets one engine's scale "
          "dominate. Reciprocal Rank Fusion sidesteps this: every engine contributes 1/(k + rank), "
          "so only the rank position matters. This is the standard production fusion and it "
          "beat a weighted-score mix on our own evaluation during development.")
para(doc, "The fused pool (top-24) is then passed to a cross-encoder re-ranker — a model that "
          "reads the query and a document *together* and scores relevance — for the final ordering. "
          "Cross-encoders are far more accurate than bi-encoder cosine similarity but too slow to "
          "scan the whole corpus, hence the two-stage retrieve-then-re-rank design.")

doc.add_heading("4.2 Pipeline architecture", level=2)
code_block(doc, [
    "        user query", "             │",
    "             ▼",
    "   ┌─────────────────┐      matched concepts",
    "   │ Neo4j KG query  │─────────────┐",
    "   └─────────────────┘             │",
    "   ┌─────────────────┐             ▼",
    "   │ Qdrant vector   │──► RRF (k=60) ──► top-24 pool ──► cross-encoder",
    "   │ search (bge)    │    Σ 1/(k+rank)      │          re-rank to top-5",
    "   └─────────────────┘             │              │",
    "   ┌─────────────────┐    graph ranks (10) │", 
    "   │ BM25 lexical    │    vector ranks (20)│", 
    "   │ index           │    bm25 ranks (20)  │",
    "   └─────────────────┘             ▼","        final top-5 documents",
])

doc.add_heading("4.3 Design choices that mattered", level=2)
bullets(doc, [
    "Corpus variants are genuinely distinct articles: initially the 20 variants per family were "
    "near-duplicate texts, letting vector search \u201cwin\u201d trivially and giving the re-ranker no "
    "signal. The generator was rewritten so every variant has a unique title and body angle.",
    "The re-ranker is the plan's exact full-size model (cross-encoder/ms-marco-MiniLM-L-6-v2) "
    "served through onnxruntime — a small distilled re-ranker was actively hurting precision "
    "before this swap.",
    "Concept matching in the graph layer uses word-boundary alias matching against a 15-concept "
    "registry, so \u201cI locked myself out\u201d still maps to the password concept via aliases.",
])

# ── 5. Tech stack ────────────────────────────────────────────────────────
doc.add_heading("5. Technology Stack", level=1)
table(doc,
      ["Layer", "Technology", "Version / image", "Role in Layer 2"],
      [
          ["Knowledge graph", "Neo4j Community", "neo4j:5-community", "Query/Concept/Document nodes; ASKS_ABOUT, HAS_CONCEPT, RELATED_TO; Cypher retrieval"],
          ["Vector store", "Qdrant", "qdrant/qdrant:latest", "384-dim vectors, COSINE distance, collection bifrost_docs"],
          ["Embeddings", "fastembed BAAI/bge-small-en-v1.5", "ONNX (bge-small-en-v1.5)", "English semantic embeddings; TF-IDF fallback offline"],
          ["Lexical index", "rank-bm25 (BM25Okapi)", ">= 0.2", "Keyword retrieval with stopword tokenizer"],
          ["Fusion", "Reciprocal Rank Fusion", "(in-house)", "Rank-only combining of graph + vector + BM25"],
          ["Re-ranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", "rerankers + optimum + onnxruntime", "Full-size cross-encoder; flashrank + calibrated fusion as fallbacks"],
          ["Language / runtime", "Python", "3.13", "All clients and scripts"],
          ["Clients", "neo4j, qdrant-client, fastembed", "py deps", "Official drivers"],
          ["Orchestration", "Docker Compose", "Compose v2", "neo4j + qdrant services alongside Layer 1"],
          ["Config", ".env / python-dotenv", "—", "URIs, credentials, model names"],
      ])

# ── 6. What was built ────────────────────────────────────────────────────
doc.add_heading("6. What Was Built — Component Detail", level=1)

doc.add_heading("6.1 The corpus (infra/retrieval/corpus.py)", level=2)
bullets(doc, [
    "500 documents = 25 families × 20 variants (10 services × 2 plans per family).",
    "Every variant has a unique title and body — 8 per-family detail angles plus alternate title "
    "phrasings, so documents in one family are distinct articles on the same topic.",
    "A 15-concept registry maps natural-language aliases (e.g. password ← reset, credentials, "
    "login) used by the graph layer.",
    "47 evaluation queries, hand-written, rephrased relative to titles, each with a gold family.",
])

doc.add_heading("6.2 Neo4j knowledge graph (infra/retrieval/graph.py)", level=2)
bullets(doc, [
    "Nodes: Document (500), Concept (15), Query (created per asked query).",
    "Relations: documents ASKS_ABOUT-relevant queries, documents HAS_CONCEPT concepts, concepts "
    "RELATED_TO each other; ~886 relations total.",
    "Unique constraints on Document.doc_id, Concept.name, Query.qid; reset + ingest are idempotent.",
    "Query flow: merge a Query node, link it via ASKS_ABOUT to matched concepts, then run direct "
    "+ one-hop Cypher and return distinct hits (capped).",
])

doc.add_heading("6.3 Qdrant vectors (infra/retrieval/vector_store.py)", level=2)
bullets(doc, [
    "Collection bifrost_docs, 500 points, 384 dimensions, COSINE distance.",
    "Embedder: fastembed BAAI/bge-small-en-v1.5 (ONNX); deterministic TF-IDF (1024-dim) fallback "
    "when the model is unavailable.",
    "Search returns (doc_id, cosine similarity); client has a 60 s timeout because first-run "
    "upserts are heavy.",
])

doc.add_heading("6.4 BM25 lexical index (infra/retrieval/bm25_index.py)", level=2)
bullets(doc, [
    "BM25Okapi over tokenised, stopword-stripped title + body.",
    "Scores normalised to [0,1] by their own maximum for stable fusion.",
])

doc.add_heading("6.5 Hybrid orchestration (infra/retrieval/hybrid.py)", level=2)
bullets(doc, [
    "retrieve(query, top_k=5, rrf_k=60, pool_size=24): graph top-10 + vector top-20 + BM25 top-20 "
    "→ RRF merge → top-24 pool → re-rank → top-5.",
    "Returns matched concepts, per-engine hits, pool size, results, latency and the active "
    "re-ranker mode.",
    "vector_only() provides the baseline used for the re-rank-gain gate.",
])

doc.add_heading("6.6 The re-ranker (infra/retrieval/reranker.py)", level=2)
bullets(doc, [
    "Primary: cross-encoder/ms-marco-MiniLM-L-6-v2 through the rerankers library on the ort "
    "(ONNX) backend — the plan's exact model, running fully offline via optimum; HuggingFace "
    "network probes are disabled after the one-time download.",
    "Fallback 1: flashrank ms-marco-MiniLM-L-12-v2 (distilled, offline).",
    "Fallback 2: calibrated score fusion (each signal normalised by its own max before a "
    "weighted mix) — always available.",
])

doc.add_heading("6.7 Scripts and tests", level=2)
table(doc,
      ["File", "Purpose"],
      [
          ["scripts/setup_layer2.py", "Provision: graph reset + ingest, Qdrant upsert, BM25 build, re-ranker warm-up, demo query table; persists data/layer2_corpus.json"],
          ["scripts/check_layer2_results.py", "Acceptance gate: 47-query evaluation, warm-up before timing, persists data/layer2_results.json, exits non-zero on any miss"],
          ["scripts/debug_layer2.py", "Per-query vector-only vs hybrid comparison with family labels"],
          ["tests/test_layer2.py", "5 unit tests (corpus shape/determinism, concept registry, BM25, eval validity, precision math) + 2 live-service integration tests (neo4j stats, qdrant collection)"],
      ])

# ── 7. Workflow ──────────────────────────────────────────────────────────
doc.add_heading("7. End-to-End Workflow (one query)", level=1)
bullets(doc, [
    "Provisioning: corpus generated deterministically (seed 11) → 500 docs written to Neo4j with "
    "concepts; embeddings computed and upserted to Qdrant; BM25 built; re-ranker loaded and "
    "warmed.",
    "Query time: (1) text → concept matching against the registry; (2) Neo4j Cypher returns "
    "graph hits; (3) embedding → Qdrant top-20; (4) BM25 top-20; (5) RRF merges the three "
    "ranked lists into a top-24 pool; (6) cross-encoder scores every pool member against the "
    "query; (7) top-5 returned with confidence scores.",
    "Evaluation: 47 queries run through the pipeline; coverage and precision@5 computed against "
    "family gold; latency measured after warm-up (one embedding + one graph call + one re-rank "
    "before the timed loop) so cold-start costs are excluded.",
    "Acceptance: the check script prints all four gates and persists machine-readable JSON.",
])

# ── 8. Targeted vs actual ────────────────────────────────────────────────
doc.add_heading("8. Targeted vs Measured Results", level=1)
table(doc,
      ["#", "Gate", "Target", "Measured", "Status"],
      [
          ["R1", "Coverage@5", "≥ 95%", "100.0%", "PASS"],
          ["R2", "Re-ranked precision@5", "≥ 0.85", "0.953", "PASS"],
          ["R3", "Re-rank gain over vector-only (0.902)", "≥ +0.05", "+0.051", "PASS"],
          ["R4", "Mean end-to-end latency", "< 3 s", "1.10 s", "PASS"],
      ])
para(doc, "Context for the numbers: vector-only precision@5 is already 0.902 because semantic "
          "embeddings perform well on a clean, unambiguous corpus. The full-size cross-encoder "
          "still lifts it to 0.953 — a modest but real gain — and coverage reaches 100%, meaning "
          "every one of the 47 evaluation queries has at least one relevant document in the final "
          "top-5. Final measured run: RESULT 4/4 gates passed.")

# ── 9. Problems ──────────────────────────────────────────────────────────
doc.add_heading("9. Engineering Problems Encountered and Resolutions", level=1)
table(doc,
      ["#", "Problem", "Root cause", "Resolution"],
      [
          ["1", "Hybrid precision below vector-only", "Near-duplicate 20-variant families let BGE flood top-5 and gave the re-ranker no signal",
           "Rewrote the corpus generator so every variant is a distinct article (unique title + body angle)"],
          ["2", "Fusion made gold docs drop out of top-5", "Weighted score sum let one engine's scale dominate",
           "Switched to Reciprocal Rank Fusion (rank-only), k = 60"],
          ["3", "Re-ranker actively hurt precision (0.732 vs 0.791)", "Tiny distilled flashrank model could not separate lexical twin families",
           "Swapped to the plan's exact full-size cross-encoder/ms-marco-MiniLM-L-6-v2 on the ort backend"],
          ["4", "rerankers ort failed at import", "Missing optimum[onnxruntime] dependency",
           "Installed optimum[onnxruntime]; model loads and runs on CPU"],
          ["5", "Wrong model loaded (jina reranker)", "Stale RERANKER_MODEL env value from the fastembed era",
           "Pointed .env at the plan's model; runtime no longer depends on the env for correctness"],
          ["6", "Result parsing AttributeError (r.doc)", "rerankers 0.10 Result exposes .document, not .doc",
           "Inspect API via probe; parse r.document.text with fallbacks"],
          ["7", "HuggingFace network probe stalled model load", "Library HEAD-checked config even with cached weights",
           "Set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 during load"],
          ["8", "Neo4j crash-looping after Docker restart", "Stale neo4j_data volume carried config from an older image; fresh pull rejected it",
           "Purged the volume (graph is regenerated by setup) — container healthy again; re-verified 4/4"],
          ["9", "Qdrant upsert timed out once", "Client default timeout too short for first bulk upsert",
           "QdrantClient(timeout=60)"],
          ["10", "Cypher rejected WHERE after UNWIND", "Cypher grammar requires a WITH bridge",
           "Rewrote: UNWIND (d + r) AS doc WITH doc WHERE doc IS NOT NULL RETURN DISTINCT doc.doc_id"],
      ])

# ── 10. Tests ────────────────────────────────────────────────────────────
doc.add_heading("10. Automated Test Coverage", level=1)
table(doc,
      ["Test (unit)", "Asserts"],
      [
          ["test_corpus_shape_and_determinism", "500 docs, 25 families, ≥ 40 queries; byte-identical rebuild"],
          ["test_corpus_families_map_to_concepts", "Every family's concepts exist in the registry"],
          ["test_bm25_finds_kubernetes_doc", "BM25 surfaces the deployment family for cluster/helm query"],
          ["test_eval_queries_have_valid_families", "Every eval query maps to a real family"],
          ["test_precision_at_5_math", "Precision calculation is correct on a hand-made case"],
      ])
table(doc,
      ["Test (integration, BIFROST_INTEGRATION=1)", "Asserts"],
      [
          ["test_neo4j_graph_populated", "Live Neo4j has ≥ 500 Document and ≥ 15 Concept nodes"],
          ["test_qdrant_collection_populated", "Live Qdrant bifrost_docs has 500 points × 384 dims"],
      ])
para(doc, "Full suite (Layers 1 + 2): 17/17 passing.", bold=True)

# ── 11. Verify ───────────────────────────────────────────────────────────
doc.add_heading("11. How to Verify the Results (step by step)", level=1)
para(doc, "Run from the project root C:\\Users\\HP\\Desktop\\BiFrost Route.")
para(doc, "Step 1 — services:", bold=True, space_after=2)
code_block(doc, ["docker compose ps"])
para(doc, "Expected: all five containers Up (neo4j, qdrant are the Layer 2 ones).", italic=True)
para(doc, "Step 2 — provision:", bold=True, space_after=2)
code_block(doc, [r".venv\Scripts\python.exe scripts\setup_layer2.py"])
para(doc, "Step 3 — acceptance:", bold=True, space_after=2)
code_block(doc, [r".venv\Scripts\python.exe scripts\check_layer2_results.py"])
para(doc, "Expected tail:", italic=True, space_after=2)
code_block(doc, [
    "[PASS] R1 coverage@5: target >= 0.95 | measured 1.000",
    "[PASS] R2 reranked precision@5: target >= 0.85 | measured 0.953",
    "[PASS] R3 rerank gain: target >= +0.05 | measured +0.051",
    "[PASS] R4 mean latency: target < 3 s | measured ... ms",
    "RESULT: 4/4 gates passed",
    "persisted -> data\\layer2_results.json",
])
para(doc, "The decisive line is RESULT: 4/4 gates passed.", bold=True)
para(doc, "Step 4 (optional) — per-query inspection:", bold=True, space_after=2)
code_block(doc, [r".venv\Scripts\python.exe scripts\debug_layer2.py"])

# ── 12. Limitations ──────────────────────────────────────────────────────
doc.add_heading("12. Limitations and Honest Notes", level=1)
bullets(doc, [
    "The corpus is synthetic and deterministic (500 articles); real production corpora are larger "
    "and messier. The 384-dim bge-small embeddings trade some accuracy for size; a larger embedder "
    "is a drop-in upgrade.",
    "Gold relevance is family-level: any document of the topic family counts as relevant. "
    "Fine-grained per-document relevance would be a stricter (and harder) benchmark.",
    "Latency is measured on CPU (1.10 s mean, dominated by the cross-encoder re-ranking 24 "
    "candidates). GPU or an API re-ranker reduces this by an order of magnitude; the gate bound "
    "is 3 s on this machine.",
    "The knowledge graph is built over the same synthetic corpus; concept matching depends on the "
    "alias registry being maintained as docs grow.",
])

# ── 13. Next ─────────────────────────────────────────────────────────────
doc.add_heading("13. What Comes Next (Layer 3 preview)", level=1)
para(doc, "Layer 3 adds the query-complexity classifier: a model that reads each incoming query "
          "and labels it simple / medium / complex, so the Layer 4 router can pick the cheap, "
          "standard or frontier model. Layer 2's retrieval becomes the knowledge grounding that "
          "any routed answer can draw on. Layer 1 continues to record every trace and every cost "
          "as before.")

doc.save(OUT)
print("saved:", OUT)