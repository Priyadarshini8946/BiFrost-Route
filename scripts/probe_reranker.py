"""Flashrank availability probe — proves the cross-encoder works offline
and reports its exact result shape (dict vs object) with real scored output.
"""
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flashrank import Ranker, RerankRequest  # noqa: E402

fr = Ranker()  # default ms-marco-MiniLM-L-12-v2 (ONNX, offline)
print("flashrank: model ready (default ms-marco-MiniLM-L-12-v2)")
out = fr.rerank(RerankRequest(
    query="how do i reset my password",
    passages=[
        {"id": "DOC-019", "text": "How to reset the bifrost password. To reset the bifrost password, open the account menu"},
        {"id": "DOC-800", "text": "Where do I find the bifrost monthly invoice. Monthly invoices for bifrost appear in Billing"},
    ],
))
items = out if isinstance(out, list) else getattr(out, "results", out)
print("shape:", "list" if isinstance(items, list) else type(items).__name__)
for item in items[:2]:
    if isinstance(item, dict):
        print("  ", item.get("id"), round(float(item.get("score", 0.0)), 4))
    else:
        print("  ", getattr(item, "id", None), round(float(getattr(item, "score", 0.0)), 4))
print("OK")
