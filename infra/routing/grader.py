"""Bifrost Route · Layer 4 — faithfulness grader (RAGAS-proxy).

Layer 5 installs the real RAGAS harness; Layer 4 needs a deterministic,
explainable quality gate *now* so the escalation path is real and testable:

  faithfulness ≈ answer-coverage: the fraction of answer tokens that the
  retrieved context supports. A grounded answer (built from the docs) scores
  ~0.95+; a hallucinated/generic answer scores far below the 0.85 gate and
  triggers escalation. Deterministic, offline, identical for mock and real
  providers.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from infra.cache.embedder import TfidfEmbedder, cosine


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


class FaithfulnessGrader:
    def __init__(self, embedder: Optional[TfidfEmbedder] = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        if not self.embedder._fitted:  # noqa: SLF001 — package contract
            self.embedder.fit(["seed corpus for the faithfulness grader",
                               "bifrost route layer 4 grounded answers"])

    @staticmethod
    def _context_texts(contexts: List[Dict]) -> List[str]:
        return ([c.get("body", "") for c in contexts] +
                [c.get("title", "") for c in contexts])

    def grade(self, answer: str, contexts: List[Dict],
              threshold: float = 0.850) -> Dict:
        """Return {score, verdict} — score < threshold triggers escalation."""
        if not answer.strip():
            return {"score": 0.0, "verdict": "fail"}
        texts = self._context_texts(contexts)
        if not texts or not any(t.strip() for t in texts):
            return {"score": 0.0, "verdict": "fail",
                    "reason": "no retrieved context"}

        ans_tok = _tokens(answer)
        if not ans_tok:
            return {"score": 0.0, "verdict": "fail", "reason": "empty answer"}

        union = set()
        for text in texts:
            union.update(_tokens(text))

        coverage = len(ans_tok & union) / len(ans_tok)

        # embedding signal as secondary evidence (may be poor on tiny vocab)
        v = self.embedder.embed(answer)
        best_cos = 0.0
        for text in texts:
            if text.strip():
                best_cos = max(best_cos, float(cosine(v, self.embedder.embed(text))))

        score = round(max(coverage, 0.5 * best_cos), 4)
        return {
            "score": score,
            "verdict": "pass" if score >= threshold else "fail",
            "coverage": round(coverage, 4),
            "best_cosine": round(best_cos, 4),
            "answer_tokens": len(ans_tok),
            "context_tokens": len(union),
        }


if __name__ == "__main__":
    g = FaithfulnessGrader()
    ctx = [{"title": "Password reset",
            "body": "You can reset your password from Settings Security. "
                    "Click the reset link and choose a new password."}]
    grounded = g.grade("Password reset: You can reset your password from "
                       "Settings Security. Click the reset link and choose "
                       "a new password.", ctx)
    print("grounded:", grounded)
    generic = g.grade("I do not have specific information on that in the "
                      "knowledge base.", ctx)
    print("generic:", generic)