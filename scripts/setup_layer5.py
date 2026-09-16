"""Layer-5 provisioning — verifies the RAGAS harness is ready to run.

Checks (fails fast, exit 1):
  1. ragas + langchain installed (real-judge stack) — or explains the
     offline proxy fallback which needs nothing extra
  2. RouteEngine + Layer-4 infra up (cache engine)
  3. collect() + proxy scoring on a 3-query smoke sample
  4. judge resolution: groq / gemini (user key) / proxy (offline)
Prints the exact command the user runs next: check_layer5_results.py.
No model is trained or downloaded here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infra.eval.ragas_harness import JudgeConfig, RagasHarness  # noqa: E402
from infra.routing.router import RouteEngine  # noqa: E402


def main() -> int:
    print("== Layer 5 provisioning (RAGAS quality harness) ==")

    # 1. judge stack
    try:
        import ragas  # noqa: F401
        from importlib.metadata import version

        print(f"  [ok] ragas {version('ragas')} installed")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] ragas unavailable ({e}) — real-judge mode disabled; "
              "offline proxy mode still works")
    cfg = JudgeConfig()
    print(f"  [ok] judge resolution: mode={cfg.mode} "
          f"(groq={bool(cfg.groq_key)} gemini={bool(cfg.gemini_key)}) "
          f"| real={'yes' if cfg.available() else 'offline proxy'}")

    # 2+3. engine smoke through the real pipeline
    eng = RouteEngine(force_dry_run=True, use_cache=False)
    eng.route("warm the stack")
    print(f"  [ok] RouteEngine warm (llm={eng.llm_mode})")

    h = RagasHarness(engine=eng)
    from infra.retrieval.corpus import build_corpus

    smoke_qs = [q["query"] for q in build_corpus()["queries"]][:3]
    rows = h.collect(smoke_qs)
    q = h.quality(rows)
    print(f"  [ok] smoke sample n={q['n']} "
          f"faithfulness={q['faithfulness']:.3f} "
          f"relevancy={q['answer_relevancy']:.3f} "
          f"context_precision={q['context_precision']:.3f} "
          f"judge={q['judge_mode']}")

    print("\nLayer 5 ready. Next commands:")
    print("  .venv\\Scripts\\python.exe scripts\\check_layer5_results.py")
    if not cfg.available():
        print("\n[optional] To use a real LLM judge (no training, just an API")
        print("key — 2 minutes): create a free key at console.groq.com or")
        print("aistudio.google.com, then add to .env:")
        print("  GROQ_API_KEY=<your-key>   (or GEMINI_API_KEY=<your-key>)")
        print("Re-run setup; gates then score with real RAGAS faithfulness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())