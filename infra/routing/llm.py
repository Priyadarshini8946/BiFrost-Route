"""Bifrost Route · Layer 4 — LLM client abstraction.

Two modes:
  • REAL   — calls Groq (llama-3.3-70b-versatile) and Google Gemini
            (gemini-2.0-flash / gemini-2.5-pro) through httpx, using keys
            from .env (GROQ_API_KEY / GEMINI_API_KEY). Opaque: the router
            sees only (model_name → completion text).
  • DRY-RUN (default when no keys) — deterministic MockLLM that grounds
            answers in the retrieved documents, so every acceptance gate
            runs today without any API key. Marked in trace as
            provider="mock".

Costs are computed by the router from the Layer-1 model registry; this
module never deals with money.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Call the model registry's providers over HTTP (OpenAI-compatible)."""

    PROVIDER_CONFIG = {
        # model_name        → (base_url, api_key_env, model_key)
        "llama-3.3-70b-versatile": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        "gemini-2.0-flash": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
        "gemini-2.5-pro": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    }

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def available(self, model_name: str) -> bool:
        cfg = self.PROVIDER_CONFIG.get(model_name)
        if cfg is None:
            return False
        _base, env = cfg
        return bool(os.environ.get(env))

    def complete(self, model_name: str, system: str, user: str,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        cfg = self.PROVIDER_CONFIG.get(model_name)
        if cfg is None:
            raise LLMError(f"unknown model {model_name!r}")
        base, env = cfg
        key = os.environ.get(env)
        if not key:
            raise LLMError(f"{env} is not set — cannot call {model_name}")
        resp = self._client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        if resp.status_code != 200:
            raise LLMError(f"{model_name} HTTP {resp.status_code}: "
                           f"{resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as exc:  # noqa: BLE001
            raise LLMError(f"bad completion payload from {model_name}: {exc}") from exc

    def close(self) -> None:
        self._client.close()


class MockLLM:
    """Deterministic dry-run generator — no network, no keys.

    Builds a grounded answer from the retrieved documents (title + body
    excerpt), so faithfulness is high when retrieval found context. When no
    document is retrieved the answer is generic → low faithfulness → the
    escalation path is exercised exactly like with a hallucinating model.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.calls: Dict[str, int] = {}

    @staticmethod
    def available(model_name: str) -> bool:
        return True

    def complete(self, model_name: str, system: str, user: str,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        self.calls[model_name] = self.calls.get(model_name, 0) + 1
        # Deterministic failure simulator (dry-run only): a query containing
        # __HALLUCINATE__ makes the mock return ungrounded text so the
        # escalation path is exercised exactly like with a hallucinating LLM.
        if "__HALLUCINATE__" in user:
            return ("The answer is a complete fabrication: the moon is made of "
                    "cheese and Bifrost runs on quantum crystals.")
        # The router injects a compact context block "@@CTX@@ ... @@ENDCTX@@".
        title, body = "", ""
        marker = "@@CTX@@"
        if marker in system:
            payload = system.split(marker, 1)[1].split("@@ENDCTX@@", 1)[0]
            for line in payload.strip().splitlines():
                line = line.strip()
                if line.startswith("T:"):
                    title = line[2:].strip()
                elif line.startswith("B:"):
                    body = line[2:].strip()
        if title or body:
            ans = f"{title}: {body[:600]}"
        else:
            ans = "I do not have specific information on that in the knowledge base."
        return ans.strip()


def make_llm(force_dry_run: bool = False) -> Any:
    """Prefer real providers when keys exist; else the deterministic mock."""
    if not force_dry_run:
        probe = LLMClient()
        for model in ("llama-3.3-70b-versatile", "gemini-2.0-flash",
                      "gemini-2.5-pro"):
            if probe.available(model):
                return probe
    return MockLLM()


if __name__ == "__main__":
    m = make_llm(force_dry_run=True)
    print("mode:", type(m).__name__)
    print(m.complete("gemini-2.5-pro",
                     "@@CTX@@\nT: Password reset\nB: You can reset via Settings.\n@@ENDCTX@@",
                     "how do I reset my password?"))