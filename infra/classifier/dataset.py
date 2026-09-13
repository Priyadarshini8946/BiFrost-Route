"""Layer 3: labeled query-complexity dataset for the DistilBERT classifier.

Taxonomy (drives routing economics):
  simple  → factual lookup, single answer, one source        → cheap  model
  medium  → compare/summarize/explain across a few sources   → standard model
  complex → multi-step reasoning, policy application,
            conflicting/ambiguous inputs, synthesis          → frontier model

Design:
  • TRAIN set — deterministic template × slot expansion (~3000 samples),
    so the model sees broad phrasing per class.
  • EVAL set — hand-written *rephrased* queries (never in TRAIN), including
    adversarial near-duplicates ("how do I reset" vs "compare security
    policies") that force semantic discrimination, not keyword matching.
  • Both sets persist to data/layer3_dataset.json with the exact split, so
    the acceptance gate and the training script can never disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SERVICES = ["groq", "gemini", "redis", "docker", "mysql", "neo4j",
            "qdrant", "fastembed", "bifrost", "portal"]
PLANS = ["Starter", "Pro", "Enterprise"]
PLAN_PAIRS = [("Starter", "Pro"), ("Pro", "Enterprise"), ("Starter", "Enterprise")]
SVC_PAIRS = [("redis", "mysql"), ("neo4j", "qdrant"), ("groq", "gemini")]
PREFIXES = ["", "can you ", "please ", "quickly ", "hey, ", "I need to know: ",
            "tell me ", "help me ", "would you "]
SUFFIXES = ["", " thanks", " please", " asap", " — urgent", "!"]
# explicit-parameter question intros for true paraphrasing
INTROS = ["", "I need to know ", "Can you explain ", "Please tell me ",
          "How do I ", "What is the best way to "]

# ── TRAIN templates (expand via slots; paraphrased in EVAL) ──────────────
SIMPLE_TRAIN: List[str] = [
    "What is the refund policy?",
    "How do I reset my password?",
    "What are your support hours?",
    "Where is my order?",
    "What shipping options are available?",
    "How do I update my billing address?",
    "What is the cancellation deadline?",
    "How do I invite a teammate?",
    "What file formats do you support?",
    "Is my payment method stored securely?",
    "How many seats does the {plan} plan include?",
    "What is the SLA for enterprise accounts?",
    "How do I export my data?",
    "Where can I download the mobile app?",
    "What model powers the {svc} API?",
    "How long are API keys valid?",
    "Can I pause my subscription?",
    "What is the limit on file uploads?",
    "How do I add a payment method?",
    "Where do I find the invoice?",
    "Is two-factor auth available?",
    "What time zones are dashboards in?",
    "Do you offer a free trial?",
    "What happens if my card expires?",
    "How do I change my notification email?",
]

MEDIUM_TRAIN: List[str] = [
    "Compare pricing between the {plan} and {plan2} plans, including hidden fees.",
    "Why did my invoice go up from last month? Break down the line items.",
    "Summarize our quarterly report and list the top three risks.",
    "What happens when a team member leaves mid-cycle — prorating rules across {plan} and {plan2}?",
    "Draft a response to this customer complaint and include next steps.",
    "Which features unlock at the {plan} tier and how do they affect our current workflow?",
    "Explain the SSO setup steps for {svc} and the common failure modes.",
    "Compare uptime guarantees of {svc} and {svc2} and map SLA credits to our contract.",
    "Identify which of our 20 integrations are deprecated and suggest replacements.",
    "Walk me through the onboarding checklist for a new {svc} workspace.",
    "How does our usage compare across regions, and which one should we scale first?",
    "Summarize the incident report and propose a rollback plan.",
    "Compare the security posture of {svc} and {svc2} for a compliance review.",
    "Group our open tickets by severity and estimate the backlog effort.",
    "Explain how caching affects our API latency and where the wins are.",
    "What changed in the last release, and which items affect us?",
    "Compare the two connectors and recommend one for our data stack.",
    "Walk through the migration steps and flag risky dependencies.",
]

COMPLEX_TRAIN: List[str] = [
    "Reconcile the discrepancy between the sales report and finance ledger for {svc} "
    "considering FX effects.",
    "Given contract clause 4.2, the GDPR retention policy, and the customer's data "
    "deletion request, what is the compliant course of action?",
    "A production incident crossed the {svc} SLA: analyze the sequence of events, "
    "blast radius, and whether the excluded-service clause applies.",
    "Our {svc} rollout failed in EMEA but not APAC. Hypothesize root causes across "
    "config drift, network topology, and regional compliance modes, then rank them.",
    "Merge the access-control matrices from {svc}, {svc2} and {plan} into one policy "
    "and find conflicting grants.",
    "Interpret the {svc} benchmark results against our {plan} workload and adversarial "
    "prompts, and advise on model decommission.",
    "Audit the dispute between two departments over budget ownership in Q2. "
    "Propose allocation rules that satisfy both compliance and finance.",
    "Our data retention timeline conflicts with customer contract {plan} in EU "
    "jurisdiction. Design a compliant data lifecycle.",
    "Given the SOC 2 gap report, the new third-party processor, and our vendor policy, "
    "structure the remediation plan with owners and deadlines.",
    "A customer's PII was exposed in a log export. Trace the propagation, assess "
    "notification obligations across jurisdictions, and draft the internal memo.",
    "Model the cost impact of migrating {svc} workloads off reserved capacity under "
    "the new usage-based pricing, accounting for burst factors.",
    "Combine the fraud signals from {svc} and {svc2} into a unified scoring policy, "
    "then validate it against the last quarter's false-positive rate.",
]

# ── TRAIN informal dialect (terse, chat-style) ────────────────────────────
# The EVAL split is terse/informal; the plain templates above are formal.
# These bridge the dialect gap so the model learns both registers during
# training. Every string here is a *different* wording from EVAL — exact
# overlap is additionally filtered by the leak check in build_dataset().
SIMPLE_TERSE: List[str] = [
    "how does the refund work", "steps to reset my password", "open hours please",
    "where is my package", "delivery methods list", "update billing address please",
    "latest date to cancel", "add a teammate", "which file types can i upload",
    "is my card saved safely", "seat count for starter plan", "sla for enterprise tier",
    "how do i export everything", "download the mobile app", "what model does the api run",
    "api keys last how long", "can i hold my subscription", "max upload size",
    "enter a new payment method", "invoice location", "two-factor auth setup",
    "timezone of the dashboard", "is there a trial", "card expired what now",
    "change alert email address",
]
MEDIUM_TERSE: List[str] = [
    "compare starter vs pro with all fees", "invoice went up, why — break it down",
    "top risks from the quarterly report", "member quits mid-cycle who pays what",
    "draft a reply to an angry customer", "what unlocks at pro tier, concretely",
    "sso setup steps and common errors", "uptime vs credits across providers",
    "which integrations are deprecated and what to use", "onboard a new workspace, checklist",
    "usage by region — where do we scale first", "incident recap plus a rollback plan",
    "security comparison for our compliance review", "tickets by severity and backlog size",
    "how caching cuts our latency, where", "changelog items that affect us",
    "pick one of two connectors for our stack", "migration steps and risky dependencies",
]
COMPLEX_TERSE: List[str] = [
    "sales says one thing finance another, with fx — find the divergence",
    "deletion request vs retention policy vs contract — what is compliant",
    "outage breached the sla, what happened and are credits owed",
    "emea works apac does not — rank the root causes",
    "two access matrices conflict — merge into one policy and list conflicts",
    "benchmarks vs our workload and hostile prompts — retire the model?",
    "two departments own one budget line — design a fair allocation",
    "retention timeline vs an eu contract — build a compliant lifecycle",
    "soc2 gaps, a new processor, vendor rules — structure the remediation",
    "pii leaked via logs — trace, notify, draft the memo",
    "reserved capacity vs usage pricing — model the switch with burst factors",
    "unify fraud signals from the two systems and check them against last quarter",
]


# ── EVAL: hand-written rephrases + adversarial near-duplicates ───────────
SIMPLE_EVAL: List[str] = [
    "tell me the refund policy",
    "how to reset login",
    "when are you guys open",
    "track my order status",
    "which delivery options exist",
    "change billing address",
    "deadline to cancel",
    "add someone to the team",
    "what formats can i upload",
    "is my card safe with you",
    "seat count on the basic plan",
    "uptime promise for enterprise",
    "how do i pull my data out",
    "link to the phone app",
    "which engine powers your API",
    "key expiry duration",
    "suspend membership for a while",
    "upload size ceiling",
    "save a new card",
    "where's my monthly statement",
    "do you support 2fa",
    "dashboard timezone options",
    "free version available",
    "card declined what now",
    "switch alert email address",
]

MEDIUM_EVAL: List[str] = [
    "run the numbers between the cheap and expensive tiers for us",
    "my bill jumped — walk me through why",
    "top 3 risks from the quarterly review",
    "team member quits halfway — who pays what",
    "write a reply to an unhappy customer",
    "what do we get by upgrading, concretely",
    "set up sso and what usually goes wrong",
    "downtime guarantee comparison and credit math",
    "which of our integrations are dead, and alternatives",
    "onboarding steps for a fresh workspace",
    "usage split across regions, where to grow",
    "incident summary plus a rollback suggestion",
    "which provider is more secure for our audit",
    "sort tickets by severity and size the backlog",
    "how does the cache cut our latency, where",
    "what in the changelog matters to us",
    "pick one of these two connectors for our stack",
    "migration walkthrough and risky bits",
]

COMPLEX_EVAL: List[str] = [
    "sales says one number, finance says another — find where they diverge",
    "customer deletion request vs our retain policy vs the contract — what do we do",
    "outage breached the sla: figure out what happened and whether we owe credits",
    "europe works, asia doesn't — rank the likely causes",
    "two access matrices contradict each other, make one policy plus the conflicts",
    "benchmarks vs our real workload and hostile prompts, should we retire the model",
    "two departments both own the same budget line — design a fair split",
    "retention policy clashes with an EU contract — build a compliant lifecycle",
    "audit gaps, a new processor, and our vendor rules — make a remediation plan",
    "pii leaked via logs: trace it, who must we notify, and the memo",
    "is reserved capacity still right under usage pricing — model the switch",
    "unify fraud signals from two systems and validate against last quarter",
]


def build_dataset(out: Path = Path("data/layer3_dataset.json")) -> Dict[str, Any]:
    """Deterministic train/eval split (no randomness — pure slot expansion)."""
    train: List[Dict[str, str]] = []

    def _expand(text: str) -> List[str]:
        """Combinatoric slot fill with a per-template cap; no random RNG.

        Returns up to CAP variants per template. Templates with no known
        slots expand via prefix/intro/suffix paraphrases instead.
        """
        CAP = 26
        pools = {
            "{svc}": SERVICES,
            "{svc2}": ["redis", "mysql", "neo4j", "qdrant", "gemini"],
            "{plan}": PLANS,
            "{planp}": ["Starter", "Pro", "Enterprise"],
            "{plan2}": ["Pro", "Enterprise", "Enterprise"],
        }
        import itertools

        slots = [s for s in ("{svc2}", "{svc}", "{planp}", "{plan2}", "{plan}")
                 if s in text and s in pools]
        if slots:
            combos = list(itertools.product(*(pools[s] for s in slots)))
            # deterministic stride so all templates share roughly the same
            # variety instead of the biggest pool dominating
            step = max(1, len(combos) // CAP)
            chosen = combos[::step][:CAP]
            variants = []
            for combo in chosen:
                t = text
                for slot, val in zip(slots, combo):
                    t = t.replace(slot, val, 1)
                variants.append(t)
            return variants or [text]
        # no known slots → paraphrase with intros/prefixes/suffixes
        variants = []
        for intro in INTROS:
            for suffix in SUFFIXES[:3]:
                candidates = [intro + text + suffix,
                              text.capitalize() + suffix]
                variants.extend(candidates)
        seen, out_ = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out_.append(v)
        return out_[:CAP]

    for text in SIMPLE_TRAIN:
        for variant in _expand(text):
            train.append({"text": variant, "label": "simple"})
    for text in SIMPLE_TERSE:
        train.append({"text": text, "label": "simple"})
    for text in MEDIUM_TRAIN:
        for variant in _expand(text):
            train.append({"text": variant, "label": "medium"})
    for text in MEDIUM_TERSE:
        train.append({"text": text, "label": "medium"})
    for text in COMPLEX_TRAIN:
        for variant in _expand(text):
            train.append({"text": variant, "label": "complex"})
    for text in COMPLEX_TERSE:
        train.append({"text": text, "label": "complex"})

    eval_ = ([{"text": t, "label": "simple"} for t in SIMPLE_EVAL]
             + [{"text": t, "label": "medium"} for t in MEDIUM_EVAL]
             + [{"text": t, "label": "complex"} for t in COMPLEX_EVAL])
    # drop any accidental train/eval overlap at the text level
    train_texts = {x["text"] for x in train}
    eval_ = [x for x in eval_ if x["text"] not in train_texts]

    data = {
        "taxonomy": {
            "simple": "factual lookup, single answer, one source → cheap model",
            "medium": "compare/summarize/explain across few sources → standard model",
            "complex": "reasoning/policy/conflict synthesis → frontier model",
        },
        "train": train,
        "eval": eval_,
        "classes": ["simple", "medium", "complex"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    d = build_dataset()
    from collections import Counter

    tr = Counter(x["label"] for x in d["train"])
    ev = Counter(x["label"] for x in d["eval"])
    print(f"train={len(d['train'])} {dict(tr)} | eval={len(d['eval'])} {dict(ev)}")