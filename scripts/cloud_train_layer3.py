#!/usr/bin/env python
"""Bifrost Route — Layer 3 self-contained cloud trainer (Kaggle / Colab).

One file, no repo needed. Run this inside a Kaggle notebook (Accelerator:
GPU T4x2) or Colab (Runtime -> Change runtime type -> T4 GPU). It will:

  1. install missing deps (torch, transformers, optimum, onnxruntime, …)
  2. build the deterministic Layer-3 dataset (or load YOUR uploaded
     layer3_dataset.json when present — enterprise dataset v2 path)
  3. fine-tune distilbert-base-uncased with class-weighted loss
  4. export the serving artifact to ONNX
  5. evaluate on the held-out eval split and write metrics.json
  6. package models/layer3_distilbert + models/layer3_onnx into
     layer3_artifacts.zip for download back to your laptop

Usage (paste whole file into one notebook cell, or run with python):
    python cloud_train_layer3.py [--epochs 5] [--batch 32]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CLASSES = ["simple", "medium", "complex"]
DATASET_JSON = Path("layer3_dataset.json")  # your upload, if any
OUT_ROOT = Path("models/layer3_distilbert")
ONNX_ROOT = Path("models/layer3_onnx")


# ── 0 · bootstrap deps ────────────────────────────────────────────────────
def bootstrap() -> None:
    missing = []
    for mod, pip in [("torch", "torch"), ("transformers", "transformers"),
                     ("onnxruntime", "onnxruntime"),
                     ("optimum", "optimum[onnxruntime]")]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip)
    if missing:
        print(f"installing: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        *missing], check=True)
    import torch
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
          f" | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")


# ── 1 · dataset ───────────────────────────────────────────────────────────
SERVICES = ["groq", "gemini", "redis", "docker", "mysql", "neo4j",
            "qdrant", "fastembed", "bifrost", "portal"]
PLANS = ["Starter", "Pro", "Enterprise"]
PREFIXES = ["", "can you ", "please ", "quickly ", "hey, ", "I need to know: ",
            "tell me ", "help me ", "would you "]
SUFFIXES = ["", " thanks", " please", " asap", "!"]
INTROS = ["", "I need to know ", "Can you explain ", "Please tell me ",
          "How do I ", "What is the best way to "]

SIMPLE_TRAIN = [
    "What is the refund policy?", "How do I reset my password?",
    "What are your support hours?", "Where is my order?",
    "What shipping options are available?", "How do I update my billing address?",
    "What is the cancellation deadline?", "How do I invite a teammate?",
    "What file formats do you support?", "Is my payment method stored securely?",
    "How many seats does the {plan} plan include?", "What is the SLA for enterprise accounts?",
    "How do I export my data?", "Where can I download the mobile app?",
    "What model powers the {svc} API?", "How long are API keys valid?",
    "Can I pause my subscription?", "What is the limit on file uploads?",
    "How do I add a payment method?", "Where do I find the invoice?",
    "Is two-factor auth available?", "What time zones are dashboards in?",
    "Do you offer a free trial?", "What happens if my card expires?",
    "How do I change my notification email?",
]
MEDIUM_TRAIN = [
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
COMPLEX_TRAIN = [
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
SIMPLE_TERSE = [
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
MEDIUM_TERSE = [
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
COMPLEX_TERSE = [
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
SIMPLE_EVAL = [
    "tell me the refund policy", "how to reset login", "when are you guys open",
    "track my order status", "which delivery options exist", "change billing address",
    "deadline to cancel", "add someone to the team", "what formats can i upload",
    "is my card safe with you", "seat count on the basic plan", "uptime promise for enterprise",
    "how do i pull my data out", "link to the phone app", "which engine powers your API",
    "key expiry duration", "suspend membership for a while", "upload size ceiling",
    "save a new card", "where's my monthly statement", "do you support 2fa",
    "dashboard timezone options", "free version available", "card declined what now",
    "switch alert email address",
]
MEDIUM_EVAL = [
    "run the numbers between the cheap and expensive tiers for us",
    "my bill jumped — walk me through why", "top 3 risks from the quarterly review",
    "team member quits halfway — who pays what", "write a reply to an unhappy customer",
    "what do we get by upgrading, concretely", "set up sso and what usually goes wrong",
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
COMPLEX_EVAL = [
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


def build_dataset() -> dict:
    import itertools

    samples: dict = {"train": [], "eval": []}
    CAP = 26
    pools = {
        "{svc}": SERVICES, "{svc2}": ["redis", "mysql", "neo4j", "qdrant", "gemini"],
        "{plan}": PLANS, "{planp}": ["Starter", "Pro", "Enterprise"],
        "{plan2}": ["Pro", "Enterprise", "Enterprise"],
    }

    def expand(text: str) -> list[str]:
        slots = [s for s in ("{svc2}", "{svc}", "{planp}", "{plan2}", "{plan}")
                 if s in text and s in pools]
        if slots:
            combos = list(itertools.product(*(pools[s] for s in slots)))
            step = max(1, len(combos) // CAP)
            out = []
            for combo in combos[::step][:CAP]:
                t = text
                for slot, val in zip(slots, combo):
                    t = t.replace(slot, val, 1)
                out.append(t)
            return out or [text]
        out = []
        for intro in INTROS:
            for suff in SUFFIXES[:3]:
                out.append(intro + text + suff)
                out.append(text.capitalize() + suff)
        seen, res = set(), []
        for v in out:
            if v not in seen:
                seen.add(v)
                res.append(v)
        return res[:CAP]

    for text in SIMPLE_TRAIN:
        for v in expand(text):
            samples["train"].append({"text": v, "label": "simple"})
    for text in SIMPLE_TERSE:
        samples["train"].append({"text": text, "label": "simple"})
    for text in MEDIUM_TRAIN:
        for v in expand(text):
            samples["train"].append({"text": v, "label": "medium"})
    for text in MEDIUM_TERSE:
        samples["train"].append({"text": text, "label": "medium"})
    for text in COMPLEX_TRAIN:
        for v in expand(text):
            samples["train"].append({"text": v, "label": "complex"})
    for text in COMPLEX_TERSE:
        samples["train"].append({"text": text, "label": "complex"})

    samples["eval"] = ([{"text": t, "label": "simple"} for t in SIMPLE_EVAL]
                       + [{"text": t, "label": "medium"} for t in MEDIUM_EVAL]
                       + [{"text": t, "label": "complex"} for t in COMPLEX_EVAL])
    train_texts = {x["text"] for x in samples["train"]}
    samples["eval"] = [x for x in samples["eval"] if x["text"] not in train_texts]
    return {"taxonomy": {}, "train": samples["train"], "eval": samples["eval"],
            "classes": CLASSES}


def load_dataset() -> dict:
    if DATASET_JSON.exists():  # your uploaded enterprise v2 dataset wins
        data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
        print(f"using uploaded dataset {DATASET_JSON} "
              f"({len(data['train'])} train / {len(data['eval'])} eval)")
        return data
    data = build_dataset()
    DATASET_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"built dataset: {len(data['train'])} train / {len(data['eval'])} eval")
    return data


# ── 1b · TF-IDF baseline (fallback + gate-C4 comparator) ──────────────────
def train_tfidf_lr() -> None:
    """Fit TF-IDF + LogisticRegression on the train split (sklearn, CPU, secs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data = load_dataset()
    texts = [x["text"] for x in data["train"]]
    labels = [CLASSES.index(x["label"]) for x in data["train"]]

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, lowercase=True)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    lr.fit(X, labels)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(vec, OUT_ROOT / "tfidf_vec.joblib")
    joblib.dump(lr, OUT_ROOT / "lr.joblib")
    print("tfidf-lr baseline trained")


# ── 2 · training ──────────────────────────────────────────────────────────
def train(epochs: int, batch: int) -> dict:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    data = load_dataset()
    labels = [CLASSES.index(x["label"]) for x in data["train"]]
    texts = [x["text"] for x in data["train"]]
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)
    w = [total / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))]
    class_weights = torch.tensor(w, dtype=torch.float32)
    print("class weights:", dict(zip(CLASSES, [round(v, 3) for v in w])))

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)})

    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)

    class DS:
        def __init__(self, e, l):
            self.e, self.l = e, l

        def __getitem__(self, i):
            return {k: torch.tensor(v[i]) for k, v in self.e.items()} \
                | {"labels": torch.tensor(self.l[i])}

        def __len__(self):
            return len(self.l)

    class WeightedTrainer(Trainer):
        def __init__(self, cw, *a, **k):
            super().__init__(*a, **k)
            self.cw = cw

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=self.cw)
            return (loss, outputs) if return_outputs else loss

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(OUT_ROOT / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        fp16=use_cuda,       # T4/A100 all support fp16; 2-3x faster on GPU
        seed=11,
        disable_tqdm=False,
    )
    trainer = WeightedTrainer(class_weights=class_weights, model=model,
                              args=args, train_dataset=DS(enc, labels))
    t0 = time.perf_counter()
    trainer.train()
    train_secs = round(time.perf_counter() - t0, 1)
    print(f"training done in {train_secs}s")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_ROOT))
    tokenizer.save_pretrained(str(OUT_ROOT))

    onnx_ok = False
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        ort = ORTModelForSequenceClassification.from_pretrained(
            str(OUT_ROOT), export=True)
        ONNX_ROOT.mkdir(parents=True, exist_ok=True)
        ort.save_pretrained(str(ONNX_ROOT))
        tokenizer.save_pretrained(str(ONNX_ROOT))
        onnx_ok = True
        print("ONNX export OK")
    except Exception as exc:  # noqa: BLE001
        print("ONNX export failed (torch artifact still saved):", exc)

    stats = evaluate()
    stats["train_secs"] = train_secs
    stats["onnx_exported"] = onnx_ok
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ── 3 · evaluation ────────────────────────────────────────────────────────
def evaluate() -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = load_dataset()
    model = AutoModelForSequenceClassification.from_pretrained(str(OUT_ROOT))
    tokenizer = AutoTokenizer.from_pretrained(str(OUT_ROOT))
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    from collections import Counter

    correct = 0
    conf = {c: Counter() for c in CLASSES}
    lats: list[float] = []
    with torch.no_grad():
        for sample in data["eval"]:
            t0 = time.perf_counter()
            enc = tokenizer(sample["text"], return_tensors="pt",
                            truncation=True, max_length=128)
            if use_cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = model(**enc).logits
            pred = CLASSES[int(logits.argmax(-1).item())]
            lats.append((time.perf_counter() - t0) * 1000.0)
            if pred == sample["label"]:
                correct += 1
            conf[sample["label"]][pred] += 1

    acc = correct / len(data["eval"])
    rows = {}
    for label in CLASSES:
        tp = conf[label][label]
        fp = sum(conf[o][label] for o in CLASSES if o != label)
        fn = sum(conf[label][o] for o in CLASSES if o != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows[label] = {"precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}
    lats.sort()
    print(f"\naccuracy: {acc:.4f}")
    for label, m in rows.items():
        print(f"  {label:7s} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f}")
    return {"accuracy": round(acc, 4), "per_class": rows,
            "latency_ms": {"mean": round(sum(lats) / len(lats), 2),
                           "p95": round(lats[int(len(lats) * 0.95)], 2)},
            "eval_n": len(data["eval"]), "mode": "distilbert-torch"}


# ── 4 · packaging ─────────────────────────────────────────────────────────
def package() -> None:
    zip_path = Path("layer3_artifacts.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (OUT_ROOT, ONNX_ROOT):
            if root.exists():
                prefix = f"models/{root.name}"  # unzip at project root ->
                for f in root.rglob("*"):       #   models/layer3_distilbert + _
                    if f.is_file() and "checkpoints" not in f.parts:
                        z.write(f, f"{prefix}/{f.relative_to(root)}")
    size_mb = round(zip_path.stat().st_size / 1_048_576, 1)
    print(f"\npackaged {zip_path} ({size_mb} MB)")
    print("DOWNLOAD:")
    if Path("/kaggle").exists():
        print("  Kaggle: open output panel (\u203a\u203a\u203a) and click download, "
              "or: from IPython.display import FileLink; FileLink('layer3_artifacts.zip')")
    elif Path("/content").exists():
        print("  Colab: from google.colab import files; files.download('layer3_artifacts.zip')")
    else:
        print(f"  local: {zip_path.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    print("=" * 72)
    print("BIFROST ROUTE · LAYER 3 CLOUD TRAINER (Kaggle T4x2 / Colab T4)")
    print("=" * 72)
    bootstrap()
    train_tfidf_lr()
    stats = train(epochs=a.epochs, batch=a.batch)
    macro_f1 = sum(v["f1"] for v in stats["per_class"].values()) / 3
    print(f"\nfinal: accuracy {stats['accuracy']} | macro-F1 {macro_f1:.4f} | "
          f"latency mean {stats['latency_ms']['mean']} ms | onnx {stats['onnx_exported']}")
    package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())