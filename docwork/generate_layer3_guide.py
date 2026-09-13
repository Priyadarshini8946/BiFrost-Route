"""Generate the Layer 3 training runbook (.docx) on the user's Desktop."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

OUT = r"C:\Users\HP\Desktop\Bifrost_Route_Layer3_Training_Guide.docx"

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


def code(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)


doc = Document()
style_body(doc)

title = doc.add_heading("Bifrost Route — Layer 3 Training Guide", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "Query-Complexity Classifier (simple / medium / complex) · DistilBERT fine-tune",
     size=12, bold=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ── 1. What we are training ──────────────────────────────────────────────
doc.add_heading("1. What You Are Training (and Why)", level=1)
para(doc, "Layer 3 builds the query-complexity classifier — the brain that decides whether a "
          "routing query is simple (→ cheap model), medium (→ standard model) or complex "
          "(→ frontier model). It is a fine-tuned distilbert-base-uncased sequence classifier "
          "(66M parameters — tiny), trained on ~1,000+ labeled queries and evaluated on 55 "
          "hand-written held-out queries that are never seen during training.")
para(doc, "Your run produces two artifacts:")
bullets(doc, [
    "models/layer3_distilbert/ — the PyTorch fine-tuned model (for evaluation and serving fallback)",
    "models/layer3_onnx/ — the ONNX export (the fast serving artifact; ~22 ms per query on CPU)",
    "Both go into layer3_artifacts.zip for you to download back to your laptop.",
])

# ── 2. Platform recommendation ───────────────────────────────────────────
doc.add_heading("2. Which Platform to Use (2026 facts)", level=1)
table(doc,
      ["", "Kaggle (free)", "Google Colab (free)", "Colab Pro (~$10/mo)"],
      [
          ["Free GPU quota", "30 hours/week — most generous", "Unpublished, dynamic", "100 compute units"],
          ["Hardware", "T4 ×2 (32 GB) or P100 (16 GB)", "T4 (sometimes K80), not guaranteed", "Guaranteed T4, L4/A100 on demand"],
          ["Session", "12 h", "12 h", "24 h"],
          ["Phone verification", "Required for GPU", "Not required", "Not required"],
          ["Verdict for this job", "Best: 30h/week, T4x2", "Fine for a 6-min job", "Only needed for heavy work"],
      ])
para(doc, "Recommendation: KAGGLE, Accelerator = “GPU T4 ×2”. DistilBERT is small — training "
          "takes roughly 5–10 minutes on a T4, so An A100/H100 (13× faster but 13× the burn) is "
          "overkill for this model. If you do not want phone verification, Colab free with the "
          "T4 GPU runtime works just as well for this size of model.", bold=True)

# ── 3. Dataset ───────────────────────────────────────────────────────────
doc.add_heading("3. The Dataset (student level → research level)", level=1)
doc.add_heading("3.1 What ships with the repo (deterministic v1)", level=2)
para(doc, "1017 labeled queries (485 simple / 315 medium / 217 complex) with a 55-query "
          "hand-written held-out eval. Deterministic — identical on every machine. This is "
          "already fixed and verified (byte-identical between local and cloud).")
doc.add_heading("3.2 Research-grade upgrade (optional, recommended)", level=2)
para(doc, "Dataset v2 follows the RouteLLM recipe (LMSYS / UC Berkeley, the paper that invented "
          "LLM routing — arxiv.org/abs/2406.18665): real user prompts + LLM-judge labels.")
bullets(doc, [
    "Real queries: sample from the public LMSYS Chatbot Arena 55k corpus "
    "(lmsys/lmsys-arena-human-preference-55k on Hugging Face) — actual human questions.",
    "Judge labels: label each query simple/medium/complex with an LLM judge (Groq "
    "llama-3.3-70b-versatile free tier, or Gemini Flash) using a strict rubric — a few dollars "
    "for ~3,000 labels.",
    "Merge + guard: the 55 eval queries stay untouched and are leak-checked out of training.",
])
code(doc, ["pip install datasets",
           "$env:GROQ_API_KEY=\"your-free-groq-key\"   # or GEMINI_API_KEY",
           r".venv\Scripts\python.exe scripts\build_l3_enterprise_dataset.py --source arena --n 3000",
           "# → data\layer3_dataset_v2.json",
           "# then upload that file to Kaggle/Colab as:  layer3_dataset.json"])
para(doc, "The trainer auto-detects layer3_dataset.json next to it and uses YOUR dataset "
          "instead of the built-in one.")

# ── 4. Option A: Kaggle step-by-step ─────────────────────────────────────
doc.add_heading("4. Option A — Train on Kaggle (recommended)", level=1)
para(doc, "Step A1 — Account:", bold=True, space_after=2)
bullets(doc, [
    "Sign up at kaggle.com (email or Google account).",
    "Verify your phone number (Settings → Phone Verification) — REQUIRED before GPU unlocks.",
])
para(doc, "Step A2 — Create a notebook:", bold=True, space_after=2)
bullets(doc, [
    "Kaggle → Notebooks → New Notebook.",
    "Right panel: set Accelerator = GPU T4 ×2 (also Internet ON).",
    "Name it bifrost-layer3-train.",
])
para(doc, "Step A3 — Paste the trainer (one file, self-contained):", bold=True, space_after=2)
bullets(doc, [
    "On your laptop, open scripts\\cloud_train_layer3.py (the file I built for you).",
    "Select all → copy; in Kaggle, paste it into the first code cell.",
    "(Optional, dataset v2) In the Kaggle left panel use “Add → Upload” to upload "
    "layer3_dataset.json next to the notebook.",
])
para(doc, "Step A4 — Run:", bold=True, space_after=2)
code(doc, ["# the pasted script does everything: installs deps, trains TF-IDF baseline,",
           "# fine-tunes DistilBERT, evaluates, exports ONNX, zips artifacts",
           "# just press Run (or Shift+Enter). Watch for:",
           "#   final: accuracy 0.9xx | macro-F1 0.8xx | onnx True"])
para(doc, "Expected runtime: ~5–10 minutes on T4. There is also a run button ▶ top-left of the "
          "cell (or the whole notebook via Run All).")
para(doc, "Step A5 — Download results:", bold=True, space_after=2)
bullets(doc, [
    "Open the Output panel (››› arrow, bottom-right of the notebook).",
    "Download layer3_artifacts.zip (≈ 270 MB).",
])
para(doc, "Step A6 — Bring it home:", bold=True, space_after=2)
code(doc, ["# on your laptop, in the project folder C:\\Users\\HP\\Desktop\\BiFrost Route",
           "Expand-Archive layer3_artifacts.zip -DestinationPath . -Force",
           "# → creates models\\layer3_distilbert\\ and models\\layer3_onnx\\",
           "Remove-Item layer3_artifacts.zip"])

# ── 5. Option B: Colab ───────────────────────────────────────────────────
doc.add_heading("5. Option B — Train on Google Colab", level=1)
bullets(doc, [
    "colab.research.google.com → New notebook (Google account; no verification needed).",
    "Runtime → Change runtime type → Hardware accelerator = T4 GPU.",
    "Paste scripts\\cloud_train_layer3.py into the first cell; press ▶.",
    "When it finishes, run the printed line to download:",
])
code(doc, ["from google.colab import files",
           "files.download('layer3_artifacts.zip')   # then unzip as in A6"])
para(doc, "Same 5–10 minute runtime. Colab's free GPU can be unavailable at peak hours — retry "
          "or use Kaggle if that happens.")

# ── 6. Option C: local CPU ───────────────────────────────────────────────
doc.add_heading("6. Option C — Train on Your Laptop (no cloud)", level=1)
para(doc, "Slower (15–40 min) but fully local, zero upload/download:")
code(doc, [r".venv\Scripts\python.exe scripts\train_layer3.py"])
para(doc, "This trains the TF-IDF baseline and DistilBERT, exports ONNX, and writes "
          "metrics.json — no cloud involved.")

# ── 7. Verify ────────────────────────────────────────────────────────────
doc.add_heading("7. After Training — How to Check the Results", level=1)
para(doc, "Step 1 — run the acceptance gate (on your laptop, project folder):", bold=True, space_after=2)
code(doc, [r".venv\Scripts\python.exe scripts\check_layer3_results.py"])
para(doc, "Expected output tail — all five gates must say PASS:", bold=True, space_after=2)
code(doc, ["  [PASS] C1 eval accuracy:      target >= 0.90 | measured 0.9xx",
           "  [PASS] C2 macro-F1:           target >= 0.85 | measured 0.8xx",
           "  [PASS] C3 min per-class F1:   target >= 0.80 | measured 0.8xx",
           "  [PASS] C4 beats TF-IDF baseline: target > 0.8xx | measured 0.9xx",
           "  [PASS] C5 mean latency:       target < 150 ms | measured ~22 ms",
           "RESULT: 5/5 gates passed",
           "persisted → data\\layer3_results.json"])
para(doc, "If a gate fails, the honest fix is data, not compute — add more real/judged queries "
          "via dataset v2, or more boundary rephrasings, then retrain. Never edit the eval split "
          "to pass (the 55 queries are the contract).", italic=True)
para(doc, "Step 2 (optional) — quick latency check of the ONNX serving artifact:")
code(doc, [r".venv\Scripts\python.exe scripts\check_layer3_results.py   # C5 already covers it"])

# ── 8. Troubleshooting ───────────────────────────────────────────────────
doc.add_heading("8. Troubleshooting", level=1)
table(doc,
      ["Symptom", "Fix"],
      [
          ["GPU greyed out on Kaggle", "Phone verification missing — Settings → Phone Verification"],
          ["torch install slow on Kaggle/Colab", "First cell already auto-installs; be patient (1-2 min)"],
          ["ONNX export failed (onnx False)", "Artifacts still saved as torch; gate runs on torch fallback — re-run and check optimum installed"],
          ["Accuracy stuck ≈0.5", "Class collapse — the weighted-loss trainer prevents this; if it still occurs, use dataset v2 (real+jjudged data)"],
          ["CUDA out of memory", "Lower batch: rerun with --batch 16"],
          ["gate says [BLOCKED]", "models/layer3_distilbert missing — complete step A6 unzip first"],
          ["Model downloads hang at first run", "One-time ~260 MB distilbert download; ensure Internet ON in Kaggle"],
      ])

# ── 9. What the acceptance gates mean ────────────────────────────────────
doc.add_heading("9. The Five Acceptance Gates (the contract)", level=1)
table(doc,
      ["#", "Gate", "Target", "What it proves"],
      [
          ["C1", "Eval accuracy", "≥ 0.90", "Overall correctness on held-out rephrased queries"],
          ["C2", "Macro-F1", "≥ 0.85", "Works on semantics, not keyword memorisation"],
          ["C3", "Min per-class F1", "≥ 0.80", "No class collapses (simple/medium/complex all held)"],
          ["C4", "Beats TF-IDF baseline", "> baseline acc", "Deep model earns its place over the cheap fallback"],
          ["C5", "Mean serving latency", "< 150 ms", "Fast enough for real-time routing on CPU"],
      ])
para(doc, "Layer 3 is COMPLETE when 5/5 gates pass. That is the signal that the router (Layer 4) "
          "has a trustworthy complexity brain to drive model selection.")

doc.save(OUT)
print("saved:", OUT)