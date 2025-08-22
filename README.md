# ZamAI Phi-3 Mini Pashto

ZamAI Phi-3 Mini Pashto is a Pashto–focused instruction-tuned variant of the base model `microsoft/Phi-3-mini-4k-instruct`.

This repository now includes:
- LoRA / QLoRA fine-tuning scripts
- DeepSpeed + Accelerate example configs
- Quantization & comparative benchmarking utilities
- Extended evaluation (BLEU / chrF / instruction metrics placeholder)
- Merge script (LoRA → full weights)
- Safety / content filtering stub
- Hugging Face Space demo scaffold (uses: `tasal9/ZamZeerak-Phi3-Pashto`)
- Pre-commit hooks, ruff config, tests & CI

> NOTE: Fine‑tuned weights are not in Git. Publish to HF Hub (`tasal9/ZamAI-Phi-3-Mini-Pashto`) after training.

---

## 1. Features

- Base: `microsoft/Phi-3-mini-4k-instruct`
- LoRA / QLoRA (4-bit NF4) with parameter-efficient fine-tuning
- DeepSpeed Zero-2 config template for larger effective batch sizes
- Evaluation:
  - Perplexity
  - Translation-style BLEU/chrF (for tasks framed as translation or normalization)
  - Instruction metrics placeholder (extend in `evaluation/metrics.py`)
- Comparison utility for different quantization loading modes
- Merge script to apply LoRA deltas to base and optionally save merged weights
- Export stubs for AWQ / GGUF (informational; may need adaptation for Phi-3 tooling maturity)
- Safety filter hook
- HF Space (Gradio) app scaffold: `hf_space/app.py` referencing model `tasal9/ZamZeerak-Phi3-Pashto`
- CI: lint + tests (pytest)
- Pre-commit: ruff, end-of-file-fixer, trailing whitespace cleaner

---

## 2. Updated Repository Layout

```
.
├── train_lora.py
├── inference.py
├── evaluate.py
├── finetune_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
├── scripts/
│   ├── run_finetune.sh
│   ├── compare_quantization.py
│   ├── merge_lora.py
│   ├── export_awq.py
│   ├── export_gguf.py
├── evaluation/
│   ├── metrics.py
│   ├── run_eval_translation.py
│   ├── run_eval_instruction.py
├── deepspeed/
│   └── ds_config_zero2.json
├── accelerate_config.yaml
├── safety/
│   └── filter.py
├── hf_space/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── .gitkeep
│   └─��� (your data files)
├── tests/
│   ├── test_clean_pashto.py
│   └── test_prompt_template.py
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── tests.yml
│   ├── pull_request_template.md
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       ├── feature_request.md
│       └── improvement_task.md
├── MODEL_CARD_TEMPLATE.md
├── DATASET_CARD_TEMPLATE.md
├── .pre-commit-config.yaml
├── README.md
├── LICENSE
└── .gitignore
```

---

## 3. New Capabilities Overview

| Area | File(s) | Description |
|------|---------|-------------|
| DeepSpeed | `deepspeed/ds_config_zero2.json` | Zero-2 config baseline |
| Accelerate | `accelerate_config.yaml` | Example multi-GPU config |
| Quantization Comparison | `scripts/compare_quantization.py` | Bench latencies & memory |
| LoRA Merge | `scripts/merge_lora.py` | Merge adapter into base |
| AWQ / GGUF Stubs | `scripts/export_awq.py`, `scripts/export_gguf.py` | Guides / placeholders |
| Instruction Eval | `evaluation/run_eval_instruction.py` | Future: custom metrics |
| Translation Eval | `evaluation/run_eval_translation.py` | BLEU/chrF via sacrebleu |
| Metrics Utils | `evaluation/metrics.py` | Shared metrics logic |
| Safety | `safety/filter.py` | Simple heuristic filter |
| Space App | `hf_space/app.py` | Gradio UI (points to Space model) |
| Testing | `tests/` + `tests.yml` | Basic unit tests |
| Pre-commit | `.pre-commit-config.yaml` | Consistent formatting & lint |
| Templates | Issue + PR + dataset/model cards | Project governance |

---

## 4. Gradio Space

The included `hf_space/app.py` assumes the deployed (or to-be-deployed) model on HF Space is `tasal9/ZamZeerak-Phi3-Pashto`. Adjust if needed.

---

## 5. LoRA Merge Example

```
python scripts/merge_lora.py \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --lora_model outputs/zamai-phi3-pashto \
  --output_dir merged_phi3_pashto
```

---

## 6. Quantization Comparison

Quick run:

```
python scripts/compare_quantization.py \
  --model_id microsoft/Phi-3-mini-4k-instruct \
  --prompt "سلام نړۍ" \
  --modes fp16 8bit 4bit
```

Outputs timing / memory summary.

---

## 7. Extended Evaluation – Translation-like

```
python evaluation/run_eval_translation.py \
  --model_id tasal9/ZamAI-Phi-3-Mini-Pashto \
  --file data/pashto_instruct_valid.jsonl \
  --field output \
  --reference_field output \
  --source_field instruction
```

---

## 8. Pre-commit

```
pip install -r requirements-dev.txt
pre-commit install
```

---

## 9. Next Ideas (Post-Integration)

- Add automatic Space deployment script
- Add streaming inference server (FastAPI)
- Integrate MT-Bench style eval harness
- Add RLHF / DPO pipeline extension

---

## 10. License & Responsibility

Same licensing as before. Provide disclaimers for non-production / sensitive uses.
