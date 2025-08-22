# ZamAI Phi-3 Mini Pashto

ZamAI Phi-3 Mini Pashto is a Pashto–focused instruction-tuned variant of the base model `microsoft/Phi-3-mini-4k-instruct`.  
This repository contains the fine-tuning scripts, configuration, inference utilities, evaluation script, Docker environment, and CI workflow to (a) reproduce continued instruction tuning on Pashto data and (b) serve / evaluate the resulting model.

> NOTE: The actual fine‑tuned weights are not stored in this Git repository (recommended best practice). After training, you can push them to the Hugging Face Hub under (example) `tasal9/ZamAI-Phi-3-Mini-Pashto` and reference them here.

---

## 1. Features

- Base: `microsoft/Phi-3-mini-4k-instruct`
- Parameter-efficient fine-tuning via LoRA (PEFT)
- Optional 4-bit QLoRA (bitsandbytes) for low VRAM environments
- Gradient accumulation & mixed precision (bfloat16)
- Simple YAML-driven configuration
- Pashto text normalization helper
- Inference script with:
  - Standard generation
  - Quantized loading (8-bit / 4-bit)
- Evaluation script (perplexity)
- Dockerfile for reproducible environment
- GitHub Actions CI (ruff lint)

---

## 2. Repository Layout

```
.
├── train_lora.py
├── inference.py
├── evaluate.py
├── finetune_config.yaml
├── requirements.txt
├── Dockerfile
├── scripts/
│   └── run_finetune.sh
├── data/
│   └── .gitkeep
├── .github/workflows/ci.yml
├── README.md
├── LICENSE
└── .gitignore
```

---

## 3. Installation

```bash
git clone https://github.com/tasal9/ZamAI-Phi-3-Mini-Pashto.git
cd ZamAI-Phi-3-Mini-Pashto
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

If using CUDA 12+ and bitsandbytes wheels are missing, consult bitsandbytes docs.

---

## 4. Preparing Pashto Dataset

You need a JSONL dataset with instruction / input / output style, e.g:

```
{"instruction": "د پښتو 'سلام' انګلیسي ته وژباړه", "input": "", "output": "Hello"}
{"instruction": "لاندې جمله ساده کړه", "input": "زه غواړم چې نن خپل کارونه په بریالیتوب بشپړ کړم.", "output": "زه غواړم نن خپل کارونه بشپړ کړم."}
```

Place dataset file at `data/pashto_instruct_train.jsonl` (or as configured in `finetune_config.yaml`).  
Validation (optional) at `data/pashto_instruct_valid.jsonl`.

---

## 5. Configuration

Edit `finetune_config.yaml`:

- `base_model_name`: HF base model (already set to Phi-3)
- `lora_r`, `lora_alpha`, `lora_dropout`
- `train_file`, `eval_file`
- `use_4bit`: enable QLoRA memory optimizations

---

## 6. Running Fine-Tuning

Quick start:

```bash
bash scripts/run_finetune.sh
```

Or manually:

```bash
python train_lora.py \
  --config finetune_config.yaml \
  --output_dir outputs/zamai-phi3-pashto \
  --push_to_hub \
  --hub_model_id tasal9/ZamAI-Phi-3-Mini-Pashto
```

(You must `huggingface-cli login` first.)

---

## 7. Inference

After weights are on HF Hub:

```bash
python inference.py \
  --model_id tasal9/ZamAI-Phi-3-Mini-Pashto \
  --prompt "په ساده پښتو کې تشریح کړه: عصبي شبکه څه ده؟"
```

Sample output:

```
[ZamAI]: عصبي شبکه د کمپيوټر يو ماډل دی چې هڅه کوي د انسان د دماغ د زده کړې طريقه تقليد کړي ...
```

---

## 8. Evaluation (Perplexity)

Compute perplexity on a held-out file:

```bash
python evaluate.py \
  --model_id tasal9/ZamAI-Phi-3-Mini-Pashto \
  --eval_file data/pashto_instruct_valid.jsonl \
  --config finetune_config.yaml
```

Outputs average loss & perplexity.

---

## 9. Docker Usage

Build image:

```bash
docker build -t zamai-phi3 .
```

Run training (mount data & output):

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  zamai-phi3 \
  python train_lora.py --config finetune_config.yaml --output_dir outputs/run1
```

Run inference:

```bash
docker run --gpus all -it zamai-phi3 \
  python inference.py --model_id tasal9/ZamAI-Phi-3-Mini-Pashto --prompt "سلام"
```

---

## 10. CI (GitHub Actions)

A simple workflow (`.github/workflows/ci.yml`) runs `ruff` on pushes & PRs to `main` for basic linting.

---

## 11. Example Training Hyperparameters (Baseline)

| Setting | Value |
|--------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Max seq length | 2048 |
| Per device batch | 1–2 (accumulation to reach effective 16) |
| LR | 2e-4 |
| Warmup ratio | 0.03 |
| Epochs | 3–5 |
| Weight decay | 0.0 |
| Gradient clip | 1.0 |

Adjust per GPU memory.

---

## 12. Pashto Tokenization Notes

Phi-3 uses a fast tokenizer; Pashto script may include diacritics. Optional normalization in the script can:
- Remove zero-width characters
- Normalize Arabic Yeh / Kaf variants

Extend `clean_pashto_text` helper in `train_lora.py` for more rules if needed.

---

## 13. Safety & Responsible Use

This model may hallucinate or produce harmful content. It should NOT be used for:
- Medical / legal advice
- High-stakes decision making

Add content filters or moderation where appropriate.

---

## 14. Roadmap

- [ ] Add DeepSpeed config
- [ ] Add richer evaluation metrics (BLEU, chrF, custom instruction metrics)
- [ ] Add dataset card
- [ ] Release quantized GGUF / AWQ variants
- [ ] Add unit tests for data pipeline

---

## 15. Citation

```
@misc{ZamAI2025,
  title  = {ZamAI Phi-3 Mini Pashto},
  author = {tasal9},
  year   = {2025},
  note   = {Fine-tuned from microsoft/Phi-3-mini-4k-instruct},
  url    = {https://huggingface.co/tasal9/ZamAI-Phi-3-Mini-Pashto}
}
```

---

## 16. License

Code: MIT (see LICENSE)  
Model Weights: follow base model license + your added terms (please ensure compatibility).

---

## 17. Quick Inference Snippet (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tasal9/ZamAI-Phi-3-Mini-Pashto"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "په پښتو کې د 'Artificial Intelligence' تشريح وکړه."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

Feel free to request enhancements or additions.