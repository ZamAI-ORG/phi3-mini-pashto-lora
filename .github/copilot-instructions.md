# ZamAI Phi-3 Mini Pashto Repository

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

This is a Pashto-focused instruction-tuned variant repository for Microsoft's Phi-3-mini-4k-instruct model. The repository is currently in a **template/planning state** - it contains configuration files and documentation but most implementation files are not yet present.

## Working Effectively

### Environment Setup (VALIDATED - WORKS)
```bash
# Install Python dependencies - takes 2-3 minutes. NEVER CANCEL - set timeout to 10+ minutes
pip install -r requirements.txt
```

**TIMING:** Package installation takes 2-3 minutes for ~50 packages including PyTorch, Transformers, PEFT, etc.

### Test Environment Validation (ALWAYS RUN FIRST)
```bash
# Test all major libraries load correctly - takes ~4 seconds
python -c "
import time
start = time.time()
import torch, transformers, accelerate, peft, datasets
print(f'All libraries imported in {time.time()-start:.2f} seconds')
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
"
```

### Configuration Validation (VALIDATED - WORKS)
```bash
# Test YAML config loading
python -c "import yaml; config = yaml.safe_load(open('finetune_config.yaml')); print('Config loaded:', config['base_model_name'])"
```

## Current Repository State

### What EXISTS and WORKS:
- `requirements.txt` - **FULLY FUNCTIONAL** (PyTorch 2.8.0, Transformers 4.55.4, PEFT 0.17.1)
- `finetune_config.yaml` - **VALID YAML** configuration for LoRA fine-tuning
- `scripts/run_finetune.sh` - **SYNTACTICALLY CORRECT** but references missing `train_lora.py`
- `Dockerfile` - **VALID** but requires large base image (>3GB download)
- `data/` directory - **EXISTS** but only contains `.gitkeep`

### What is MISSING (referenced in README but not implemented):
- **ALL Python implementation files:** `train_lora.py`, `inference.py`, `evaluate.py`
- **ALL subdirectories:** `evaluation/`, `deepspeed/`, `safety/`, `hf_space/`, `tests/`
- **Development configs:** `requirements-dev.txt`, `pyproject.toml`, `.pre-commit-config.yaml`
- **GitHub workflows:** `.github/workflows/`

## Docker (PARTIALLY VALIDATED)

### Docker Build - **DO NOT ATTEMPT**
```bash
# Docker build FAILS due to disk space requirements (needs >8GB free space)
# Base image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel is >3GB
# Expected error: "no space left on device"
```

**NEVER attempt Docker builds in this environment - they require >8GB free disk space.**

## Model and Data Access

### Model Loading Limitations
```bash
# Model download FAILS in this environment (no internet access to HuggingFace Hub)
# Expected error: "We couldn't connect to 'https://huggingface.co'"
```

**CRITICAL:** This environment cannot download models from HuggingFace Hub. Any scripts requiring model downloads will fail.

## Testing and Validation

### What to Test After Changes
1. **Always run environment validation first** (library imports test above)
2. **Test YAML configuration loading** if modifying config files
3. **Validate any new Python syntax** with `python -m py_compile <file>`
4. **Never attempt to run missing scripts** like `train_lora.py`, `inference.py`, etc.

### Pre-commit and Linting
```bash
# Pre-commit is NOT configured (missing .pre-commit-config.yaml)
# Manual linting options:
python -m flake8 <file>      # If implementing Python files
python -m black <file>       # If implementing Python files  
```

## Scripts and Commands

### scripts/run_finetune.sh
```bash
# This script FAILS - references missing train_lora.py
# Expected error: "can't open file 'train_lora.py': No such file or directory"
```

**Do not run this script** - it requires implementing `train_lora.py` first.

## Development Workflow

### When Adding New Features:
1. **Start with configuration files** (modify `finetune_config.yaml` if needed)
2. **Implement Python files** referenced in existing scripts
3. **Test each component in isolation** before integration
4. **Always validate imports work** after adding new dependencies

### Key Implementation Priority Order:
1. `train_lora.py` - Core training script referenced by `run_finetune.sh`
2. `inference.py` - For model inference testing
3. `evaluate.py` - For model evaluation
4. Missing directory structures (`evaluation/`, `tests/`, etc.)

## Repository Structure Reference

### Working Files (verified present):
```
.
├── requirements.txt          # WORKS - installs successfully
├── finetune_config.yaml     # WORKS - valid YAML configuration
├── Dockerfile               # VALID - but requires >8GB disk space
├── scripts/
│   └── run_finetune.sh      # VALID syntax - but requires train_lora.py
├── data/
│   └── .gitkeep            # EXISTS - placeholder only
├── README.md               # EXISTS - describes planned features
└── LICENSE                 # EXISTS - MIT license
```

### Missing Implementation (from README but not present):
```
├── train_lora.py           # MISSING - main training script
├── inference.py            # MISSING - inference script  
├── evaluate.py             # MISSING - evaluation script
├── requirements-dev.txt    # MISSING - development dependencies
├── pyproject.toml          # MISSING - project configuration
├── evaluation/             # MISSING - evaluation utilities
├── deepspeed/              # MISSING - DeepSpeed configurations
├── accelerate_config.yaml  # MISSING - Accelerate configuration
├── safety/                 # MISSING - content filtering
├── hf_space/              # MISSING - Gradio Space app
├── tests/                 # MISSING - unit tests
├── .pre-commit-config.yaml # MISSING - pre-commit hooks
└── .github/               # MISSING - GitHub workflows
```

## Timing and Performance Notes

- **Environment setup:** 2-3 minutes for pip install
- **Library imports:** ~4 seconds for all major ML libraries  
- **YAML config loading:** <1 second
- **Docker build:** FAILS - requires >8GB free space, would take 10+ minutes if successful

## Critical Warnings

- **NEVER CANCEL pip install** - Set timeout to 10+ minutes minimum
- **DO NOT attempt Docker builds** - Will fail due to disk space constraints
- **DO NOT run scripts referencing missing .py files** - They will fail immediately
- **Model downloads WILL FAIL** - No internet access to HuggingFace Hub in this environment

## When Instructions Are Incomplete

If these instructions don't cover your specific task:
1. **First check** if required implementation files exist using `ls -la <filename>`
2. **Test basic Python syntax** with `python -c "import <module>"`
3. **Search the repository** for related configuration or example files
4. **Only then** explore with additional bash commands or searches

This repository is a **template/planning state** - expect to implement most functionality from scratch based on the configuration files and README specifications.