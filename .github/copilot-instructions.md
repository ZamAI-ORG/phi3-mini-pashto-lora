# Copilot Instructions for ZamAI-Phi-3-Mini-Pashto

**ALWAYS follow these instructions first and fallback to additional search and context gathering only if the information here is incomplete or found to be in error.**

## Project Overview

ZamAI Phi-3 Mini Pashto is a Pashto-focused instruction-tuned variant of microsoft/Phi-3-mini-4k-instruct. This repository contains:
- Evaluation utilities for BLEU/chrF metrics
- Safety filtering functionality  
- Gradio demo interface
- PDF book downloading tools
- Pre-commit hooks and CI/CD setup
- Testing infrastructure

**Important**: Some features mentioned in README.md are planned but not yet implemented (train_lora.py, inference.py, evaluate.py in root).

## Working Effectively

### Initial Setup
Run these commands in order for a fresh repository setup:

```bash
# Install dependencies - NEVER CANCEL: Takes ~5 minutes
pip install -r requirements-dev.txt  # Set timeout to 10+ minutes

# Install pre-commit hooks
pre-commit install

# Set Python path for imports to work correctly
export PYTHONPATH=.
```

### Testing
```bash
# Run tests - Takes ~1 second, set timeout to 30+ seconds for safety
python -m pytest tests/ -v

# Run tests with coverage - Takes ~1 second, set timeout to 30+ seconds  
python -m pytest tests/ -v --cov=evaluation --cov=safety --cov-report=term-missing
```

### Linting and Code Quality
```bash
# Check code style - Takes ~0.01 seconds, set timeout to 30+ seconds
ruff check .

# Check code formatting - Takes ~0.01 seconds, set timeout to 30+ seconds  
ruff format --check .

# Fix linting issues automatically
ruff check --fix .
ruff format .
```

**CRITICAL**: ALWAYS run linting before committing or CI will fail.

### Running Evaluation Scripts
```bash
# IMPORTANT: Always set PYTHONPATH=. for scripts to work
export PYTHONPATH=.

# Test evaluation metrics
python -c "from evaluation.metrics import compute_bleu, compute_chrf; print('Metrics work')"

# Run translation evaluation (requires model and data)
python evaluation/run_eval_translation.py \
  --model_id tasal9/ZamAI-Phi-3-Mini-Pashto \
  --file data/pashto_instruct_valid.jsonl \
  --field output \
  --reference_field output \
  --source_field instruction

# Test quantization comparison
python scripts/compare_quantization.py --help
```

### Running the Gradio Demo
```bash
# Test imports work (requires no model download)
python -c "import sys; sys.path.append('.'); import hf_space.app; print('Gradio imports work')"

# Run the demo (requires model download - will take significant time)
cd hf_space && python app.py
```

### Safety Filtering
```bash
# Test safety filter
export PYTHONPATH=.
python -c "from safety.filter import SafetyFilter; sf = SafetyFilter(); print('Safety filter works')"
```

## Manual Validation Scenarios

### After Making Code Changes
Always run these complete validation scenarios to ensure changes work correctly:

1. **Full Development Cycle Test**:
   ```bash
   # Set up environment
   export PYTHONPATH=.
   
   # Test linting (CRITICAL - CI will fail if this fails)
   ruff check . && ruff format --check .
   
   # Test core functionality
   python -c "from evaluation.metrics import compute_bleu; from safety.filter import SafetyFilter; print('Core imports work')"
   
   # Run test suite
   python -m pytest tests/ -v --cov=evaluation --cov=safety
   
   # Test evaluation functionality
   python -c "
   from evaluation.metrics import compute_bleu, compute_chrf
   preds = ['hello world']
   refs = ['hello world'] 
   bleu = compute_bleu(preds, refs)
   chrf = compute_chrf(preds, refs)
   print(f'BLEU: {bleu[\"bleu\"]:.1f}, chrF: {chrf[\"chrf\"]:.1f}')
   print('Evaluation metrics working correctly')
   "
   
   echo "All validations passed - ready to commit"
   ```

2. **Safety Filter Validation**:
   ```bash
   export PYTHONPATH=.
   python -c "
   from safety.filter import SafetyFilter
   sf = SafetyFilter()
   safe_text = 'Hello, how are you?'
   unsafe_text = 'violence and harm'
   print('Safe text:', sf.check_text(safe_text))
   print('Unsafe text:', sf.check_text(unsafe_text))
   print('Safety filter working correctly')
   "
   ```

3. **Script Functionality Check**:
   ```bash
   # Test help outputs work
   PYTHONPATH=. python evaluation/run_eval_translation.py --help
   PYTHONPATH=. python scripts/compare_quantization.py --help
   python scripts/download_pdf_books.py --help
   echo "All script help commands work"
   ```

1. **Core Functionality Test**:
   ```bash
   export PYTHONPATH=.
   python -c "from evaluation.metrics import compute_bleu; from safety.filter import SafetyFilter; print('Core imports work')"
   ```

2. **Linting Validation** (CRITICAL for CI):
   ```bash
   ruff check . && ruff format --check . && echo "Linting passed"
   ```

3. **Test Suite Validation**:
   ```bash
   python -m pytest tests/ -v --cov=evaluation --cov=safety
   ```

4. **Import Path Validation**:
   ```bash
   PYTHONPATH=. python evaluation/run_eval_translation.py --help
   PYTHONPATH=. python scripts/compare_quantization.py --help
   ```

5. **Functional Evaluation Test**:
   ```bash
   export PYTHONPATH=.
   python -c "
   from evaluation.metrics import compute_bleu, compute_chrf
   preds = ['hello world']
   refs = ['hello world'] 
   bleu = compute_bleu(preds, refs)
   chrf = compute_chrf(preds, refs)
   print(f'BLEU: {bleu[\"bleu\"]:.1f}, chrF: {chrf[\"chrf\"]:.1f}')
   print('Evaluation metrics working correctly')
   "
   ```

## Timeout and Timing Guidelines

**NEVER CANCEL these operations - Wait for completion:**

- `pip install -r requirements-dev.txt`: Takes ~5 minutes, set timeout to 10+ minutes
- Model downloads/fine-tuning: Can take 30+ minutes, set timeout to 60+ minutes
- Large data processing: Can take 15+ minutes, set timeout to 30+ minutes

**Quick operations (but use safe timeouts):**
- Tests: ~1 second, set timeout to 30+ seconds  
- Linting: ~0.01 seconds, set timeout to 30+ seconds
- Script help/imports: ~1 second, set timeout to 30+ seconds

**Measured Timing Reference** (from validation):
- `python -m pytest tests/ -v`: 0.7 seconds
- `ruff check .`: 0.01 seconds  
- `ruff format --check .`: 0.01 seconds
- `pip install -r requirements-dev.txt`: ~300 seconds (5 minutes)
- Core imports (`python -c "from evaluation.metrics import ..."`): 0.1 seconds
- Full validation cycle: ~1 second (excluding pip install)

## Critical Requirements

### Before Committing
ALWAYS run these commands before committing (CI will fail otherwise):
```bash
ruff check . && ruff format --check .
python -m pytest tests/ -v
```

### Python Path Setup
Most scripts require `PYTHONPATH=.` to work correctly:
```bash
export PYTHONPATH=.
# OR prefix commands with: PYTHONPATH=. python script.py
```

### Working vs Planned Features
**Working features:**
- Evaluation metrics (evaluation/metrics.py)
- Safety filtering (safety/filter.py)  
- Gradio demo (hf_space/app.py)
- PDF downloaders (scripts/download_*.py)
- Testing infrastructure (tests/)
- Linting with ruff

**Planned but not implemented:**
- train_lora.py (mentioned in README but doesn't exist)
- inference.py (mentioned in README but doesn't exist)  
- evaluate.py (mentioned in README but doesn't exist)

## Repository Structure

```
.
├── README.md
├── requirements.txt          # Main dependencies
├── requirements-dev.txt      # Development dependencies  
├── pyproject.toml           # Project configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .github/
│   └── workflows/
│       └── tests.yml        # CI pipeline
├── evaluation/
│   ├── metrics.py           # BLEU/chrF utilities
│   ├── run_eval_translation.py
│   └── run_eval_instruction.py
├── safety/
│   └── filter.py            # Content safety filtering
├── hf_space/
│   ├── app.py              # Gradio demo interface
│   └── requirements.txt
├── scripts/
│   ├── compare_quantization.py
│   ├── download_pdf_books.py
│   ├── download_afghan_books.sh
│   └── tune_headers.py
├── tests/
│   ├── test_clean_pashto.py
│   └── test_prompt_template.py
├── data/                    # Empty with .gitkeep
└── deepspeed/
    └── ds_config_zero2.json
```

## Common Troubleshooting

1. **Import Errors**: Always use `export PYTHONPATH=.` or `PYTHONPATH=. python script.py`
2. **CI Failures**: Run linting first: `ruff check . && ruff format --check .`
3. **Test Failures**: Ensure dependencies installed: `pip install -r requirements-dev.txt`
4. **Pre-commit Issues**: Use ruff directly instead of pre-commit hooks if network issues
5. **Model Errors**: Most evaluation scripts require actual models - use test imports first

## Critical Warnings and Known Issues

### DO NOT Do These Things:
- **NEVER** try to run scripts without `export PYTHONPATH=.` - imports will fail
- **NEVER** commit without running linting first - CI will fail
- **NEVER** assume train_lora.py, inference.py, evaluate.py exist in root - they don't yet
- **NEVER** cancel pip install operations - they take 5+ minutes and are critical
- **NEVER** run pre-commit hooks without stable internet - use ruff directly instead

### Network and Timing Issues:
- Pre-commit hook installation may timeout due to network issues - this is expected
- Use `ruff check .` and `ruff format .` directly instead of pre-commit hooks
- PDF downloader scripts require interactive confirmation - not suitable for automation
- Model-based operations can take 30+ minutes - always set generous timeouts

### Missing Features (Do NOT try to use):
The following are mentioned in README.md but not implemented:
- `train_lora.py` (root level) - doesn't exist
- `inference.py` (root level) - doesn't exist  
- `evaluate.py` (root level) - doesn't exist
- Use `evaluation/run_eval_*.py` scripts instead for evaluation tasks

## Development Workflow

1. Make changes to code
2. ALWAYS set `export PYTHONPATH=.`
3. Test imports: `python -c "from module import function"`
4. Run linting: `ruff check . && ruff format --check .`
5. Run tests: `python -m pytest tests/ -v`
6. Validate specific functionality with test scenarios
7. Commit changes

## File Locations Reference

- **Core evaluation**: `evaluation/metrics.py`
- **Safety filtering**: `safety/filter.py`
- **Main demo**: `hf_space/app.py`
- **Test files**: `tests/test_*.py`
- **CI configuration**: `.github/workflows/tests.yml`
- **Linting config**: `pyproject.toml` (ruff section)
- **Dependencies**: `requirements.txt`, `requirements-dev.txt`