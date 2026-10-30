# LLM-post-training-bugfixer

Comparing LLM post-training techniques (SFT, DPO, RL, distillation, and optional continued pretraining) for teaching a small open model to fix Python bugs, evaluated with a reproducible, test-verified harness.

## Overview

The project builds a bug-fixing agent and measures how each post-training technique changes its performance. The agent takes a bug report (failing pytest output), retrieves the relevant source files, proposes a patch, runs the tests, and retries with feedback if the fix fails.

The same agent, prompts, retrieval, and test budget are used for every model variant, so differences in results can be attributed to the training and not to a different workflow.

**Research question:** How do SFT, preference optimization, RL with execution-based rewards, and distillation affect bug-fixing success, latency, and cost for an open-weight model?

### Model variants

| Tag | Variant | Technique |
|---|---|---|
| A | Baseline | Original instruct model, no training |
| B | CPT (optional) | Continued pretraining on a Python corpus |
| C | SFT | LoRA fine-tuning on gold fixes |
| D | SFT + DPO | Preference optimization on execution-labeled pairs |
| E | SFT + DPO + GRPO | RL with test-based rewards |
| F | Distilled student | Smaller model trained on verified teacher outputs |

### How it works

- **Benchmark:** `tasks.py` injects single-line bugs (operator swaps, flipped comparisons, and so on) into real repositories that have pytest suites. A task is kept only if the clean suite passes and the mutated one fails. Every task is verifiable by running tests, so one pipeline gives you an eval set, SFT targets, DPO pairs, and RL rewards.
- **Edit format:** the model outputs SEARCH/REPLACE blocks, and our code applies them and generates the diff. This is far more reliable than asking a small model for raw unified diffs.
- **Retrieval (LlamaIndex):** function-level chunks are embedded and used to locate the relevant files, and the model then sees whole files. The index is built once from the clean source and cached on disk.
- **Model interface (LangChain):** `ChatOpenAI` pointed at any OpenAI-compatible endpoint (Hugging Face router, vLLM, Ollama). Baseline and fine-tuned models share identical agent code, and only the URL and model name change.
- **Safeguards:** test files cannot be edited (enforced in code), a patch only counts if the full test count passes, and train/val/test are split by file so no file appears in both train and test.

## Project structure

```
bugfix-post-training/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── repos/              # target repos (gitignored)
├── data/               # generated tasks, prompts, SFT/DPO data (gitignored)
├── runs/               # trained checkpoints (gitignored)
├── results/            # evaluation outputs (gitignored)
└── src/bugfix/
    ├── config.py       # environment variables
    ├── llm.py          # LangChain model interface
    ├── edits.py        # SEARCH/REPLACE parsing and application
    ├── sandbox.py      # temp workspaces and pytest runner
    ├── code_index.py   # LlamaIndex retrieval
    ├── agent.py        # the bug-fixing repair loop
    ├── tasks.py        # bug injection and benchmark generation
    ├── scoring.py      # single-shot scoring and RL reward
    ├── evaluate.py     # benchmark runner and comparison table
    ├── collect.py      # prompts, gold SFT, sampling, DPO pairs
    ├── train_utils.py  # LoRA config, model loading, merging
    ├── train_cpt.py    # continued pretraining
    ├── train_sft.py    # supervised fine-tuning
    ├── train_dpo.py    # direct preference optimization
    ├── train_grpo.py   # GRPO with test-based rewards
    └── app.py          # Streamlit demo
```

## Requirements

- Python 3.10+
- Git
- A Hugging Face account and access token, with the Llama license accepted on the model page
- **Laptop (agent, evaluation, demo):** no GPU needed
- **GPU machine (training):** an NVIDIA GPU with enough memory for QLoRA on an 8B model (roughly 24 GB is comfortable). A 3B model is a cheaper way to debug the training scripts. GRPO is the most memory-hungry stage.

## Setup

```bash
git clone <your-repo-url> bugfix-post-training
cd bugfix-post-training

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
```

Edit `.env`:

```
LLM_BASE_URL=https://router.huggingface.co/v1
LLM_API_KEY=
HF_TOKEN=hf_your_token_here
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
EMBED_MODEL=BAAI/bge-small-en-v1.5
```

Notes:

- `LLM_API_KEY` falls back to `HF_TOKEN` when empty.
- Hosted inference availability changes. If the baseline run fails with a model-not-found or provider error, check the current Hugging Face Inference Providers docs. You may need a provider suffix on the model name, or you can serve the model yourself with vLLM or Ollama and point `LLM_BASE_URL` at it.
- The training libraries (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`) are only needed on the GPU machine. On a laptop you can install the lighter dependencies and skip the rest.

### Target repositories

Clone pure-Python repos with fast, clean pytest suites into `repos/`:

```bash
git clone --depth 1 https://github.com/mahmoud/boltons repos/boltons
git clone --depth 1 https://github.com/pytoolz/toolz repos/toolz
```

Run `python -m pytest -q` inside each one first. If a suite fails or takes over a minute, swap in a different repo. Task generation skips repos whose suites don't pass cleanly.

If pytest tries to collect the cloned repos when you run tests from the project root, add this to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

## Running the pipeline

All commands run from the project root. Start with steps 1 and 2 on your laptop, and read a few trajectories in `results/A_baseline.jsonl` before spending any GPU money.

### 1. Generate the benchmark and prompts

```bash
python -m bugfix.tasks --repos repos/boltons repos/toolz --out data/tasks.jsonl --max-per-file 4
python -m bugfix.collect prompts --out data/prompts.jsonl
```

If you change `SYSTEM_PROMPT` or the context format in `agent.py`, rerun the `prompts` command and everything downstream.

### 2. Baseline (variant A)

```bash
python -m bugfix.evaluate --tag A_baseline --split test --limit 100 --attempts 3
```

Check the result before going further:

- Near 100%: the bugs are too easy. Add more repos or harder mutations.
- Near 0%: check `retrieval_recall` in the summary before blaming the model.

### 3. SFT (variant C)

On a GPU machine:

```bash
python -m bugfix.collect gold-sft --out data/sft_gold.jsonl
python -m bugfix.train_sft --base meta-llama/Llama-3.1-8B-Instruct \
  --data data/sft_gold.jsonl --out runs/sft --four-bit
```

Serve the merged model and evaluate it:

```bash
vllm serve runs/sft/merged --served-model-name sft --max-model-len 8192 --port 8000

python -m bugfix.evaluate --tag C_sft --model-name sft \
  --base-url http://localhost:8000/v1 --api-key EMPTY
```

### 4. DPO (variant D)

Sample candidate fixes from the SFT model (on-policy), build preference pairs from test results, then train:

```bash
python -m bugfix.collect sample --split train --n 8 --model-name sft \
  --base-url http://localhost:8000/v1 --api-key EMPTY --out data/samples_sft.jsonl
python -m bugfix.collect dpo --samples data/samples_sft.jsonl --out data/dpo.jsonl
python -m bugfix.train_dpo --base runs/sft/merged --data data/dpo.jsonl \
  --out runs/dpo --four-bit
```

Serve `runs/dpo/merged` as `dpo` and evaluate with `--tag D_dpo`.

### 5. GRPO (variant E)

```bash
python -m bugfix.train_grpo --base runs/dpo/merged --samples data/samples_sft.jsonl \
  --out runs/grpo --four-bit
```

Serve `runs/grpo/merged` as `grpo` and evaluate with `--tag E_grpo`. Passing `--samples` restricts training to tasks with mixed outcomes, since all-pass and all-fail groups give no learning signal.

### 6. Distillation (variant F)

Use your best checkpoint or a larger hosted model as the teacher:

```bash
python -m bugfix.collect sample --split train --n 8 \
  --model-name TEACHER --base-url TEACHER_URL --out data/samples_teacher.jsonl
python -m bugfix.collect sft-from-samples --samples data/samples_teacher.jsonl \
  --out data/sft_distill.jsonl
python -m bugfix.train_sft --base meta-llama/Llama-3.2-3B-Instruct \
  --data data/sft_distill.jsonl --out runs/student
```

Serve `runs/student/merged` and evaluate with `--tag F_student`. This is response-level distillation: verified teacher outputs, then SFT on the student.

### 7. Continued pretraining (variant B, optional)

```bash
python -m bugfix.train_cpt --base meta-llama/Llama-3.1-8B-Instruct \
  --corpus-dir path/to/python_corpus --out runs/cpt
```

The corpus must not include your evaluation repos. This stage is expensive and can degrade an instruct model, so recheck instruction following afterward.

### 8. Compare results and run the demo

```bash
python -m bugfix.evaluate --compare A_baseline C_sft D_dpo E_grpo F_student
streamlit run src/bugfix/app.py
```

The demo has a tab for running the agent on a bug report (it works on a temp copy, so your repo is never modified) and a tab showing the comparison table from `results/`.

## Metrics

| Metric | Meaning |
|---|---|
| `success@1` | Fixed on the first attempt |
| `success@k` |