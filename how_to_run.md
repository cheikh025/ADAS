# How to Run ADAS

All commands run from: `C:\Users\cheikh\Desktop\baseline\ADAS`

---

## Prerequisites (do once, for all benchmarks)

**Create and activate the conda environment**
```
conda create -n adas python=3.11
conda activate adas
pip install -r requirements.txt
pip install json_repair python-dotenv
```

**Set API keys in `.env` at the repo root**
```
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
```
The key is auto-selected based on `--base_url`: `openrouter.ai` → `OPENROUTER_API_KEY`, `groq.com` → `GROQ_API_KEY`, `openai.com` → `OPENAI_API_KEY`.

---

## MATH

### Step 1 — Build the dataset
```
python dataset/build_math_4subjects.py
```
Output: `dataset/math_4subjects.jsonl`  
Settings: 4 subjects (Prealgebra, Number Theory, Precalculus, Counting & Probability), Level 5 only, 30 per subject = 120 total, seed=42.

### Step 2 — Run the search
```
python _math/search_ours.py --search_model google/gemini-2.5-flash --base_url https://openrouter.ai/api/v1
```
To match RobustMas settings (search LLM thinks at high, exec LLM no thinking):
```
python _math/search_ours.py --search_model google/gemini-2.5-flash --eval_model openai/gpt-4.1-nano --base_url https://openrouter.ai/api/v1 --search_thinking high --no_exec_thinking
```
Results saved to: `results/math_ours_results_run_archive.json`

### Step 3 — Evaluate best discovered agent
In `eval_best_workflow.py` set `DATASET = "MATH"` (line 46), then:
```
python eval_best_workflow.py
```

### Step 4 — Compare all baselines + best agent
In `eval_all_agents.py` set `DATASET = "MATH"` (line 39), then:
```
python eval_all_agents.py
```
Evaluates COT, COT-SC, Reflexion, and best ADAS agent on held-out data — zero overlap with training.

---

## MMLU-Pro

### Step 1 — Build the dataset
```
python dataset/build_mmlu_pro_4categories.py
```
Output: `dataset/mmlu_pro_4categories.csv`  
Settings: 4 categories (law, history, philosophy, engineering), 20 per category = 80 total.

### Step 2 — Run the search
```
python _mmlu_pro/search_ours.py --search_model google/gemini-2.5-flash --base_url https://openrouter.ai/api/v1
```
To match RobustMas settings (search LLM thinks at medium, exec LLM no thinking):
```
python _mmlu_pro/search_ours.py --search_model google/gemini-2.5-flash --eval_model openai/gpt-4.1-nano --base_url https://openrouter.ai/api/v1 --search_thinking medium --no_exec_thinking
```
Results saved to: `results/mmlu_pro_ours_results_run_archive.json`

### Step 3 — Evaluate best discovered agent
In `eval_best_workflow.py` set `DATASET = "MMLUPro"` (line 46), then:
```
python eval_best_workflow.py
```

### Step 4 — Compare all baselines + best agent
In `eval_all_agents.py` set `DATASET = "MMLU_PRO"` (line 39), then:
```
python eval_all_agents.py
```

---

## FullStack

### Step 1 — Start SandboxFusion
SandboxFusion must be running before any evaluation call is made.
```
docker run -p 8080:8080 bytedance/sandbox-fusion:latest
```
If running at a different address, set the environment variable:
```
set SANDBOX_FUSION_ENDPOINT=http://<host>:<port>
```

### Step 2 — Build the dataset
```
python dataset/build_fullstack_subset.py
```
Output: `dataset/fullstack_subset.jsonl`  
Settings: 4 categories (Advanced Programming, Scientific Computing, Data Analysis, Desktop and Web Development), hard difficulty, 3 per category = 12 total, seed=42.

### Step 3 — Run the search
```
python _fullstack/search_ours.py --search_model google/gemini-2.5-flash --base_url https://openrouter.ai/api/v1
```
To match RobustMas settings (search LLM thinks at high, exec LLM no thinking):
```
python _fullstack/search_ours.py --search_model google/gemini-2.5-flash --eval_model openai/gpt-4.1-nano --base_url https://openrouter.ai/api/v1 --search_thinking high --no_exec_thinking
```
Keep `--max_workers` low (default: 3) — each eval call hits SandboxFusion.  
Results saved to: `results/fullstack_ours_results_run_archive.json`

### Step 4 — Evaluate best discovered agent
In `eval_best_workflow.py` set `DATASET = "FullStack"` (line 46), then:
```
python eval_best_workflow.py
```
SandboxFusion must be running.

### Step 5 — Compare all baselines + best agent
In `eval_all_agents.py` set `DATASET = "FullStack"` (line 39), then:
```
python eval_all_agents.py
```

---

## Additional search_ours.py flags

| Flag | Default | Description |
|---|---|---|
| `--n_generation` | 20 (30 for FullStack) | Number of search iterations |
| `--max_workers` | 50 (3 for FullStack) | Parallel threads for evaluation |
| `--total_token_budget` | None (unlimited) | Stop when total tokens (search + exec) exceed this value |
| `--search_temperature` | 0.8 | Temperature for the meta-LLM (design generation + reflexion) |
| `--eval_temperature` | 1.0 | Temperature for agent LLM calls during evaluation |
| `--provider_order` | None | Comma-separated OpenRouter provider order for the **exec** LLM, e.g. `"openai,Together"` |
| `--no_exec_thinking` | False | Disable thinking for the exec LLM (`reasoning.effort=none`) |
| `--search_provider_order` | None | Comma-separated OpenRouter provider order for the **search/meta** LLM, e.g. `"novita,Together"` |
| `--search_thinking` | None | reasoning.effort for the search/meta LLM: `none` / `medium` / `high`. Default: no override |
| `--expr_name` | `*_ours_results` | Prefix for the saved archive JSON file |

## Notes

- Results accumulate incrementally in `results/<expr_name>_run_archive.json` — safe to interrupt and resume.
- Token usage is saved to `results/<expr_name>_token_usage.json`.
- `--search_model` drives the meta-LLM that writes new agent code. `--eval_model` is the LLM invoked inside `forward()` at eval time (defaults to `--search_model` when omitted).
- `--no_exec_thinking` disables thinking for the exec LLM only. Use `--search_thinking high` (or `medium`) to explicitly set the search/meta LLM thinking level — mirroring RobustMas where optimizer uses `effort: high` and executor uses `effort: none`.
- The `debug_max` flag (default: 3) limits retries on code execution errors during search — not a user-facing tunable in normal runs.
