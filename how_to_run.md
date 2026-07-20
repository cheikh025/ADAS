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
Settings: 4 categories (Advanced Programming, Scientific Computing, Data Analysis, Desktop and Web Development), all difficulties, 20 per category = 80 total, seed=42.

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

## SciCode

SciCode is reconstructed independently from the official
`SciCode1/SciCode` Hugging Face dataset at pinned revision
`4510f6a6aa27c43fad7b43da2c59602a86e88480`. ADAS does not read or copy a
ReMAS JSONL. Frozen problem IDs, ordering, and full-record hashes reproduce the
same ReMAS search and held-out data and make data drift fail fast.

### Step 1 — Install the official comparison helpers

Some official tests import `scicode.compare`. Install the helper package at the
pinned source commit:

```
python -m pip install "scicode @ git+https://github.com/scicode-bench/SciCode.git@e3158ea011d4235245a547460d3688d7ccbf9900"
```

### Step 2 — Reconstruct the exact records

```
python dataset/build_scicode_data.py
```

This creates:

- `dataset/scicode_validate.jsonl`: 9 complete main problems, 3 per field,
  containing 38 scored subproblems and 120 official tests.
- `dataset/scicode_test.jsonl`: 30 complete main problems, 10 per field,
  containing 113 scored subproblems and 358 official tests.
- `dataset/scicode_manifest.json`: source metadata and a hash of every complete
  record.

The search order is Mathematics `40, 31, 18`; Physics `19, 73, 67`; Material
Science `36, 80, 35`. The held-out order is Mathematics
`24, 5, 3, 78, 54, 9, 74, 29, 1, 63`; Physics
`6, 37, 8, 70, 48, 59, 72, 58, 45, 49`; Material Science
`47, 64, 21, 77, 42, 39, 34, 27, 79, 51`. The two sets do not overlap.

### Step 3 — Provide the official numeric targets

To download and validate the approximately 1 GiB official file:

```
python dataset/download_scicode_data.py
```

To reuse an existing official copy without duplicating it, set an absolute
path. For example, on Windows:

```
set SCICODE_H5_PATH=C:\Users\cheikh\.ReMAS\data\scicode\test_data.h5
```

Only the numeric targets are reused. The ADAS JSONLs are still reconstructed
from the pinned official dataset.

### Step 4 — Run architecture search

```
python _scicode/search_ours.py \
  --search_model google/gemini-2.5-flash \
  --eval_model openai/gpt-4.1-nano \
  --base_url https://openrouter.ai/api/v1 \
  --search_thinking high --no_exec_thinking
```

Every record is a complete main-problem trajectory. Subproblems execute in
order and earlier generated functions are supplied to later steps. Official
tests are never placed in an LLM prompt. Up to three complete trajectories run
concurrently. Search fitness is the equal-weight mean of pooled subproblem pass
rate in Mathematics, Physics, and Material Science, matching ReMAS. Global
subproblem pass rate and Main Problem Resolve Rate are retained as diagnostics.

### Step 5 — Evaluate held-out data

Set `DATASET = "SciCode"` in `eval_best_workflow.py` (or
`eval_all_agents.py`) and run the chosen script:

```
python eval_best_workflow.py
python eval_all_agents.py
```

The held-out protocol uses the frozen 30 complete problems directly, with no
resampling, and runs one round. Token usage is reported per scored subproblem.

Verification:

```
python -m unittest discover -s tests -p "test_scicode.py" -v
```

The standard ADAS safety warning applies especially here: both the evolved
`forward()` architecture and generated step code execute with the current
user's permissions. A temporary subprocess prevents one failed test from
crashing the evaluator, but it is not a security sandbox.

---

## Mind2Web

The ReMAS experiment calls this benchmark Mind2Web (sometimes Web2Mind). ADAS
reconstructs the same proxy from the official `osunlp/Mind2Web` public training
split and official saved candidate ranks. It does not read or copy a ReMAS
JSONL, and it is not the private official Mind2Web test split.

The source is pinned to revision
`17ece8eb89862368edc0cc806acee6fca5163474`. The official
`scores_all_data.pkl` must be exactly 245,190,981 bytes with SHA-256
`884c97cd9ae0544485d21ea39e0d46422aee0291969a7324e56df3a84466dbd7`.

### Step 1 — Reconstruct the exact records

The first complete build downloads about 5.93 GB and can require roughly 15 GB
of temporary space. Check free disk space before running it. To reuse the
verified ReMAS rank file already present on this machine without copying it:

```
set MIND2WEB_DATA_DIR=C:\Users\cheikh\.ReMAS\data\mind2web
python dataset/build_mind2web_data.py
```

The builder verifies the rank artifact and official 1,009-task source pool,
then creates compact pruned records:

- `dataset/mind2web_validate.jsonl`: 60 complete tasks, 20 per domain.
- `dataset/mind2web_test.jsonl`: 300 disjoint tasks, 100 per domain.
- `dataset/mind2web_manifest.json`: pinned settings, ordered IDs, and hashes of
  all complete records.

The domain order is Travel, Shopping, Entertainment. Search uses one shared
seed-42 RNG sequentially across those pools. Held-out sampling removes every
search ID and uses one shared seed-99 RNG in the same order. Do not reset the
RNG separately for each domain.

An optional full 60-task ReMAS artifact can verify IDs/order but is never used
as a data source:

```
python dataset/build_mind2web_data.py --verify-remas-artifact C:\path\to\eval_data.jsonl
```

The current ReMAS workspace contains only a three-task smoke artifact, so it is
not a valid full-protocol verifier.

### Step 2 — Run architecture search

```
python _mind2web/search_ours.py \
  --search_model deepseek/deepseek-v4-flash \
  --eval_model deepseek/deepseek-v4-flash \
  --base_url https://openrouter.ai/api/v1
```

For OpenRouter, ADAS automatically matches the Mind2Web-specific ReMAS policy
for optimizer and executor calls: DeepSeek only, fallbacks disabled, and
reasoning effort `none`. This override is scoped to Mind2Web.

Each task is a complete human action trajectory, but each current action gets a
fresh ADAS agent instance. Inputs are teacher-forced with only gold preceding
actions; generated actions never change later inputs. Actions run independently
under a global limit of 32 LLM calls. The context uses official top-20 saved
ranks and official DOM neighborhood pruning. If the ranker misses every gold
element, the step scores zero without an LLM call—the gold is never injected.

Fitness is the equal-weight mean of Travel, Shopping, and Entertainment
task-macro Step Success Rates. A step succeeds only when the predicted element
is acceptable and official action F1 is exactly 1. Element accuracy, action F1,
strict task success, candidate recall, and generation errors are diagnostics.

### Step 3 — Evaluate held-out data

Set `DATASET = "Mind2Web"` in `eval_best_workflow.py` or
`eval_all_agents.py`, then run:

```
python eval_best_workflow.py
python eval_all_agents.py
```

Held-out evaluation loads the frozen 300 records directly, validates their full
contents against the manifest, performs no resampling, and runs one round.
Token usage is normalized per action.

Verification:

```
python -m unittest discover -s tests -p "test_mind2web.py" -v
```

The generated-data/manifest test is skipped until the complete official corpus
has been built; all fixture-based protocol, DOM, parsing, scoring, and causal
execution tests run without the large download.

The adapted official DOM utilities and MIT license are recorded in
`THIRD_PARTY_NOTICES.md`.

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

Mind2Web additionally uses `--max_task_workers 50` and
`--max_llm_calls 32`. Its OpenRouter provider/reasoning policy is mandatory and
overrides the generic routing flags.

## Notes

- Results accumulate incrementally in `results/<expr_name>_run_archive.json` — safe to interrupt and resume.
- Token usage is saved to `results/<expr_name>_token_usage.json`.
- `--search_model` drives the meta-LLM that writes new agent code. `--eval_model` is the LLM invoked inside `forward()` at eval time (defaults to `--search_model` when omitted).
- `--no_exec_thinking` disables thinking for the exec LLM only. Use `--search_thinking high` (or `medium`) to explicitly set the search/meta LLM thinking level — mirroring RobustMas where optimizer uses `effort: high` and executor uses `effort: none`.
- The `debug_max` flag (default: 3) limits retries on code execution errors during search — not a user-facing tunable in normal runs.
