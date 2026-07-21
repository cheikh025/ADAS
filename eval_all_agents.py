"""
ADAS Comprehensive Held-out Evaluation

Evaluates the following agents on the SAME fresh held-out queries:
  1. Chain-of-Thought (CoT)
  2. Self-Consistency with Chain-of-Thought (CoT-SC)
  3. Self-Refine (Reflexion)
  4. Best searched agent (highest median fitness from archive)

Supports DATASET = "MATH", "MMLU_PRO", "FullStack", "SciCode", or "Mind2Web"

Run:
    python eval_all_agents.py
"""

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
_ADAS_DIR  = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← change here
# ─────────────────────────────────────────────────────────────────────────────
DATASET          = "MATH"   # "MATH", "MMLU_PRO", "FullStack", "SciCode", or "Mind2Web"
NUM_EVAL_QUERIES = 100      # held-out queries per subject / category
MAX_WORKERS      = 50       # parallel threads
SEED             = 99       # sampling seed (training used 42)
K_ROUND_EVAL     =  3       # evaluation rounds per agent (scores averaged, std reported)
SCICODE_K_ROUND_EVAL = 1
MIND2WEB_K_ROUND_EVAL = 1
# ─────────────────────────────────────────────────────────────────────────────

# 3 subjects — matches MAS_pro and AFlow for a fair comparison
# Note: ADAS was searched on 4 subjects (add "Prealgebra" here if needed)
MATH_SUBJECTS = [
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
MMLU_PRO_CATEGORIES = ["physics", "economics", "philosophy", "engineering"]
FULLSTACK_SUBJECTS = [
    "Advanced Programming",
    "Scientific Computing",
    "Data Analysis",
    "Desktop and Web Development",
]
SCICODE_FIELDS = ["mathematics", "physics", "material_science"]
MIND2WEB_DOMAINS = ["travel", "shopping", "entertainment"]
MATH_LEVEL = "Level 5"

MATH_TRAIN_JSONL        = _ADAS_DIR / "dataset/math_4subjects.jsonl"
MMLU_PRO_TRAIN_CSV      = _ADAS_DIR / "dataset/mmlu_pro_4categories.csv"
FULLSTACK_TRAIN_JSONL   = _ADAS_DIR / "dataset/fullstack_subset.jsonl"
SCICODE_TEST_JSONL      = _ADAS_DIR / "dataset/scicode_test.jsonl"
MIND2WEB_TEST_JSONL     = _ADAS_DIR / "dataset/mind2web_test.jsonl"
MATH_RAW_TEST_DIR       = _ADAS_DIR / "data/math_hf_cache/MATH/test"
MMLU_PRO_HF_CACHE       = _ADAS_DIR / "data/mmlu_pro_hf_cache"
FULLSTACK_HF_CACHE      = _ADAS_DIR / "data/fullstack_hf_cache"

SANDBOX_ENDPOINT        = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080")
SANDBOX_COMPILE_TIMEOUT = 120
SANDBOX_RUN_TIMEOUT     = 120

MATH_ARCHIVE      = _ADAS_DIR / "results/math_ours_results_run_archive.json"
MMLU_PRO_ARCHIVE  = _ADAS_DIR / "results/mmlu_pro_ours_results_run_archive.json"
FULLSTACK_ARCHIVE = _ADAS_DIR / "results/fullstack_ours_results_run_archive.json"
SCICODE_ARCHIVE   = _ADAS_DIR / "results/scicode_ours_results_run_archive.json"
MIND2WEB_ARCHIVE  = _ADAS_DIR / "results/mind2web_ours_results_run_archive.json"

# Baselines pulled from the initial archive entries
BASELINE_NAMES = [
    "Chain-of-Thought",
  #  "Self-Consistency with Chain-of-Thought",
   # "Self-Refine (Reflexion)",
   # "LLM Debate",
]
SCICODE_BASELINE_NAMES = [
    "Scientific Chain-of-Thought",
    "Scientific Self-Consistency",
    "Scientific Reflexion",
    "Triangulated Scientific Experts",
]
MIND2WEB_BASELINE_NAMES = [
    "Web Action Chain-of-Thought",
    "Web Action Self-Consistency",
    "Web Action Reflexion",
    "Web Action Debate",
]

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
_MMLU_PRO_L2I = {c: i for i, c in enumerate("ABCDEFGHIJ")}


# ─────────────────────────────────────────────────────────────────────────────
# Archive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_median(fitness_str: str) -> float:
    m = re.search(r"Median:\s*([\d.]+)%", fitness_str)
    return float(m.group(1)) if m else 0.0


def find_best_agent(archive_path: Path) -> dict:
    with open(archive_path) as f:
        archive = json.load(f)
    candidates = [
        a for a in archive
        if "fitness" in a and isinstance(a.get("generation"), int)
    ]
    if not candidates:
        raise ValueError(f"No evolved agents found in {archive_path}.")

    def fitness_value(agent: dict) -> float:
        score = agent.get("fitness_score")
        if isinstance(score, (int, float)):
            return 100.0 * float(score)
        return _parse_median(str(agent.get("fitness", "")))

    best = max(candidates, key=fitness_value)
    print(f"  Best agent: gen={best['generation']}, name={best['name']!r}")
    print(f"  Fitness: {best['fitness']}")
    return best


def get_baseline_agents(archive_path: Path) -> List[dict]:
    """Return the dataset-appropriate canonical baseline definitions.

    Mind2Web uses the current definitions from ``mind2web_prompt.py`` so a
    completed search archive cannot silently keep stale baseline prompts during
    held-out comparison. The searched best agent is still loaded from the
    completed archive separately.
    """
    if DATASET == "SciCode":
        names = SCICODE_BASELINE_NAMES
    elif DATASET == "Mind2Web":
        names = MIND2WEB_BASELINE_NAMES
        from _mind2web.mind2web_prompt import get_init_archive

        initial = {a["name"]: a for a in get_init_archive()}
    else:
        names = BASELINE_NAMES
    if DATASET != "Mind2Web":
        with open(archive_path) as f:
            archive = json.load(f)
        initial = {a["name"]: a for a in archive if a.get("generation") == "initial"}
    selected = []
    for name in names:
        if name in initial:
            selected.append(initial[name])
        else:
            print(f"  [warn] baseline '{name}' not found in archive — skipping")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Token tracking helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_exec_tokens(mod) -> None:
    if hasattr(mod, '_exec_input_tokens'):
        mod._exec_input_tokens  = 0
        mod._exec_output_tokens = 0


def _read_exec_tokens(mod) -> int:
    if hasattr(mod, '_exec_input_tokens'):
        return mod._exec_input_tokens + mod._exec_output_tokens
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Module loading
# ─────────────────────────────────────────────────────────────────────────────

def load_search_module(dataset: str, args):
    """
    Dynamically load the correct search_ours module (_math, _mmlu_pro, or _fullstack),
    patch API globals, and return the module object.
    All agents share this single loaded module.
    """
    if dataset == "MATH":
        subdir = "_math"
    elif dataset == "FullStack":
        subdir = "_fullstack"
    elif dataset == "SciCode":
        subdir = "_scicode"
    elif dataset == "Mind2Web":
        subdir = "_mind2web"
    else:
        subdir = "_mmlu_pro"
    module_dir = _ADAS_DIR / subdir
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"search_ours_{dataset.lower()}",
            module_dir / "search_ours.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    import openai
    mod.client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    mod.MODEL = args.eval_model
    mod.EVAL_TEMPERATURE = args.eval_temperature
    mod.SEARCH_TEMPERATURE = 0.8
    mod.MAX_TOKENS = 32768
    mod.EXEC_MAX_TOKENS = args.exec_max_tokens
    mod.EXEC_NO_THINKING = args.no_exec_thinking
    mod.PROVIDER_ROUTING = {"order": [p.strip() for p in args.provider_order.split(",")], "allow_fallbacks": True} if args.provider_order else None
    mod.SEARCHING_MODE = True
    if hasattr(mod, "configure_reasoning_backend"):
        mod.configure_reasoning_backend(
            args.base_url, args.reasoning_backend
        )
    if dataset == "Mind2Web":
        mod.PROVIDER_ROUTING = mod.mind2web_provider_routing(
            args.base_url, mod.PROVIDER_ROUTING
        )
        if "openrouter.ai" in args.base_url.lower():
            mod.EXEC_NO_THINKING = True
    return mod


def setup_forward(mod, code_str: str):
    """exec agent code in the module's namespace and attach to AgentSystem."""
    namespace = {}
    exec(code_str, vars(mod), namespace)
    callables = [k for k, v in namespace.items() if callable(v)]
    if not callables:
        raise ValueError("No callable found in agent code.")
    func = namespace[callables[0]]
    setattr(mod.AgentSystem, "forward", func)
    return mod.AgentSystem()


# ─────────────────────────────────────────────────────────────────────────────
# Training fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def load_math_fingerprints() -> set:
    fps = set()
    with open(MATH_TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fps.add(json.loads(line)["problem"])
    print(f"  MATH training fingerprints: {len(fps)}")
    return fps


def load_mmlu_pro_fingerprints() -> set:
    import pandas as pd
    df = pd.read_csv(MMLU_PRO_TRAIN_CSV)
    fps = set(df["Question"].tolist())
    print(f"  MMLU-Pro training fingerprints: {len(fps)}")
    return fps


# ─────────────────────────────────────────────────────────────────────────────
# Held-out data  (built ONCE, shared across all agents)
# ─────────────────────────────────────────────────────────────────────────────

def build_math_heldout(rng: random.Random) -> List[dict]:
    """
    Load MATH Level-5 test problems for MATH_SUBJECTS from the raw cache,
    exclude training fingerprints, sample up to NUM_EVAL_QUERIES per subject.
    """
    fps = load_math_fingerprints()

    if not MATH_RAW_TEST_DIR.exists():
        raise FileNotFoundError(
            f"MATH raw data not found: {MATH_RAW_TEST_DIR}\n"
            "Run: python dataset/build_math_4subjects.py"
        )

    raw = []
    for subj_dir in MATH_RAW_TEST_DIR.iterdir():
        if not subj_dir.is_dir():
            continue
        for jf in subj_dir.glob("*.json"):
            with open(jf, encoding="utf-8") as f:
                raw.append(json.load(f))

    records = []
    for subject in MATH_SUBJECTS:
        pool = [
            r for r in raw
            if r.get("type") == subject
            and r.get("level") == MATH_LEVEL
            and r["problem"] not in fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {subject}: only {n} held-out available (requested {NUM_EVAL_QUERIES})")
        for r in rng.sample(pool, n):
            records.append({"subject": subject, "problem": r["problem"], "solution": r["solution"]})
        print(f"  {subject}: {n} held-out queries")
    return records


def build_mmlu_pro_heldout(rng: random.Random) -> List[dict]:
    """
    Load MMLU-Pro test examples from HuggingFace cache, exclude training
    fingerprints, sample up to NUM_EVAL_QUERIES per category.
    Row format matches format_multichoice_question (keys: Question, A-J, Answer, Category).
    """
    fps = load_mmlu_pro_fingerprints()
    LETTERS = list("ABCDEFGHIJ")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("  Loading TIGER-Lab/MMLU-Pro from HuggingFace cache …")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test",
                      cache_dir=str(MMLU_PRO_HF_CACHE))
    all_data = list(ds)

    records = []
    for category in MMLU_PRO_CATEGORIES:
        pool = [
            r for r in all_data
            if r["category"] == category and r["question"] not in fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {category}: only {n} held-out available (requested {NUM_EVAL_QUERIES})")
        for r in rng.sample(pool, n):
            options = list(r["options"])
            while len(options) < 10:
                options.append("")
            row = {"Question": r["question"], "Answer": str(r["answer"]).upper(),
                   "Category": category}
            for i, letter in enumerate(LETTERS):
                row[letter] = options[i]
            records.append(row)
        print(f"  {category}: {n} held-out queries")
    return records


def load_fullstack_fingerprints() -> set:
    fps = set()
    with open(FULLSTACK_TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fps.add(json.loads(line)["id"])
    print(f"  FullStack training fingerprints: {len(fps)}")
    return fps


def build_fullstack_heldout(rng: random.Random) -> List[dict]:
    fps = load_fullstack_fingerprints()
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("  Loading FullStackBench from HuggingFace cache ...")
    ds = load_dataset(
        "ByteDance/FullStackBench", "en", split="test",
        cache_dir=str(FULLSTACK_HF_CACHE) if FULLSTACK_HF_CACHE.exists() else None,
    )
    test_split = list(ds)

    records = []
    for category in FULLSTACK_SUBJECTS:
        pool = [
            ex for ex in test_split
            if ex["labels"].get("category") == category
            and ex["id"] not in fps
        ]
        hard   = [ex for ex in pool if ex["labels"].get("difficulty") == "hard"]
        medium = [ex for ex in pool if ex["labels"].get("difficulty") == "medium"]
        easy   = [ex for ex in pool if ex["labels"].get("difficulty") == "easy"]

        target_hard = NUM_EVAL_QUERIES // 2
        got_hard    = rng.sample(hard,   min(target_hard,         len(hard)))
        remaining   = NUM_EVAL_QUERIES - len(got_hard)
        got_medium  = rng.sample(medium, min(remaining,           len(medium)))
        remaining  -= len(got_medium)
        got_easy    = rng.sample(easy,   min(remaining,           len(easy)))

        for ex in got_hard + got_medium + got_easy:
            records.append({
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                "raw_example": dict(ex),
            })
        n = len(got_hard) + len(got_medium) + len(got_easy)
        print(f"  {category}: {len(got_hard)}h + {len(got_medium)}m + {len(got_easy)}e = {n} held-out queries")
    return records


def build_scicode_heldout() -> List[dict]:
    from _scicode.scicode_runtime import load_scicode_records

    records = load_scicode_records(SCICODE_TEST_JSONL, "heldout")
    print("  SciCode: 30 fixed main problems (10 per field; no resampling)")
    return records


def build_mind2web_heldout() -> List[dict]:
    from _mind2web.mind2web_runtime import load_mind2web_records

    records = load_mind2web_records(MIND2WEB_TEST_JSONL, "heldout")
    print("  Mind2Web: 300 fixed tasks (100 per domain; no resampling)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runners
# ─────────────────────────────────────────────────────────────────────────────

def run_math_eval(agent_system, held_out: List[dict], score_math, InfoCls=None) -> Dict[str, float]:
    _Info = InfoCls if InfoCls is not None else Info
    task_queue = [_Info("task", "User", ex["problem"], -1) for ex in held_out]
    solutions  = [ex["solution"] for ex in held_out]
    subjects   = [ex["subject"]  for ex in held_out]

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=min(len(held_out), MAX_WORKERS)) as exe:
        future_to_idx = {exe.submit(agent_system.forward, task): i for i, task in enumerate(task_queue)}
        results = [None] * len(task_queue)
        try:
            for future in tqdm(as_completed(future_to_idx, timeout=600), total=len(task_queue), desc="  running"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=180)
                except Exception:
                    results[idx] = None
        except TimeoutError:
            pass

    buckets: Dict[str, List[int]] = {s: [] for s in MATH_SUBJECTS}
    for idx, res in enumerate(results):
        try:
            pred = res.content if hasattr(res, 'content') else str(res)
            correct = int(score_math(solutions[idx], pred))
        except Exception:
            correct = 0
        buckets[subjects[idx]].append(correct)
    return {s: (sum(v) / len(v) if v else 0.0) for s, v in buckets.items()}


def run_mmlu_pro_eval(agent_system, held_out: List[dict],
                      format_q_fn, extract_answer_fn, InfoCls=None) -> Dict[str, float]:
    _Info = InfoCls if InfoCls is not None else Info
    task_queue = [_Info("task", "User", format_q_fn(ex), -1) for ex in held_out]
    answers    = [_MMLU_PRO_L2I.get(ex["Answer"], -1) for ex in held_out]
    categories = [ex["Category"] for ex in held_out]

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=min(len(held_out), MAX_WORKERS)) as exe:
        future_to_idx = {exe.submit(agent_system.forward, task): i for i, task in enumerate(task_queue)}
        results = [None] * len(task_queue)
        try:
            for future in tqdm(as_completed(future_to_idx, timeout=600), total=len(task_queue), desc="  running"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=180)
                except Exception:
                    results[idx] = None
        except TimeoutError:
            pass

    buckets: Dict[str, List[int]] = {c: [] for c in MMLU_PRO_CATEGORIES}
    for idx, res in enumerate(results):
        raw = res.content if hasattr(res, "content") else str(res)
        letter = extract_answer_fn(raw)
        pred_idx = _MMLU_PRO_L2I.get(letter, -1) if letter else -1
        buckets[categories[idx]].append(int(pred_idx == answers[idx]))
    return {c: (sum(v) / len(v) if v else 0.0) for c, v in buckets.items()}


def run_fullstack_eval(agent_system, held_out: List[dict], mod) -> Dict[str, float]:
    from _fullstack.utils import format_task, score_fullstack

    def run_and_score(idx):
        ex = held_out[idx]
        task = mod.Info("task", "User", format_task(ex), -1)
        try:
            res = agent_system.forward(task)
            prediction = res.content if hasattr(res, 'content') else str(res)
            if "```" not in prediction:
                lang = ex.get("programming_language", "python")
                prediction = f"``` {lang}\n{prediction}\n```"
            return score_fullstack(
                prediction=prediction,
                raw_example=ex["raw_example"],
                sandbox_endpoint=SANDBOX_ENDPOINT,
                compile_timeout=SANDBOX_COMPILE_TIMEOUT,
                run_timeout=SANDBOX_RUN_TIMEOUT,
            )
        except Exception as e:
            print(f"  Scoring error q{idx}: {e}")
            return 0.0

    workers = min(len(held_out), MAX_WORKERS)
    scores = [0.0] * len(held_out)
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=workers) as exe:
        future_to_idx = {exe.submit(run_and_score, i): i for i in range(len(held_out))}
        try:
            for future in tqdm(as_completed(future_to_idx, timeout=900), total=len(held_out), desc="  running"):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result(timeout=360)
                except Exception:
                    scores[idx] = 0.0
        except TimeoutError:
            pass

    per_category: Dict[str, List[float]] = {s: [] for s in FULLSTACK_SUBJECTS}
    for idx, score in enumerate(scores):
        per_category[held_out[idx]["category"]].append(score)

    return {s: (sum(v) / len(v) if v else 0.0) for s, v in per_category.items()}


def run_scicode_eval(held_out: List[dict], mod, args):
    from _scicode.scicode_runtime import evaluate_scicode

    return evaluate_scicode(
        held_out,
        mod.AgentSystem,
        lambda prompt: mod.Info("task", "User", prompt, -1),
        h5py_file=args.scicode_h5py_file,
        max_workers=args.scicode_max_workers,
        evaluation_timeout=args.scicode_evaluation_timeout,
        description="SciCode held-out evaluation",
    )


def run_mind2web_eval(held_out: List[dict], mod, args):
    from _mind2web.mind2web_runtime import evaluate_mind2web

    return evaluate_mind2web(
        held_out,
        mod.AgentSystem,
        lambda prompt: mod.Info("task", "User", prompt, -1),
        max_task_workers=args.mind2web_max_task_workers,
        max_llm_calls=args.mind2web_max_llm_calls,
        description="Mind2Web held-out evaluation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save(all_results: Dict[str, dict], all_stds: Dict[str, dict],
                   subjects: List[str], token_usage: Dict[str, float]):
    col_w = 36
    cell_w = 22
    agent_names = list(all_results.keys())

    def _table(f=None):
        def w(s):
            if f:
                f.write(s)
            else:
                print(s, end="")

        w(f"  {'Subject/Category':<{col_w}}")
        for name in agent_names:
            w(f"  {name[:20]:<{cell_w}}")
        w("\n  " + "-" * (col_w + (cell_w + 2) * len(agent_names)) + "\n")
        for s in subjects:
            w(f"  {s:<{col_w}}")
            for name in agent_names:
                avg = all_results[name].get(s, float('nan'))
                std = all_stds[name].get(s, 0.0)
                w(f"  {avg:.3f}±{std:.3f}{'':>{cell_w - 11}}")
            w("\n")
        w("\n")
        w(f"  {'AVERAGE':<{col_w}}")
        for name in agent_names:
            avg = sum(all_results[name][s] for s in subjects) / len(subjects)
            std = math.sqrt(sum(all_stds[name][s] ** 2 for s in subjects)) / len(subjects)
            w(f"  {avg:.3f}±{std:.3f}{'':>{cell_w - 11}}")
        w("\n\n")
        token_unit = (
            "subproblem" if DATASET == "SciCode"
            else "action" if DATASET == "Mind2Web"
            else "query"
        )
        w(f"  {f'avg_tokens/{token_unit}':<{col_w}}")
        for name in agent_names:
            w(f"  {token_usage.get(name, 0.0):<{cell_w}.1f}")
        w("\n")

    # console
    print("\n" + "=" * 70)
    eval_rounds = (
        SCICODE_K_ROUND_EVAL if DATASET == "SciCode"
        else MIND2WEB_K_ROUND_EVAL if DATASET == "Mind2Web"
        else K_ROUND_EVAL
    )
    print(f"SUMMARY — {DATASET}  [{eval_rounds} rounds, mean ± std]")
    print("=" * 70)
    _table()
    print("=" * 70)

    # file
    out_dir = _ADAS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"heldout_eval_{DATASET.lower()}_{timestamp}.txt"

    with open(out_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"ADAS COMPREHENSIVE HELD-OUT EVALUATION — {DATASET}\n")
        f.write("=" * 70 + "\n")
        queries_per_subject = (
            10 if DATASET == "SciCode"
            else 100 if DATASET == "Mind2Web"
            else NUM_EVAL_QUERIES
        )
        f.write(f"Queries/subject:   {queries_per_subject}\n")
        f.write(f"Eval rounds (k):   {eval_rounds}\n")
        f.write(f"Sampling seed:     {SEED}\n")
        f.write(f"Date:              {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        _table(f)
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', type=str, default='deepseek/deepseek-v4-flash')
    parser.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1')
    parser.add_argument('--reasoning_backend', choices=['auto', 'openrouter', 'local'],
                        default='auto', help='Reasoning request schema for the endpoint')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--eval_temperature', type=float, default=1.0)
    parser.add_argument('--exec_max_tokens', type=int, default=8600)
    parser.add_argument('--provider_order', type=str, default="alibaba, deepseek ")
    parser.add_argument('--no_exec_thinking', action='store_true', default=True)
    parser.add_argument('--scicode_h5py_file', type=str,
                        default=os.environ.get('SCICODE_H5_PATH'),
                        help='Official SciCode test_data.h5 (or set SCICODE_H5_PATH)')
    parser.add_argument('--scicode_evaluation_timeout', type=int, default=1800)
    parser.add_argument('--scicode_max_workers', type=int, default=3)
    parser.add_argument('--mind2web_max_task_workers', type=int, default=50)
    parser.add_argument('--mind2web_max_llm_calls', type=int, default=32)
    args = parser.parse_args()
    if args.api_key is None:
        url = args.base_url.lower()
        if "groq" in url:
            args.api_key = os.environ.get("GROQ_API_KEY")
        elif "openrouter" in url:
            args.api_key = os.environ.get("OPENROUTER_API_KEY")
        elif "openai" in url:
            args.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            args.api_key = (os.environ.get("OPENROUTER_API_KEY")
                            or os.environ.get("GROQ_API_KEY")
                            or os.environ.get("OPENAI_API_KEY"))
    if not args.api_key:
        raise ValueError("Provide --api_key or set OPENROUTER_API_KEY / GROQ_API_KEY / OPENAI_API_KEY")
    return args


def main():
    args = _parse_args()
    os.chdir(_ADAS_DIR)
    rng = random.Random(SEED)
    if DATASET == "MATH":
        subjects     = MATH_SUBJECTS
        archive_path = MATH_ARCHIVE
    elif DATASET == "FullStack":
        subjects     = FULLSTACK_SUBJECTS
        archive_path = FULLSTACK_ARCHIVE
    elif DATASET == "SciCode":
        subjects     = SCICODE_FIELDS
        archive_path = SCICODE_ARCHIVE
    elif DATASET == "Mind2Web":
        subjects     = MIND2WEB_DOMAINS
        archive_path = MIND2WEB_ARCHIVE
    else:
        subjects     = MMLU_PRO_CATEGORIES
        archive_path = MMLU_PRO_ARCHIVE

    print("=" * 70)
    print(f"ADAS COMPREHENSIVE HELD-OUT EVALUATION  —  {DATASET}")
    query_count = (
        10 if DATASET == "SciCode"
        else 100 if DATASET == "Mind2Web"
        else NUM_EVAL_QUERIES
    )
    print(f"Queries/subject: {query_count}  |  seed: {SEED}")
    print("=" * 70)

    # 1. Load module once — shared by all agents
    print("\nLoading search module …")
    mod = load_search_module(DATASET, args)

    # 2. Collect configured baselines; trajectory benchmarks include their best agent.
    print("\nCollecting agents from archive …")
    agents_to_run: List[tuple] = []

    for entry in get_baseline_agents(archive_path):
        agents_to_run.append((entry["name"], entry["code"]))
    if DATASET in {"SciCode", "Mind2Web"}:
        best = find_best_agent(archive_path)
        agents_to_run.append((f"Best searched: {best['name']}", best["code"]))

    print(f"\nAgents to evaluate ({len(agents_to_run)}):")
    for name, _ in agents_to_run:
        print(f"  • {name}")

    # 3. Build held-out queries ONCE — every agent sees the exact same set
    print(f"\nBuilding held-out queries (seed={SEED}) …")
    if DATASET == "MATH":
        held_out = build_math_heldout(rng)
    elif DATASET == "FullStack":
        held_out = build_fullstack_heldout(rng)
    elif DATASET == "SciCode":
        held_out = build_scicode_heldout()
    elif DATASET == "Mind2Web":
        held_out = build_mind2web_heldout()
    else:
        held_out = build_mmlu_pro_heldout(rng)
    print(f"Total held-out examples: {len(held_out)}")

    # 4. Evaluate each agent on the same data; trajectory benchmarks use one round.
    eval_rounds = (
        SCICODE_K_ROUND_EVAL if DATASET == "SciCode"
        else MIND2WEB_K_ROUND_EVAL if DATASET == "Mind2Web"
        else K_ROUND_EVAL
    )
    all_results: Dict[str, dict] = {}
    all_stds: Dict[str, dict] = {}
    token_usage: Dict[str, float] = {}
    for agent_name, agent_code in agents_to_run:
        print(f"\n{'─' * 70}")
        print(f"Agent: {agent_name}  [{eval_rounds} rounds]")
        print(f"{'─' * 70}")
        agent_system = setup_forward(mod, agent_code)

        rounds = []
        round_tokens_list = []
        scicode_subproblem_count = 0
        mind2web_action_count = 0
        for k in range(eval_rounds):
            print(f"  Round {k + 1}/{eval_rounds}")
            mod.EVAL_SEED = k
            _reset_exec_tokens(mod)
            if DATASET == "MATH":
                round_result = run_math_eval(agent_system, held_out, mod.score_math, mod.Info)
            elif DATASET == "FullStack":
                round_result = run_fullstack_eval(agent_system, held_out, mod)
            elif DATASET == "SciCode":
                evaluation = run_scicode_eval(held_out, mod, args)
                round_result = evaluation.metrics["field_subproblem_pass_rates"]
                scicode_subproblem_count += evaluation.metrics["total_subproblems"]
            elif DATASET == "Mind2Web":
                evaluation = run_mind2web_eval(held_out, mod, args)
                round_result = {
                    scenario_id: values["macro_step_success_rate"]
                    for scenario_id, values in evaluation.metrics["scenarios"].items()
                }
                mind2web_action_count += evaluation.metrics["n_steps"]
            else:
                round_result = run_mmlu_pro_eval(
                    agent_system, held_out,
                    mod.format_multichoice_question,
                    mod.extract_answer,
                    mod.Info,
                )
            round_tokens_list.append(_read_exec_tokens(mod))
            rounds.append(round_result)

        per_subject_avg = {s: sum(r[s] for r in rounds) / eval_rounds for s in subjects}
        per_subject_std = {s: math.sqrt(sum((r[s] - per_subject_avg[s]) ** 2 for r in rounds) / eval_rounds) for s in subjects}
        if DATASET == "SciCode":
            token_denominator = scicode_subproblem_count
        elif DATASET == "Mind2Web":
            token_denominator = mind2web_action_count
        else:
            token_denominator = eval_rounds * len(held_out)
        avg_tokens = sum(round_tokens_list) / token_denominator if token_denominator else 0.0
        token_usage[agent_name] = avg_tokens

        avg = sum(per_subject_avg[s] for s in subjects) / len(subjects)
        for s in subjects:
            print(f"  {s:<35s}  {per_subject_avg[s]:.3f} ± {per_subject_std[s]:.3f}")
        token_unit = (
            "subproblem" if DATASET == "SciCode"
            else "action" if DATASET == "Mind2Web"
            else "query"
        )
        print(f"  {'AVERAGE':<35s}  {avg:.3f}  avg_tokens/{token_unit}: {avg_tokens:.1f}")
        all_results[agent_name] = per_subject_avg
        all_stds[agent_name] = per_subject_std

    # 5. Print summary table + save
    print_and_save(all_results, all_stds, subjects, token_usage)


if __name__ == "__main__":
    main()
