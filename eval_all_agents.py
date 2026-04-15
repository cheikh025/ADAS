"""
ADAS Comprehensive Held-out Evaluation

Evaluates the following agents on the SAME fresh held-out queries:
  1. Chain-of-Thought (CoT)
  2. Self-Consistency with Chain-of-Thought (CoT-SC)
  3. Self-Refine (Reflexion)
  4. Best searched agent (highest median fitness from archive)

Supports DATASET = "MATH" or "MMLU_PRO"

Run:
    python eval_all_agents.py
"""

import importlib.util
import json
import os
import random
import re
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
_ADAS_DIR  = Path(__file__).parent.resolve()
_AFLOW_DIR = _ADAS_DIR.parent / "AFlow"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← change here
# ─────────────────────────────────────────────────────────────────────────────
DATASET          = "MATH"   # "MATH"  or  "MMLU_PRO"
NUM_EVAL_QUERIES = 100      # held-out queries per subject / category
MAX_WORKERS      = 50       # parallel threads
SEED             = 99       # sampling seed (training used 42)
# ─────────────────────────────────────────────────────────────────────────────

# 3 subjects — matches MAS_pro and AFlow for a fair comparison
# Note: ADAS was searched on 4 subjects (add "Prealgebra" here if needed)
MATH_SUBJECTS = [
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
MMLU_PRO_CATEGORIES = ["law", "history", "philosophy", "engineering"]
MATH_LEVEL = "Level 5"

MATH_TRAIN_JSONL   = _ADAS_DIR / "dataset/math_4subjects.jsonl"
MMLU_PRO_TRAIN_CSV = _ADAS_DIR / "dataset/mmlu_pro_4categories.csv"
MATH_RAW_TEST_DIR  = _AFLOW_DIR / "data/math_hf_cache/MATH/test"
MMLU_PRO_HF_CACHE  = _AFLOW_DIR / "data/mmlu_pro_hf_cache"

MATH_ARCHIVE     = _ADAS_DIR / "results/math_ours_results_run_archive.json"
MMLU_PRO_ARCHIVE = _ADAS_DIR / "results/mmlu_pro_ours_results_run_archive.json"

# Baselines pulled from the initial archive entries
BASELINE_NAMES = [
    "Chain-of-Thought",
    "Self-Consistency with Chain-of-Thought",
    "Self-Refine (Reflexion)",
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
    candidates = [a for a in archive if "fitness" in a]
    best = max(candidates, key=lambda a: _parse_median(a["fitness"]))
    print(f"  Best agent: gen={best['generation']}, name={best['name']!r}")
    print(f"  Fitness: {best['fitness']}")
    return best


def get_baseline_agents(archive_path: Path) -> List[dict]:
    """Return initial-archive entries for BASELINE_NAMES, in that order."""
    with open(archive_path) as f:
        archive = json.load(f)
    initial = {a["name"]: a for a in archive if a.get("generation") == "initial"}
    selected = []
    for name in BASELINE_NAMES:
        if name in initial:
            selected.append(initial[name])
        else:
            print(f"  [warn] baseline '{name}' not found in archive — skipping")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Module loading
# ─────────────────────────────────────────────────────────────────────────────

def load_search_module(dataset: str):
    """
    Dynamically load the correct search_ours module (_math or _mmlu_pro),
    patch API globals, and return the module object.
    All agents share this single loaded module.
    """
    subdir = "_math" if dataset == "MATH" else "_mmlu_pro"
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
    mod.client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    mod.MODEL = "openai/gpt-4o-mini-2024-07-18"
    mod.EVAL_TEMPERATURE = 1.0
    mod.SEARCH_TEMPERATURE = 0.8
    mod.MAX_TOKENS = 32768
   # mod.PROVIDER_ROUTING = {
   #     "order": ["Google Vertex", "Together", "Groq"],
   #     "allow_fallbacks": True,
   # }
    mod.SEARCHING_MODE = True
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
            "Run: python AFlow/data/build_math_validate.py  (downloads the cache)"
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
        results = list(tqdm(exe.map(agent_system.forward, task_queue),
                            total=len(task_queue), desc="  running"))

    buckets: Dict[str, List[int]] = {s: [] for s in MATH_SUBJECTS}
    for idx, res in enumerate(results):
        try:
            pred = res.content if isinstance(res, Info) else str(res)
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
        results = list(tqdm(exe.map(agent_system.forward, task_queue),
                            total=len(task_queue), desc="  running"))

    buckets: Dict[str, List[int]] = {c: [] for c in MMLU_PRO_CATEGORIES}
    for idx, res in enumerate(results):
        raw = res.content if hasattr(res, "content") else str(res)
        letter = extract_answer_fn(raw)
        pred_idx = _MMLU_PRO_L2I.get(letter, -1) if letter else -1
        buckets[categories[idx]].append(int(pred_idx == answers[idx]))
    return {c: (sum(v) / len(v) if v else 0.0) for c, v in buckets.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save(all_results: Dict[str, dict], subjects: List[str]):
    col_w = 36
    agent_names = list(all_results.keys())

    def _table(f=None):
        def w(s):
            if f:
                f.write(s)
            else:
                print(s, end="")

        w(f"  {'Subject/Category':<{col_w}}")
        for name in agent_names:
            w(f"  {name[:16]:<16}")
        w("\n  " + "-" * (col_w + 18 * len(agent_names)) + "\n")
        for s in subjects:
            w(f"  {s:<{col_w}}")
            for name in agent_names:
                w(f"  {all_results[name].get(s, float('nan')):.4f}          ")
            w("\n")
        w("\n")
        w(f"  {'AVERAGE':<{col_w}}")
        for name in agent_names:
            avg = sum(all_results[name][s] for s in subjects) / len(subjects)
            w(f"  {avg:.4f}          ")
        w("\n")

    # console
    print("\n" + "=" * 70)
    print(f"SUMMARY — {DATASET}")
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
        f.write(f"Queries/subject:   {NUM_EVAL_QUERIES}\n")
        f.write(f"Sampling seed:     {SEED}\n")
        f.write(f"Date:              {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        _table(f)
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(_ADAS_DIR)
    rng = random.Random(SEED)
    subjects     = MATH_SUBJECTS if DATASET == "MATH" else MMLU_PRO_CATEGORIES
    archive_path = MATH_ARCHIVE  if DATASET == "MATH" else MMLU_PRO_ARCHIVE

    print("=" * 70)
    print(f"ADAS COMPREHENSIVE HELD-OUT EVALUATION  —  {DATASET}")
    print(f"Queries/subject: {NUM_EVAL_QUERIES}  |  seed: {SEED}")
    print("=" * 70)

    # 1. Load module once — shared by all agents
    print("\nLoading search module …")
    mod = load_search_module(DATASET)

    # 2. Collect agents: baselines first, then best searched
    print("\nCollecting agents from archive …")
    agents_to_run: List[tuple] = []

    for entry in get_baseline_agents(archive_path):
        agents_to_run.append((entry["name"], entry["code"]))

    best_entry = find_best_agent(archive_path)
    label = f"Best-ADAS ({best_entry['name'][:20]})"
    agents_to_run.append((label, best_entry["code"]))

    print(f"\nAgents to evaluate ({len(agents_to_run)}):")
    for name, _ in agents_to_run:
        print(f"  • {name}")

    # 3. Build held-out queries ONCE — every agent sees the exact same set
    print(f"\nBuilding held-out queries (seed={SEED}) …")
    if DATASET == "MATH":
        held_out = build_math_heldout(rng)
    else:
        held_out = build_mmlu_pro_heldout(rng)
    print(f"Total held-out examples: {len(held_out)}")

    # 4. Evaluate each agent on the same queries
    all_results: Dict[str, dict] = {}
    for agent_name, agent_code in agents_to_run:
        print(f"\n{'─' * 70}")
        print(f"Agent: {agent_name}")
        print(f"{'─' * 70}")
        agent_system = setup_forward(mod, agent_code)

        if DATASET == "MATH":
            per_subject = run_math_eval(agent_system, held_out, mod.score_math, mod.Info)
        else:
            per_subject = run_mmlu_pro_eval(
                agent_system, held_out,
                mod.format_multichoice_question,
                mod.extract_answer,
                mod.Info,
            )

        avg = sum(per_subject[s] for s in subjects) / len(subjects)
        for s in subjects:
            print(f"  {s:<35s}  {per_subject[s]:.4f}")
        print(f"  {'AVERAGE':<35s}  {avg:.4f}")
        all_results[agent_name] = per_subject

    # 5. Print summary table + save
    print_and_save(all_results, subjects)


if __name__ == "__main__":
    main()
