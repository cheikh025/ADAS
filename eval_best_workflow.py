"""
ADAS Held-out Evaluation of Best Workflow

Evaluates the best ADAS agent (highest median fitness) on queries
that were NOT seen during training (search phase).

Configuration:
    Set DATASET = "MATH" or "MMLU" below, then run from any directory:
        python Baseline/ADAS/eval_best_workflow.py

Held-out query sampling:
    1. Fingerprint every problem in the training dataset file
       (math_4subjects.jsonl or mmlu_4subjects.csv — all 120 rows,
        including the ~80 that were never actually used during search).
    2. Load the full raw test split from the same source.
    3. Exclude fingerprinted problems — zero overlap guaranteed.
    4. Sample up to NUM_EVAL_QUERIES per subject (seed=99).

Training recap:
    MATH : math_4subjects.jsonl  — 120 total (30/subj × 4), shuffle_seed=0,
           valid_size=40  →  first 40 after shuffle were used in search.
    MMLU : mmlu_4subjects.csv    — 120 total (20/subj × 6), shuffle_seed=0,
           valid_size=40  →  first 40 after shuffle were used in search.
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
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# ─── paths ────────────────────────────────────────────────────────────────────
_ADAS_DIR  = Path(__file__).parent.resolve()
_AFLOW_DIR = _ADAS_DIR.parent / "AFlow"   # reuse AFlow's raw data caches

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET          = "MMLUPro"   # "MATH", "MMLU", or "MMLUPro"
NUM_EVAL_QUERIES = 50      # held-out queries per subject
MAX_WORKERS      = 50       # parallel threads
SEED             = 99       # sampling seed (training used 42)
K_ROUND_EVAL     = 5       # number of eval rounds (scores averaged); matches MAS_pro
# ──────────────────────────────────────────────────────────────────────────────

MATH_SUBJECTS = [
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
MMLU_SUBJECTS = [
    "international_law",
    "anatomy",
    "business_ethics",
    "college_chemistry",
    "moral_scenarios",
    "econometrics",
]
MMLU_PRO_SUBJECTS = [
    "law",
    "history",
    "philosophy",
    "engineering",
]
MATH_LEVEL = "Level 5"

MATH_TRAIN_JSONL      = _ADAS_DIR / "dataset/math_4subjects.jsonl"
MMLU_TRAIN_CSV        = _ADAS_DIR / "dataset/mmlu_4subjects.csv"
MMLU_PRO_VALIDATE_JSONL = _AFLOW_DIR / "data/datasets/mmlu_pro_validate.jsonl"
MATH_RAW_TEST_DIR     = _AFLOW_DIR / "data/math_hf_cache/MATH/test"
MMLU_HF_CACHE         = _AFLOW_DIR / "data/mmlu_hf_cache"
MMLU_PRO_HF_CACHE     = _AFLOW_DIR / "data/mmlu_pro_hf_cache"

MATH_ARCHIVE     = _ADAS_DIR / "results/math_ours_results_run_archive.json"
MMLU_ARCHIVE     = _ADAS_DIR / "results/mmlu_ours_results_run_archive.json"
MMLU_PRO_ARCHIVE = _ADAS_DIR / "results/mmlu_pro_ours_results_run_archive.json"

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
LETTER_TO_INDEX     = {"A": 0, "B": 1, "C": 2, "D": 3}
MMLU_PRO_LETTERS    = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
MMLU_PRO_LETTER_TO_INDEX = {l: i for i, l in enumerate(MMLU_PRO_LETTERS)}


# ─────────────────────────────────────────────────────────────────────────────
# Best-agent discovery
# ─────────────────────────────────────────────────────────────────────────────

def _parse_median(fitness_str: str) -> float:
    m = re.search(r"Median:\s*([\d.]+)%", fitness_str)
    return float(m.group(1)) if m else 0.0


def find_best_agent(archive_path: Path) -> dict:
    with open(archive_path) as f:
        archive = json.load(f)
    # Only consider evolved generations (integer generation), not the initial baselines
    candidates = [
        a for a in archive
        if "fitness" in a and isinstance(a.get("generation"), int)
    ]
    if not candidates:
        raise ValueError(
            f"No evolved agents found in {archive_path}. "
            "All entries have generation='initial'. Run the ADAS search first."
        )
    best = max(candidates, key=lambda a: _parse_median(a["fitness"]))
    print(f"Best agent: gen={best['generation']}, name={best['name']}")
    print(f"  Fitness: {best['fitness']}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Module loading (mirrors how search_ours.py sets up globals)
# ─────────────────────────────────────────────────────────────────────────────

def load_search_module(dataset: str):
    """
    Dynamically load the correct search_ours module (_math or _mmlu),
    patching its globals so the forward function code can call
    LLMAgentBase, get_json_response_from_gpt, client, MODEL, etc.
    """
    module_dir = _ADAS_DIR / ("_math" if dataset == "MATH" else "_mmlu")
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

    # Patch API globals
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    import openai
    mod.client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    mod.MODEL = "openai/gpt-4o-mini-2024-07-18"
    mod.EVAL_TEMPERATURE = 1.0
    mod.SEARCH_TEMPERATURE = 0.8
    mod.MAX_TOKENS = 32768
    #mod.PROVIDER_ROUTING = {
    #    "order": ["Google Vertex", "Together", "Groq"],
    #    "allow_fallbacks": True,
    #}
    mod.SEARCHING_MODE = True   # not used during eval, but set for safety
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Forward function setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_forward(mod, code_str: str):
    """exec the agent code in the module's global namespace and attach to AgentSystem."""
    namespace = {}
    exec(code_str, vars(mod), namespace)
    names = [k for k, v in namespace.items() if callable(v)]
    if not names:
        raise ValueError("No callable found in agent code.")
    func = namespace[names[0]]
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
    print(f"  Math training fingerprints: {len(fps)}")
    return fps


def load_mmlu_fingerprints() -> set:
    import pandas as pd
    df = pd.read_csv(MMLU_TRAIN_CSV)
    fps = set(df["Question"].tolist())
    print(f"  MMLU training fingerprints: {len(fps)}")
    return fps


def load_mmlu_pro_fingerprints() -> set:
    if not MMLU_PRO_VALIDATE_JSONL.exists():
        print(f"  [warn] mmlu_pro_validate.jsonl not found: {MMLU_PRO_VALIDATE_JSONL}")
        return set()
    fps = set()
    with open(MMLU_PRO_VALIDATE_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fps.add(json.loads(line)["question"])
    print(f"  MMLU-Pro training fingerprints: {len(fps)}")
    return fps


# ─────────────────────────────────────────────────────────────────────────────
# Held-out data building
# ─────────────────────────────────────────────────────────────────────────────

def build_math_heldout(rng: random.Random) -> List[dict]:
    """
    Load MATH Level-5 test problems for the 4 subjects from the raw cache,
    exclude training fingerprints, sample up to NUM_EVAL_QUERIES per subject.
    Returns list of dicts: {subject, problem, solution, level, type}.
    """
    fps = load_math_fingerprints()

    if not MATH_RAW_TEST_DIR.exists():
        raise FileNotFoundError(
            f"MATH raw data not found: {MATH_RAW_TEST_DIR}\n"
            "Run: python Baseline/AFlow/data/build_math_validate.py"
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
        sampled = rng.sample(pool, n)
        for r in sampled:
            records.append({
                "subject": subject,
                "problem": r["problem"],
                "solution": r["solution"],
                "level": r.get("level", MATH_LEVEL),
                "type": r.get("type", subject),
            })
        print(f"  {subject}: {n} held-out queries")
    return records


def build_mmlu_heldout(rng: random.Random) -> List[dict]:
    """
    Load MMLU test examples from HuggingFace cache, exclude training
    fingerprints, sample up to NUM_EVAL_QUERIES per subject.
    Returns list of dicts with ADAS CSV-style keys (Question, A, B, C, D, Answer, Subject).
    """
    fps = load_mmlu_fingerprints()
    LETTER = ["A", "B", "C", "D"]

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print("  Loading MMLU from HuggingFace cache...")
    ds = load_dataset("cais/mmlu", "all", cache_dir=str(MMLU_HF_CACHE))
    test_split = list(ds["test"])

    records = []
    for subject in MMLU_SUBJECTS:
        pool = [
            r for r in test_split
            if r["subject"] == subject
            and r["question"] not in fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {subject}: only {n} held-out available (requested {NUM_EVAL_QUERIES})")
        sampled = rng.sample(pool, n)
        for r in sampled:
            choices = list(r["choices"])
            records.append({
                "Question": r["question"],
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "Answer": LETTER[r["answer"]],
                "Subject": subject,
            })
        print(f"  {subject}: {n} held-out queries")
    return records


def build_mmlu_pro_heldout(rng: random.Random) -> List[dict]:
    """
    Load MMLU-Pro test examples for the 4 categories, exclude training
    fingerprints, sample up to NUM_EVAL_QUERIES per category.
    Returns list of dicts: {question, options, formatted_choices, answer, subject}.
    """
    fps = load_mmlu_pro_fingerprints()

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print("  Loading MMLU-Pro from HuggingFace cache...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(MMLU_PRO_HF_CACHE), split="test")
    test_split = list(ds)

    records = []
    for category in MMLU_PRO_SUBJECTS:
        pool = [
            r for r in test_split
            if r["category"] == category
            and r["question"] not in fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {category}: only {n} held-out available (requested {NUM_EVAL_QUERIES})")
        sampled = rng.sample(pool, n)
        for r in sampled:
            options = list(r["options"])
            formatted = "\n".join(f"{MMLU_PRO_LETTERS[i]}) {opt}" for i, opt in enumerate(options))
            records.append({
                "question": r["question"],
                "options": options,
                "formatted_choices": formatted,
                "answer": str(r["answer"]).upper(),
                "subject": category,
            })
        print(f"  {category}: {n} held-out queries")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mmlu_prediction(res) -> int:
    """Extract predicted choice index (0-3) from agent output. Returns -1 on failure."""
    try:
        content = res.content if hasattr(res, "content") else str(res)
        if content in LETTER_TO_INDEX:
            return LETTER_TO_INDEX[content]
        if isinstance(res, str):
            text = res
        else:
            text = content
        for letter in ("A", "B", "C", "D"):
            if f"{letter})" in text or f"({letter})" in text:
                return LETTER_TO_INDEX[letter]
    except Exception:
        pass
    return -1


def run_mmlu_eval(agent_system, held_out: List[dict], format_q_fn) -> Dict[str, float]:
    questions = [format_q_fn(ex) for ex in held_out]
    answers   = [LETTER_TO_INDEX[ex["Answer"]] for ex in held_out]
    subjects  = [ex["Subject"] for ex in held_out]
    task_queue = [Info("task", "User", q, -1) for q in questions]

    workers = min(len(held_out), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        from tqdm import tqdm
        results = list(tqdm(exe.map(agent_system.forward, task_queue),
                            total=len(task_queue), desc="MMLU eval"))

    per_subject: Dict[str, List[int]] = {s: [] for s in MMLU_SUBJECTS}
    for idx, res in enumerate(results):
        pred_idx = _extract_mmlu_prediction(res)
        correct = int(pred_idx == answers[idx])
        per_subject[subjects[idx]].append(correct)

    return {s: (sum(v) / len(v) if v else 0.0) for s, v in per_subject.items()}


def _extract_mmlu_pro_prediction(res) -> int:
    """
    Extract predicted choice index (0-9) from agent output.
    Mirrors the original ADAS extraction logic (evaluate_forward_fn) but extended to A-J.
    Agent forward() typically returns a list of Info namedtuples: [thinking, answer, ...]
    """
    def _from_str(s: str) -> int:
        s = s.strip()
        if s in MMLU_PRO_LETTER_TO_INDEX:
            return MMLU_PRO_LETTER_TO_INDEX[s]
        for letter in MMLU_PRO_LETTERS:
            if f"{letter})" in s or f"({letter})" in s:
                return MMLU_PRO_LETTER_TO_INDEX[letter]
        return -1

    try:
        if isinstance(res, list):
            # Most agents: [thinking_Info, answer_Info] — answer at index 1
            if len(res) > 1:
                idx = _from_str(str(res[1].content))
                if idx != -1:
                    return idx
            # Fallback: scan all Info items for a single-letter content
            for item in res:
                if isinstance(item, Info):
                    c = str(item.content).strip()
                    if c in MMLU_PRO_LETTER_TO_INDEX:
                        return MMLU_PRO_LETTER_TO_INDEX[c]
        elif hasattr(res, "content"):
            return _from_str(str(res.content))
        else:
            return _from_str(str(res))
    except Exception:
        pass
    return -1


def run_math_eval(agent_system, score_math_fn, held_out: List[dict]) -> Dict[str, float]:
    questions  = [ex["problem"]  for ex in held_out]
    solutions  = [ex["solution"] for ex in held_out]
    subjects   = [ex["subject"]  for ex in held_out]
    task_queue = [Info("task", "User", q, -1) for q in questions]

    workers = min(len(held_out), MAX_WORKERS)
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=workers) as exe:
        results = list(tqdm(exe.map(agent_system.forward, task_queue),
                            total=len(task_queue), desc="MATH eval"))

    buckets: Dict[str, List[int]] = {s: [] for s in MATH_SUBJECTS}
    for idx, res in enumerate(results):
        try:
            pred = res.content if isinstance(res, Info) else str(res)
            correct = int(score_math_fn(solutions[idx], pred))
        except Exception:
            correct = 0
        buckets[subjects[idx]].append(correct)
    return {s: (sum(v) / len(v) if v else 0.0) for s, v in buckets.items()}


def run_mmlu_pro_eval(agent_system, held_out: List[dict]) -> Dict[str, float]:
    def _format_q(ex):
        return f"{ex['question']}\n{ex['formatted_choices']}"

    questions = [_format_q(ex) for ex in held_out]
    answers   = [MMLU_PRO_LETTER_TO_INDEX.get(ex["answer"], -1) for ex in held_out]
    subjects  = [ex["subject"] for ex in held_out]
    task_queue = [Info("task", "User", q, -1) for q in questions]

    workers = min(len(held_out), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        from tqdm import tqdm
        results = list(tqdm(exe.map(agent_system.forward, task_queue),
                            total=len(task_queue), desc="MMLUPro eval"))

    per_subject: Dict[str, List[int]] = {s: [] for s in MMLU_PRO_SUBJECTS}
    for idx, res in enumerate(results):
        pred_idx = _extract_mmlu_pro_prediction(res)
        correct = int(pred_idx == answers[idx])
        per_subject[subjects[idx]].append(correct)

    return {s: (sum(v) / len(v) if v else 0.0) for s, v in per_subject.items()}


# ─────────────────────────────────────────────────────────────────────────────
# K-round averaging
# ─────────────────────────────────────────────────────────────────────────────

def _avg_rounds(rounds: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-subject accuracy dicts across K evaluation rounds."""
    keys = rounds[0].keys()
    return {k: sum(r[k] for r in rounds) / len(rounds) for k in keys}


def _reset_exec_tokens(mod) -> None:
    """Zero the execution token counters in the loaded search module (if present)."""
    if hasattr(mod, '_exec_input_tokens'):
        mod._exec_input_tokens  = 0
        mod._exec_output_tokens = 0


def _read_exec_tokens(mod) -> int:
    """Return total execution tokens consumed since last reset (input + output)."""
    if hasattr(mod, '_exec_input_tokens'):
        return mod._exec_input_tokens + mod._exec_output_tokens
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(per_subject: dict, agent_name: str, dataset: str, avg_tokens_per_query: float = 0.0):
    out_dir = _ADAS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"heldout_eval_{dataset.lower()}_{timestamp}.txt"

    if dataset == "MATH":
        subjects = MATH_SUBJECTS
    elif dataset == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    else:
        subjects = MMLU_SUBJECTS
    avg = sum(per_subject[s] for s in subjects) / len(subjects)

    with open(out_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"ADAS HELD-OUT EVALUATION — {dataset}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Agent:             {agent_name}\n")
        f.write(f"Queries/subject:   {NUM_EVAL_QUERIES}\n")
        f.write(f"Eval rounds (k):   {K_ROUND_EVAL}\n")
        f.write(f"Sampling seed:     {SEED}\n")
        f.write(f"Date:              {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        for s in subjects:
            f.write(f"  {s:<35s}  {per_subject[s]:.4f}\n")
        f.write(f"\n  {'AVERAGE':<35s}  {avg:.4f}  avg_tokens/query: {avg_tokens_per_query:.1f}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {out_file}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(_ADAS_DIR)   # needed so relative dataset paths resolve
    rng = random.Random(SEED)

    print("=" * 70)
    print(f"ADAS HELD-OUT EVALUATION  —  {DATASET}")
    print(f"Queries/subject: {NUM_EVAL_QUERIES}  |  seed: {SEED}")
    print("=" * 70)

    # 1. Find best agent
    if DATASET == "MATH":
        archive_path = MATH_ARCHIVE
    elif DATASET == "MMLUPro":
        archive_path = MMLU_PRO_ARCHIVE
    else:
        archive_path = MMLU_ARCHIVE
    best_agent = find_best_agent(archive_path)

    # 2. Load module and set up agent
    print("\nLoading search module and patching globals …")
    mod = load_search_module(DATASET)

    # For MMLUPro: patch LLMAgentBase so it instructs the model to return A-J
    # instead of the hardcoded "A or B or C or D" that the _mmlu module uses.
    if DATASET == "MMLUPro":
        _original_gp = mod.LLMAgentBase.generate_prompt
        def _patched_gp(self, input_infos, instruction):
            sys_prompt, prompt = _original_gp(self, input_infos, instruction)
            sys_prompt = sys_prompt.replace(
                "A or B or C or D.",
                "A or B or C or D or E or F or G or H or I or J.",
            )
            return sys_prompt, prompt
        mod.LLMAgentBase.generate_prompt = _patched_gp
        print("  Patched LLMAgentBase to support A-J choices.")

    agent_system = setup_forward(mod, best_agent["code"])
    print("Agent ready.")

    # 3. Build held-out data
    print("\nBuilding held-out queries (excluding training set) …")
    if DATASET == "MATH":
        held_out = build_math_heldout(rng)
    elif DATASET == "MMLUPro":
        held_out = build_mmlu_pro_heldout(rng)
    else:
        held_out = build_mmlu_heldout(rng)
    print(f"Total held-out examples: {len(held_out)}")

    # 4. Run evaluation — K_ROUND_EVAL rounds, scores averaged (matches MAS_pro)
    print(f"\nRunning evaluation ({K_ROUND_EVAL} rounds × {len(held_out)} queries, max_workers={MAX_WORKERS}) …")
    rounds = []
    round_tokens: List[int] = []
    for k in range(K_ROUND_EVAL):
        print(f"  Round {k + 1}/{K_ROUND_EVAL}")
        _reset_exec_tokens(mod)
        if DATASET == "MATH":
            round_result = run_math_eval(agent_system, mod.score_math, held_out)
        elif DATASET == "MMLUPro":
            round_result = run_mmlu_pro_eval(agent_system, held_out)
        else:
            round_result = run_mmlu_eval(agent_system, held_out, mod.format_multichoice_question)
        round_tokens.append(_read_exec_tokens(mod))
        rounds.append(round_result)
    per_subject = _avg_rounds(rounds)
    total_tokens   = sum(round_tokens)
    avg_tokens_per_query = total_tokens / (K_ROUND_EVAL * len(held_out)) if held_out else 0.0

    # 5. Print + save
    if DATASET == "MATH":
        subjects = MATH_SUBJECTS
    elif DATASET == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    else:
        subjects = MMLU_SUBJECTS
    avg = sum(per_subject[s] for s in subjects) / len(subjects)

    print("\n" + "=" * 70)
    print(f"RESULTS  —  {DATASET}  ({best_agent['name']})  [avg of {K_ROUND_EVAL} rounds]")
    print("=" * 70)
    for s in subjects:
        print(f"  {s:<35s}  {per_subject[s]:.4f}")
    print(f"\n  {'AVERAGE':<35s}  {avg:.4f}  avg_tokens/query: {avg_tokens_per_query:.1f}")
    print("=" * 70)

    save_results(per_subject, best_agent["name"], DATASET, avg_tokens_per_query)


if __name__ == "__main__":
    main()
