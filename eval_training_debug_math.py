"""
ADAS MATH — Debug Evaluation on Training Data

Runs the best ADAS agent on all 120 training problems (30 per subject x 4),
which is what ADAS should use during training (matching AFlow/MAS).

Run from any directory:
    python Baseline/ADAS/eval_training_debug_math.py
"""

import importlib.util
import json
import os
import re
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

# ─── paths ────────────────────────────────────────────────────────────────────
_ADAS_DIR = Path(__file__).parent.resolve()

MATH_TRAIN_JSONL = _ADAS_DIR / "dataset/math_4subjects.jsonl"
MATH_ARCHIVE     = _ADAS_DIR / "results/math_ours_results_run_archive.json"

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MAX_WORKERS = 10   # parallel threads
# ──────────────────────────────────────────────────────────────────────────────

MATH_SUBJECTS = [
    "Prealgebra",
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])


# ─────────────────────────────────────────────────────────────────────────────
# Best-agent discovery
# ─────────────────────────────────────────────────────────────────────────────

def _parse_median(fitness_str: str) -> float:
    m = re.search(r"Median:\s*([\d.]+)%", fitness_str)
    return float(m.group(1)) if m else 0.0


def find_best_agent(archive_path: Path) -> dict:
    with open(archive_path) as f:
        archive = json.load(f)
    candidates = [a for a in archive if "fitness" in a]
    best = max(candidates, key=lambda a: _parse_median(a["fitness"]))
    print(f"Best agent : gen={best['generation']}, name={best['name']}")
    print(f"  Fitness  : {best['fitness']}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Module loading (mirrors how search_ours.py sets up globals)
# ─────────────────────────────────────────────────────────────────────────────

def load_search_module():
    module_dir = _ADAS_DIR / "_math"
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "search_ours_math_debug",
            module_dir / "search_ours.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    # Patch API globals exactly as eval_best_workflow.py does
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    import openai
    mod.client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    mod.MODEL              = "openai/gpt-oss-20b"
    mod.EVAL_TEMPERATURE   = 1.0
    mod.SEARCH_TEMPERATURE = 0.8
    mod.MAX_TOKENS         = 64768
    mod.PROVIDER_ROUTING   = {
        "order": ["Google Vertex", "Together", "Groq"],
        "allow_fallbacks": True,
    }
    mod.SEARCHING_MODE = True
    return mod


def setup_forward(mod, code_str: str):
    namespace = {}
    exec(code_str, vars(mod), namespace)
    names = [k for k, v in namespace.items() if callable(v)]
    if not names:
        raise ValueError("No callable found in agent code.")
    func = namespace[names[0]]
    setattr(mod.AgentSystem, "forward", func)
    return mod.AgentSystem()


# ─────────────────────────────────────────────────────────────────────────────
# Training data — all 120 problems (30 per subject x 4)
# ─────────────────────────────────────────────────────────────────────────────

def load_training_examples() -> List[dict]:
    """Load all 120 problems from math_4subjects.jsonl (30 per subject x 4)."""
    examples = []
    with open(MATH_TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Training examples loaded: {len(examples)}")
    from collections import Counter
    for s, c in Counter(ex["subject"] for ex in examples).items():
        print(f"  {s}: {c}")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(agent_system, examples: List[dict], score_math) -> Dict[str, float]:
    questions  = [ex["problem"]  for ex in examples]
    solutions  = [ex["solution"] for ex in examples]
    subjects   = [ex["subject"]  for ex in examples]
    task_queue = [Info("task", "User", q, -1) for q in questions]

    workers = min(len(examples), MAX_WORKERS)
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=workers) as exe:
        results = list(tqdm(
            exe.map(agent_system.forward, task_queue),
            total=len(task_queue),
            desc="MATH training-data eval",
        ))

    print("\n--- PER-PROBLEM DEBUG ---")
    buckets: Dict[str, List[int]] = {s: [] for s in MATH_SUBJECTS}
    all_correct = []
    for idx, res in enumerate(results):
        subject  = subjects[idx]
        solution = solutions[idx]
        try:
            raw_type = type(res).__name__
            pred     = res.content if isinstance(res, Info) else str(res)
            correct  = int(score_math(solution, pred))
            # Truncate for readability
            pred_snippet = pred[:120].replace("\n", " ")
        except Exception as e:
            pred     = f"[ERROR: {e}]"
            pred_snippet = pred
            correct  = 0
            raw_type = type(res).__name__

        status = "OK " if correct else "WRONG"
        print(f"  [{idx:02d}] {subject:<30s} {status}  type={raw_type}")
        if not correct:
            # Show ground truth answer and prediction snippet for wrong answers
            gt_snippet  = solution[:80].replace("\n", " ")
            print(f"       GT  : {gt_snippet}")
            print(f"       PRED: {pred_snippet}")

        buckets[subject].append(correct)
        all_correct.append(correct)

    print("--- END DEBUG ---\n")

    per_subject = {s: (sum(v) / len(v) if v else float("nan"))
                   for s, v in buckets.items()}
    overall = sum(all_correct) / len(all_correct) if all_correct else 0.0
    from _math.utils import bootstrap_confidence_interval
    bootstrap_str = bootstrap_confidence_interval(all_correct)
    return per_subject, overall, bootstrap_str


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(per_subject: dict, overall: float, agent_name: str,
                 reported_median: float, bootstrap_str: str = ""):
    out_dir = _ADAS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = out_dir / f"training_debug_math_{timestamp}.txt"

    subjects_with_data = [s for s in MATH_SUBJECTS if s in per_subject]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ADAS MATH — DEBUG EVAL ON TRAINING DATA\n")
        f.write("=" * 70 + "\n")
        f.write(f"Agent:              {agent_name}\n")
        f.write(f"Training examples:  120  (30 per subject x 4)\n")
        f.write(f"Reported Median:    {reported_median:.1f}%\n")
        f.write(f"Date:               {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        for s in subjects_with_data:
            f.write(f"  {s:<35s}  {per_subject[s]:.4f}\n")
        f.write(f"\n  {'OVERALL (all 120)':<35s}  {overall:.4f}  "
                f"({overall * 100:.1f}%)  <-- compare to Median above\n")
        if bootstrap_str:
            f.write(f"  Bootstrap (same as search fitness): {bootstrap_str}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {out_file}")
    return out_file


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(_ADAS_DIR)

    print("=" * 70)
    print("ADAS MATH — DEBUG EVAL ON TRAINING DATA")
    print("Training set: 120 problems  (30 per subject x 4)")
    print("=" * 70)

    # 1. Best agent
    best_agent     = find_best_agent(MATH_ARCHIVE)
    reported_median = _parse_median(best_agent["fitness"])

    # 2. Load module + set up agent
    print("\nLoading search module and patching globals …")
    mod          = load_search_module()
    agent_system = setup_forward(mod, best_agent["code"])
    score_math   = mod.score_math
    print("Agent ready.")

    # 3. Reconstruct training examples
    print("\nReconstructing training examples …")
    training = load_training_examples()

    # 4. Evaluate
    print(f"\nRunning evaluation on {len(training)} training problems "
          f"(max_workers={MAX_WORKERS}) …")
    per_subject, overall, bootstrap_str = run_eval(agent_system, training, score_math)

    # 5. Print summary
    print("\n" + "=" * 70)
    print(f"RESULTS  —  MATH training data  ({best_agent['name']})")
    print("=" * 70)
    for s in MATH_SUBJECTS:
        score = per_subject.get(s, float("nan"))
        print(f"  {s:<35s}  {score:.4f}")
    print(f"\n  {'OVERALL (all 120)':<35s}  {overall:.4f}  ({overall * 100:.1f}%)  <-- actual")
    print(f"  {'Bootstrap (same as search fitness)':<35s}  {bootstrap_str}")
    print(f"  {'Reported training Median':<35s}  {reported_median:.1f}%  <-- during search")
    print("=" * 70)

    save_results(per_subject, overall, best_agent["name"], reported_median, bootstrap_str)


if __name__ == "__main__":
    main()
