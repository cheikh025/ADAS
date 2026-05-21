"""
Creates dummy run_archive.json files for MATH, MMLU-Pro, and FullStack
from the baseline agents defined in each prompt file.

Run from the ADAS directory:
    python create_dummy_archive.py

This lets you run eval_all_agents.py and eval_best_workflow.py on a machine
that has never run the ADAS search.
"""

import json
import os
import sys
from pathlib import Path

_ADAS_DIR = Path(__file__).parent.resolve()

DATASETS = {
    "mmlu_pro": {
        "prompt_dir": "_mmlu_pro",
        "prompt_module": "mmlu_pro_prompt",
        "archive_path": "results/mmlu_pro_ours_results_run_archive.json",
    },
    "math": {
        "prompt_dir": "_math",
        "prompt_module": "math_prompt",
        "archive_path": "results/math_ours_results_run_archive.json",
    },
    "fullstack": {
        "prompt_dir": "_fullstack",
        "prompt_module": "fullstack_prompt",
        "archive_path": "results/fullstack_ours_results_run_archive.json",
    },
}


def create_archive(dataset_key: str):
    cfg = DATASETS[dataset_key]
    prompt_dir = _ADAS_DIR / cfg["prompt_dir"]
    archive_path = _ADAS_DIR / cfg["archive_path"]

    if archive_path.exists():
        print(f"  [{dataset_key}] archive already exists at {archive_path} — skipping")
        return

    sys.path.insert(0, str(prompt_dir))
    try:
        import importlib
        mod = importlib.import_module(cfg["prompt_module"])
        entries = mod.get_init_archive()
    finally:
        sys.path.pop(0)

    archive = []
    for entry in entries:
        record = dict(entry)
        record["generation"] = "initial"
        archive.append(record)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "w") as f:
        json.dump(archive, f, indent=4)
    print(f"  [{dataset_key}] wrote {len(archive)} baseline agents → {archive_path}")


if __name__ == "__main__":
    print("Creating dummy archives from baseline agents ...\n")
    for key in DATASETS:
        create_archive(key)
    print("\nDone.")
