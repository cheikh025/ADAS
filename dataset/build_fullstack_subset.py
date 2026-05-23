"""
Build ADAS-compatible JSONL with the same FullStackBench subset policy used by AFlow.

Categories: Advanced Programming, Scientific Computing, Data Analysis,
            Desktop and Web Development
Locale: en, Seed: 42, N_HARD: 10, N_MEDIUM: 10 = 20 per category = 80 total.
Stratified by difficulty: 10 hard + 10 medium, no easy.

Output: dataset/fullstack_subset.jsonl
Each row: {id, content, category, difficulty, programming_language, raw_example}
raw_example is the full HF row needed by SandboxFusion as provided_data.

Run:
    cd C:/Users/cheikh/Desktop/baseline/ADAS
    python dataset/build_fullstack_subset.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

CATEGORIES = [
    "Advanced Programming",
    "Scientific Computing",
    "Data Analysis",
    "Desktop and Web Development",
]
LOCALE = "en"
SEED = 42
N_HARD = 10
N_MEDIUM = 10

OUTPUT_PATH = Path(__file__).parent / "fullstack_subset.jsonl"


def main():
    print("Loading FullStackBench from HuggingFace (ByteDance/FullStackBench)...")
    dataset = load_dataset("ByteDance/FullStackBench", LOCALE, split="test")
    data = list(dataset)
    print(f"Total examples: {len(data)}")

    rows = []

    for category in CATEGORIES:
        rng = random.Random(SEED)
        filtered = [ex for ex in data if ex["labels"].get("category") == category]
        hard   = [ex for ex in filtered if ex["labels"].get("difficulty") == "hard"]
        medium = [ex for ex in filtered if ex["labels"].get("difficulty") == "medium"]

        sampled_hard   = rng.sample(hard,   min(N_HARD,   len(hard)))
        sampled_medium = rng.sample(medium, min(N_MEDIUM, len(medium)))
        sampled = sampled_hard + sampled_medium

        print(f"  {category}: {len(hard)} hard, {len(medium)} medium → sampled {len(sampled_hard)}h + {len(sampled_medium)}m")

        for ex in sampled:
            rows.append({
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                "raw_example": dict(ex),
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
