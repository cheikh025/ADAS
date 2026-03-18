"""
Build ADAS-compatible JSONL with the same FullStackBench subset used by MAS.

Categories: Advanced Programming, Operating System, Machine Learning
Difficulty: hard, Locale: en, Seed: 42, N_PER_CATEGORY: 3 = 9 total.

Output: dataset/fullstack_subset.jsonl
Each row: {id, content, category, difficulty, programming_language, raw_example}
raw_example is the full HF row needed by SandboxFusion as provided_data.

Run:
    cd C:/Users/cheikh/Desktop/ADAS
    python dataset/build_fullstack_subset.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

CATEGORIES = ["Advanced Programming", "Operating System", "Machine Learning"]
DIFFICULTY = "hard"
LOCALE = "en"
SEED = 42
N_PER_CATEGORY = 3

OUTPUT_PATH = Path(__file__).parent / "fullstack_subset.jsonl"


def main():
    print("Loading FullStackBench from HuggingFace (ByteDance/FullStackBench)...")
    dataset = load_dataset("ByteDance/FullStackBench", LOCALE, split="test")
    data = list(dataset)
    print(f"Total examples: {len(data)}")

    rng = random.Random(SEED)
    rows = []

    for category in CATEGORIES:
        filtered = [
            ex for ex in data
            if ex["labels"].get("category") == category
            and ex["labels"].get("difficulty") == DIFFICULTY
        ]
        sampled = rng.sample(filtered, min(N_PER_CATEGORY, len(filtered)))
        print(f"  {category}: {len(filtered)} hard examples → sampled {len(sampled)}")

        for ex in sampled:
            rows.append({
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                "raw_example": dict(ex),  # full row for SandboxFusion provided_data
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
