"""
Build ADAS-compatible CSV for MMLU-Pro with 4 selected categories.

Categories: law, history, philosophy, engineering
Each with 20 questions sampled from the test split.
Output: dataset/mmlu_pro_4categories.csv
Columns: Question, A, B, C, D, E, F, G, H, I, J, Answer, Category

Run:
    cd C:/Users/cheikh/Desktop/baseline/ADAS
    python dataset/build_mmlu_pro_4categories.py
"""

import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

CATEGORIES = [
    "law",
    "history",
    "philosophy",
    "engineering",
]

SEED = 42
N_PER_CATEGORY = 20
OUTPUT_PATH = Path(__file__).parent / "mmlu_pro_4categories.csv"

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def main():
    rng = random.Random(SEED)

    print("Loading TIGER-Lab/MMLU-Pro from HuggingFace...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    all_data = list(dataset)
    print(f"  Total test examples: {len(all_data)}")

    rows = []
    for category in CATEGORIES:
        cat_data = [ex for ex in all_data if ex["category"] == category]
        sampled = rng.sample(cat_data, min(N_PER_CATEGORY, len(cat_data)))
        print(f"  {category}: {len(cat_data)} available → {len(sampled)} sampled")

        for ex in sampled:
            options = list(ex["options"])  # up to 10 options
            # Pad to exactly 10 options if fewer
            while len(options) < 10:
                options.append("")
            row = {
                "Question": ex["question"],
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
                "E": options[4],
                "F": options[5],
                "G": options[6],
                "H": options[7],
                "I": options[8],
                "J": options[9],
                "Answer": str(ex["answer"]).upper(),
                "Category": category,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=True)
    print(f"\nWrote {len(df)} rows → {OUTPUT_PATH}")
    print(df["Category"].value_counts().to_string())


if __name__ == "__main__":
    main()
