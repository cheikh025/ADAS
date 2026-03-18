"""
Build ADAS-compatible CSV with the same MMLU subset used by MAS and AFlow.

Same 4 subjects, seed=42, 10 per subject = 40 total rows.
Output: dataset/mmlu_4subjects.csv  (columns: Question, A, B, C, D, Answer, Subject)

Run:
    cd C:/Users/cheikh/Desktop/ADAS
    python dataset/build_mmlu_4subjects.py
"""

import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SUBJECTS = [
    "international_law",
    "anatomy",
    "business_ethics",
    "college_chemistry",
    "moral_scenarios",
    "econometrics"
]

SEED = 42
N_PER_SUBJECT = 20
OUTPUT_PATH = Path(__file__).parent / "mmlu_4subjects.csv"

LETTER = ["A", "B", "C", "D"]


def main():
    rng = random.Random(SEED)
    rows = []

    for subject in SUBJECTS:
        print(f"Loading {subject}...")
        dataset = load_dataset("cais/mmlu", subject, split="test")
        data = list(dataset)
        sampled = rng.sample(data, min(N_PER_SUBJECT, len(data)))
        print(f"  {len(data)} available → {len(sampled)} sampled")

        for ex in sampled:
            rows.append({
                "Question": ex["question"],
                "A": ex["choices"][0],
                "B": ex["choices"][1],
                "C": ex["choices"][2],
                "D": ex["choices"][3],
                "Answer": LETTER[ex["answer"]],
                "Subject": subject,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=True)
    print(f"\nWrote {len(df)} rows → {OUTPUT_PATH}")
    print(df["Subject"].value_counts().to_string())


if __name__ == "__main__":
    main()
