"""
Build ADAS-compatible JSONL with the same MATH subset used by MAS and AFlow.

Same 4 subjects, Level 5, seed=42, 10 per subject = 40 total.
Downloads from modelscope (same source as MAS optimizer).
Output: dataset/math_4subjects.jsonl

Run:
    cd C:/Users/cheikh/Desktop/ADAS
    python dataset/build_math_4subjects.py
"""

import json
import random
import shutil
import zipfile
from pathlib import Path

import requests

SUBJECTS = [
    "Prealgebra",
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
LEVEL = "Level 5"
SEED = 42
N_PER_SUBJECT = 30

OUTPUT_PATH = Path(__file__).parent / "math_4subjects.jsonl"
CACHE_DIR = Path(__file__).parent / "_math_cache"
MATH_URL = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"


def download_and_extract():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "MATH.zip"

    if not zip_path.exists():
        print(f"Downloading MATH data...")
        r = requests.get(MATH_URL, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    zip_path.unlink()


def load_test_split():
    test_dir = CACHE_DIR / "MATH" / "test"
    records = []
    for subject_dir in test_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        for jf in subject_dir.glob("*.json"):
            with open(jf, "r", encoding="utf-8") as f:
                ex = json.load(f)
            records.append(ex)
    return records


def main():
    math_root = CACHE_DIR / "MATH"
    if not math_root.exists():
        download_and_extract()

    all_data = load_test_split()
    print(f"Loaded {len(all_data)} test problems total.")

    rng = random.Random(SEED)
    rows = []

    for subject in SUBJECTS:
        filtered = [
            ex for ex in all_data
            if ex.get("type") == subject and ex.get("level") == LEVEL
        ]
        sampled = rng.sample(filtered, min(N_PER_SUBJECT, len(filtered)))
        print(f"  {subject} ({LEVEL}): {len(filtered)} available → {len(sampled)} sampled")
        for ex in sampled:
            rows.append({
                "subject": subject,
                "problem": ex["problem"],
                "solution": ex["solution"],
                "level": ex["level"],
                "type": ex["type"],
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} records → {OUTPUT_PATH}")

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleaned up cache → {CACHE_DIR}")


if __name__ == "__main__":
    main()
