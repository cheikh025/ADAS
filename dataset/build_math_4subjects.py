"""
Build ADAS-compatible JSONL with the same MATH subset used by MAS and AFlow.

Subjects: Algebra, Geometry
Level: Level 5
Seed: 42
N/subject: 20
Split: test

Output: dataset/math_4subjects.jsonl

Run:
    cd C:/Users/cheikh/Desktop/ADAS
    python dataset/build_math_4subjects.py
"""

import json
import random
import zipfile
from pathlib import Path

import requests

SUBJECTS = [
    "Algebra",
    "Geometry",
]

LEVEL = "Level 5"
SEED = 42
N_PER_SUBJECT = 20

OUTPUT_PATH = Path(__file__).parent / "math_4subjects.jsonl"
CACHE_DIR = Path(__file__).parent.parent / "data" / "math_hf_cache"
MATH_URL = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"


def download_and_extract() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "MATH.zip"

    if not zip_path.exists():
        print(f"Downloading MATH data from {MATH_URL} ...")
        response = requests.get(MATH_URL, stream=True)
        response.raise_for_status()

        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Extracting MATH.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)

    zip_path.unlink(missing_ok=True)


def load_test_split() -> list[dict]:
    test_dir = CACHE_DIR / "MATH" / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Expected MATH test folder not found: {test_dir}")

    records = []

    for subject_dir in sorted(test_dir.iterdir(), key=lambda p: p.name):
        if not subject_dir.is_dir():
            continue

        for json_file in sorted(subject_dir.glob("*.json"), key=lambda p: p.name):
            with json_file.open("r", encoding="utf-8") as f:
                records.append(json.load(f))

    return records


def main() -> None:
    test_dir = CACHE_DIR / "MATH" / "test"

    if not test_dir.exists():
        download_and_extract()

    all_data = load_test_split()
    print(f"Loaded {len(all_data)} test problems total.")

    rng = random.Random(SEED)
    rows = []

    for subject in SUBJECTS:
        filtered = sorted(
            [
                ex for ex in all_data
                if ex.get("type") == subject and ex.get("level") == LEVEL
            ],
            key=lambda ex: ex["problem"],
        )

        sampled = rng.sample(filtered, min(N_PER_SUBJECT, len(filtered)))

        print(
            f"  {subject} ({LEVEL}): "
            f"{len(filtered)} available -> {len(sampled)} sampled"
        )

        for ex in sampled:
            rows.append({
                "subject": subject,
                "problem": ex["problem"],
                "solution": ex["solution"],
                "level": ex["level"],
                "type": ex["type"],
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} records -> {OUTPUT_PATH}")
    print(f"Full test data kept at {CACHE_DIR}")


if __name__ == "__main__":
    main()