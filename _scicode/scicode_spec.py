"""Frozen ReMAS-compatible SciCode data protocol for ADAS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCICODE_DATASET = "SciCode1/SciCode"
SCICODE_DATASET_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
SCICODE_REFERENCE_COMMIT = "e3158ea011d4235245a547460d3688d7ccbf9900"
SCICODE_SPLIT_SEED = 42
SCICODE_HELDOUT_SEED = 99

SCICODE_FIELDS: dict[str, tuple[str, ...]] = {
    "mathematics": (
        "1", "3", "4", "5", "9", "18", "24", "29", "31", "40", "54",
        "63", "74", "78",
    ),
    "physics": (
        "2", "6", "7", "8", "11", "13", "14", "15", "17", "19", "20",
        "22", "23", "28", "32", "33", "37", "38", "43", "45", "48",
        "49", "50", "52", "57", "58", "59", "61", "62", "65", "67",
        "69", "70", "71", "72", "73", "75",
    ),
    "material_science": (
        "21", "27", "34", "35", "36", "39", "42", "47", "51", "64",
        "77", "79", "80",
    ),
}

SCICODE_FIELD_LABELS = {
    "mathematics": "Mathematics",
    "physics": "Physics",
    "material_science": "Material Science",
}

SCICODE_SEARCH_IDS: dict[str, tuple[str, ...]] = {
    "mathematics": ("40", "31", "18"),
    "physics": ("19", "73", "67"),
    "material_science": ("36", "80", "35"),
}

SCICODE_HELDOUT_IDS: dict[str, tuple[str, ...]] = {
    "mathematics": ("24", "5", "3", "78", "54", "9", "74", "29", "1", "63"),
    "physics": ("6", "37", "8", "70", "48", "59", "72", "58", "45", "49"),
    "material_science": ("47", "64", "21", "77", "42", "39", "34", "27", "79", "51"),
}

SCICODE_SKIPPED_STEPS = frozenset({"13.6", "62.1", "76.3"})
SCICODE_REFERENCE_URL = (
    "https://raw.githubusercontent.com/scicode-bench/SciCode/"
    + SCICODE_REFERENCE_COMMIT
    + "/eval/data/{step_id}.txt"
)


def ordered_pairs(split: str) -> list[tuple[str, str]]:
    if split == "search":
        mapping = SCICODE_SEARCH_IDS
    elif split == "heldout":
        mapping = SCICODE_HELDOUT_IDS
    else:
        raise ValueError(f"Unknown SciCode split: {split!r}")
    return [(field, problem_id) for field, ids in mapping.items() for problem_id in ids]


def canonical_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_split_records(records: Iterable[dict[str, Any]], split: str) -> None:
    records = list(records)
    actual = [
        (str(record.get("field", "")), str(record.get("problem_id", "")))
        for record in records
    ]
    expected = ordered_pairs(split)
    if actual != expected:
        raise ValueError(
            f"SciCode {split} records do not match the frozen ReMAS order. "
            f"Expected {expected}; received {actual}."
        )

    seen: set[str] = set()
    for record in records:
        problem_id = str(record["problem_id"])
        field = str(record["field"])
        if problem_id in seen:
            raise ValueError(f"Duplicate SciCode problem ID in {split}: {problem_id}")
        if problem_id not in SCICODE_FIELDS.get(field, ()):
            raise ValueError(
                f"SciCode problem {problem_id} does not belong to field {field!r}."
            )
        steps = record.get("sub_steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"SciCode problem {problem_id} has no sub_steps.")
        seen.add(problem_id)

    other = "heldout" if split == "search" else "search"
    overlap = seen.intersection(problem_id for _, problem_id in ordered_pairs(other))
    if overlap:
        raise ValueError(
            f"SciCode search/held-out overlap detected: {sorted(overlap, key=int)}"
        )


def validate_manifest_records(
    records: Iterable[dict[str, Any]],
    split: str,
    manifest_path: str | Path,
) -> None:
    records = list(records)
    validate_split_records(records, split)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SciCode manifest is missing: {manifest_path}. "
            "Run `python dataset/build_scicode_data.py`."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != SCICODE_DATASET:
        raise ValueError("SciCode manifest names an unexpected source dataset.")
    if manifest.get("dataset_revision") != SCICODE_DATASET_REVISION:
        raise ValueError("SciCode manifest revision drifted from the pinned revision.")

    expected_hashes = manifest.get(f"{split}_record_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError(f"SciCode manifest has no {split}_record_hashes mapping.")
    actual_hashes = {
        str(record["problem_id"]): canonical_record_hash(record) for record in records
    }
    if actual_hashes != expected_hashes:
        mismatched = sorted(
            {
                *set(actual_hashes).symmetric_difference(expected_hashes),
                *(
                    problem_id
                    for problem_id in set(actual_hashes).intersection(expected_hashes)
                    if actual_hashes[problem_id] != expected_hashes[problem_id]
                ),
            },
            key=int,
        )
        raise ValueError(
            f"SciCode {split} record contents drifted from the manifest: {mismatched}."
        )
