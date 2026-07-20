#!/usr/bin/env python3
"""Reconstruct ADAS's exact ReMAS-compatible SciCode records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _scicode.scicode_spec import (  # noqa: E402
    SCICODE_DATASET,
    SCICODE_DATASET_REVISION,
    SCICODE_HELDOUT_IDS,
    SCICODE_HELDOUT_SEED,
    SCICODE_REFERENCE_URL,
    SCICODE_SEARCH_IDS,
    SCICODE_SKIPPED_STEPS,
    SCICODE_SPLIT_SEED,
    canonical_record_hash,
    ordered_pairs,
    validate_split_records,
)


DEFAULT_OUTPUT_DIR = ROOT / "dataset"
DEFAULT_ASSET_DIR = ROOT / "data" / "scicode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact ReMAS-compatible SciCode records for ADAS."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--skip-reference-download",
        action="store_true",
        help="Do not prefetch official implementations for globally skipped steps.",
    )
    return parser.parse_args()


def load_official_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install SciCode dependencies with: pip install -r requirements.txt"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for official_split in ("validation", "test"):
        dataset = load_dataset(
            SCICODE_DATASET,
            split=official_split,
            revision=SCICODE_DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
        for raw in dataset:
            row = dict(raw)
            problem_id = str(row["problem_id"])
            if problem_id in rows:
                raise RuntimeError(
                    f"Duplicate SciCode problem ID across official splits: {problem_id}"
                )
            row["problem_id"] = problem_id
            row["official_split"] = official_split
            rows[problem_id] = row
    return rows


def materialize(
    rows: dict[str, dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for field, problem_id in ordered_pairs(split):
        if problem_id not in rows:
            raise RuntimeError(
                f"Pinned SciCode dataset is missing required {split} problem {problem_id}."
            )
        record = dict(rows[problem_id])
        record["field"] = field
        records.append(record)
    validate_split_records(records, split)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def prefetch_reference_steps(asset_dir: Path) -> dict[str, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests from requirements.txt.") from exc

    reference_dir = asset_dir / "reference_steps"
    reference_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for step_id in sorted(SCICODE_SKIPPED_STEPS):
        target = reference_dir / f"{step_id}.txt"
        if not target.is_file():
            response = requests.get(
                SCICODE_REFERENCE_URL.format(step_id=step_id), timeout=60
            )
            response.raise_for_status()
            target.write_text(response.text, encoding="utf-8")
        hashes[step_id] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    asset_dir = args.asset_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows = load_official_rows(asset_dir / "hf_cache")
    search = materialize(rows, "search")
    heldout = materialize(rows, "heldout")

    search_path = output_dir / "scicode_validate.jsonl"
    heldout_path = output_dir / "scicode_test.jsonl"
    write_jsonl(search_path, search)
    write_jsonl(heldout_path, heldout)

    reference_hashes = (
        {} if args.skip_reference_download else prefetch_reference_steps(asset_dir)
    )
    manifest = {
        "dataset": SCICODE_DATASET,
        "dataset_revision": SCICODE_DATASET_REVISION,
        "split_seed": SCICODE_SPLIT_SEED,
        "heldout_seed": SCICODE_HELDOUT_SEED,
        "search_ids": {key: list(value) for key, value in SCICODE_SEARCH_IDS.items()},
        "heldout_ids": {key: list(value) for key, value in SCICODE_HELDOUT_IDS.items()},
        "search_record_hashes": {
            str(record["problem_id"]): canonical_record_hash(record)
            for record in search
        },
        "heldout_record_hashes": {
            str(record["problem_id"]): canonical_record_hash(record)
            for record in heldout
        },
        "reference_step_hashes": reference_hashes,
    }
    manifest_path = output_dir / "scicode_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Search:   {len(search)} exact main problems -> {search_path}")
    print(f"Held-out: {len(heldout)} exact main problems -> {heldout_path}")
    print(f"Manifest: {manifest_path}")
    print("Search IDs:", [problem_id for _, problem_id in ordered_pairs("search")])
    print("Held-out IDs:", [problem_id for _, problem_id in ordered_pairs("heldout")])


if __name__ == "__main__":
    main()
