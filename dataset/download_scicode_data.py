#!/usr/bin/env python3
"""Download and validate SciCode's official numerical target file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "data" / "scicode" / "test_data.h5"
SCICODE_TEST_DATA_FILE_ID = "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW"
SCICODE_EXPECTED_ROOT_GROUPS = 338
SKIPPED_STEPS = {"13.6", "62.1", "76.3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official SciCode test_data.h5."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def expected_paths(dataset_dir: Path) -> list[str]:
    paths: list[str] = []
    for filename in ("scicode_validate.jsonl", "scicode_test.jsonl"):
        source = dataset_dir / filename
        if not source.is_file():
            continue
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                problem: dict[str, Any] = json.loads(line)
                for step in problem.get("sub_steps", []):
                    step_id = str(step.get("step_number"))
                    if step_id in SKIPPED_STEPS:
                        continue
                    for test_index, _ in enumerate(step.get("test_cases") or [], 1):
                        paths.append(f"{step_id}/test{test_index}")
    return paths


def validate_h5(path: Path, dataset_dir: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"SciCode HDF5 file is missing or empty: {path}")
    if not h5py.is_hdf5(path):
        raise ValueError(f"File is not valid HDF5 data: {path}")

    expected = expected_paths(dataset_dir)
    with h5py.File(path, "r") as handle:
        root_groups = len(handle.keys())
        missing = [group_path for group_path in expected if group_path not in handle]
    if root_groups != SCICODE_EXPECTED_ROOT_GROUPS:
        raise ValueError(
            "SciCode HDF5 root-group count is unexpected: "
            f"expected {SCICODE_EXPECTED_ROOT_GROUPS}, received {root_groups}."
        )
    if missing:
        raise ValueError(
            f"SciCode HDF5 is missing {len(missing)} required test groups: "
            + ", ".join(missing[:10])
        )
    return root_groups, len(expected)


def main() -> None:
    args = parse_args()
    target = args.output.expanduser().resolve()
    dataset_dir = ROOT / "dataset"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not args.force:
        groups, checked = validate_h5(target, dataset_dir)
        print(f"SciCode targets are valid: {target}")
        print(f"Root groups: {groups}; verified paths: {checked}")
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown with: pip install -r requirements.txt") from exc

    temporary = target.with_name(target.name + ".download")
    downloaded = gdown.download(
        id=SCICODE_TEST_DATA_FILE_ID,
        output=str(temporary),
        quiet=False,
        resume=True,
    )
    if downloaded is None:
        raise RuntimeError("Google Drive download failed.")

    groups, checked = validate_h5(temporary, dataset_dir)
    os.replace(temporary, target)
    print(f"SciCode targets ready: {target}")
    print(f"Size: {target.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"Root groups: {groups}; verified paths: {checked}")


if __name__ == "__main__":
    main()
