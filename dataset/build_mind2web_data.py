#!/usr/bin/env python3
"""Build ADAS's deterministic ReMAS-compatible Mind2Web proxy splits.

The builder reads the pinned official public train split and official saved
candidate ranks. It writes compact pruned contexts and frozen action targets;
neither ReMAS data nor a ReMAS JSONL is used as an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _mind2web.mind2web_dom import format_pruned_html  # noqa: E402
from _mind2web.mind2web_spec import (  # noqa: E402
    MIND2WEB_DATASET_ID,
    MIND2WEB_DATASET_REVISION,
    MIND2WEB_DOMAINS,
    MIND2WEB_HELDOUT_SEED,
    MIND2WEB_SCORE_FILE,
    MIND2WEB_SCORE_SHA256,
    MIND2WEB_SCORE_SIZE,
    MIND2WEB_SEARCH_SEED,
    MIND2WEB_TOP_K,
    MIND2WEB_TRAIN_TASKS,
    canonical_record_hash,
    select_protocol_splits,
    validate_split_records,
)


DEFAULT_OUTPUT_DIR = ROOT / "dataset"
DEFAULT_DATA_DIR = ROOT / "data" / "mind2web"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the exact ReMAS-compatible Mind2Web splits for ADAS."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MIND2WEB_DATA_DIR", DEFAULT_DATA_DIR)),
        help="Official dataset cache and scores_all_data.pkl directory.",
    )
    parser.add_argument(
        "--verify-remas-artifact",
        type=Path,
        default=None,
        help=(
            "Optional eval_data.jsonl from a completed full ReMAS run. Used "
            "only to verify search IDs/order, never as a data source."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_file_is_valid(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == MIND2WEB_SCORE_SIZE
        and sha256_file(path) == MIND2WEB_SCORE_SHA256
    )


def ensure_score_file(data_dir: Path) -> Path:
    score_path = data_dir / MIND2WEB_SCORE_FILE
    if score_file_is_valid(score_path):
        return score_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install Mind2Web dependencies from requirements.txt."
        ) from exc
    downloaded = Path(
        hf_hub_download(
            repo_id=MIND2WEB_DATASET_ID,
            repo_type="dataset",
            filename=MIND2WEB_SCORE_FILE,
            revision=MIND2WEB_DATASET_REVISION,
            local_dir=data_dir,
            force_download=score_path.exists(),
        )
    )
    if not score_file_is_valid(downloaded):
        raise RuntimeError(
            f"Official Mind2Web rank file failed verification: {downloaded}"
        )
    return downloaded


def load_official_rows(data_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install Mind2Web dependencies from requirements.txt."
        ) from exc
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        MIND2WEB_DATASET_ID,
        split="train",
        cache_dir=str(cache_dir),
        revision=MIND2WEB_DATASET_REVISION,
    )
    if len(dataset) != MIND2WEB_TRAIN_TASKS:
        raise RuntimeError(
            f"Expected {MIND2WEB_TRAIN_TASKS} official train tasks; "
            f"loaded {len(dataset)}."
        )
    return dataset


def node_id(candidate: dict[str, Any]) -> str:
    value = candidate.get("backend_node_id", "")
    return "" if value is None else str(value)


def build_step(
    action: dict[str, Any],
    index: int,
    action_reprs: list[str],
    annotation_id: str,
    candidate_ranks: dict,
    top_k: int = MIND2WEB_TOP_K,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positives = list(action.get("pos_candidates") or [])
    positive_ids = {node_id(candidate) for candidate in positives}
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in positives + list(action.get("neg_candidates") or []):
        candidate_id = node_id(candidate)
        if not candidate_id:
            raise ValueError(
                f"Mind2Web action {action.get('action_uid', index)!r} has a "
                "candidate without backend_node_id."
            )
        candidate_by_id.setdefault(candidate_id, candidate)

    action_uid = str(action.get("action_uid", ""))
    sample_id = f"{annotation_id}_{action_uid}"
    ranking = candidate_ranks[sample_id]
    ranked_options = [
        candidate
        for candidate_id, candidate in candidate_by_id.items()
        if ranking[candidate_id] < top_k
    ]
    ranked_options.sort(
        key=lambda candidate: (ranking[node_id(candidate)], node_id(candidate))
    )
    candidate_ids = [node_id(candidate) for candidate in ranked_options]
    page_context, candidate_reprs = format_pruned_html(
        str(action.get("cleaned_html") or ""), candidate_ids
    )
    random.Random(f"mind2web-ranked-v1:{sample_id}").shuffle(ranked_options)

    lines: list[str] = []
    letter_to_node_id: dict[str, str] = {}
    acceptable_letters: list[str] = []
    for option_index, candidate in enumerate(ranked_options):
        letter = LETTERS[option_index]
        candidate_id = node_id(candidate)
        letter_to_node_id[letter] = candidate_id
        lines.append(f"{letter}) {candidate_reprs[candidate_id]}")
        if candidate_id in positive_ids:
            acceptable_letters.append(letter)

    operation = action.get("operation") or {}
    step = {
        "step": index,
        "action_uid": action_uid,
        "previous_actions": action_reprs[:index],
        "page_context": page_context,
        "candidates": "\n".join(lines),
        "letter_to_node_id": letter_to_node_id,
    }
    gold = {
        "step": index,
        "candidate_letters": list(letter_to_node_id),
        "acceptable_letters": acceptable_letters,
        "acceptable_node_ids": sorted(positive_ids),
        "op": str(operation.get("op", "CLICK")).upper(),
        "value": operation.get("value", "") or "",
    }
    return step, gold


def build_example(
    row: dict[str, Any], candidate_ranks: dict
) -> dict[str, Any] | None:
    annotation_id = str(row.get("annotation_id", ""))
    action_reprs = list(row.get("action_reprs") or [])
    steps: list[dict[str, Any]] = []
    gold_actions: list[dict[str, Any]] = []
    for index, action in enumerate(row.get("actions") or []):
        step, gold = build_step(
            dict(action), index, action_reprs, annotation_id, candidate_ranks
        )
        steps.append(step)
        gold_actions.append(gold)
    if not steps:
        return None
    return {
        "id": annotation_id,
        "task": row.get("confirmed_task", ""),
        "domain": row.get("domain", ""),
        "subdomain": row.get("subdomain", ""),
        "website": row.get("website", ""),
        "num_steps": len(steps),
        "action_reprs": action_reprs,
        "steps": steps,
        "gold_actions": gold_actions,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def verify_remas_artifact(search: list[dict[str, Any]], artifact: Path) -> None:
    rows = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = [(str(row["scenario_id"]), str(row["id"])) for row in search]
    expected = [
        (str(row.get("scenario_id", "")), str(row.get("id", ""))) for row in rows
    ]
    if actual != expected:
        raise ValueError(
            "Generated Mind2Web search IDs/order do not match the supplied "
            "full ReMAS artifact."
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing the complete pinned Mind2Web public-train proxy.")
    print("The first build downloads about 5.93 GB and can require 15 GB temporary space.")
    score_path = ensure_score_file(data_dir)
    rows = load_official_rows(data_dir)

    print(f"Loading verified official candidate ranks: {score_path}")
    with score_path.open("rb") as stream:
        candidate_results = pickle.load(stream)
    candidate_ranks = candidate_results["ranks"]
    del candidate_results

    records: list[dict[str, Any]] = []
    for row in rows:
        record = build_example(dict(row), candidate_ranks)
        if record is not None:
            records.append(record)
    del candidate_ranks
    if len(records) != MIND2WEB_TRAIN_TASKS:
        raise RuntimeError(
            f"Prepared {len(records)} Mind2Web tasks; "
            f"expected {MIND2WEB_TRAIN_TASKS}."
        )

    search, heldout = select_protocol_splits(records)
    validate_split_records(search, "search")
    validate_split_records(heldout, "heldout")
    if args.verify_remas_artifact is not None:
        verify_remas_artifact(
            search, args.verify_remas_artifact.expanduser().resolve()
        )

    search_path = output_dir / "mind2web_validate.jsonl"
    heldout_path = output_dir / "mind2web_test.jsonl"
    write_jsonl(search_path, search)
    write_jsonl(heldout_path, heldout)

    manifest = {
        "dataset": MIND2WEB_DATASET_ID,
        "dataset_revision": MIND2WEB_DATASET_REVISION,
        "source_train_tasks": MIND2WEB_TRAIN_TASKS,
        "score_file": MIND2WEB_SCORE_FILE,
        "score_file_size": MIND2WEB_SCORE_SIZE,
        "score_file_sha256": MIND2WEB_SCORE_SHA256,
        "top_k": MIND2WEB_TOP_K,
        "search_seed": MIND2WEB_SEARCH_SEED,
        "heldout_seed": MIND2WEB_HELDOUT_SEED,
        "domains": [domain for _, domain in MIND2WEB_DOMAINS],
        "search_ids": [str(record["id"]) for record in search],
        "heldout_ids": [str(record["id"]) for record in heldout],
        "search_record_hashes": {
            str(record["id"]): canonical_record_hash(record) for record in search
        },
        "heldout_record_hashes": {
            str(record["id"]): canonical_record_hash(record) for record in heldout
        },
        "search_tasks": len(search),
        "search_actions": sum(int(record["num_steps"]) for record in search),
        "heldout_tasks": len(heldout),
        "heldout_actions": sum(int(record["num_steps"]) for record in heldout),
        "domain_pool_counts": {
            domain: sum(record.get("domain") == domain for record in records)
            for _, domain in MIND2WEB_DOMAINS
        },
    }
    manifest_path = output_dir / "mind2web_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Search:   {len(search)} tasks -> {search_path}")
    print(f"Held-out: {len(heldout)} tasks -> {heldout_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
