"""Frozen deterministic Mind2Web protocol shared by ADAS data and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


MIND2WEB_DATASET_ID = "osunlp/Mind2Web"
MIND2WEB_DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
MIND2WEB_TRAIN_TASKS = 1009
MIND2WEB_TOP_K = 20
MIND2WEB_SEARCH_SEED = 42
MIND2WEB_HELDOUT_SEED = 99
MIND2WEB_SEARCH_TASKS_PER_DOMAIN = 20
MIND2WEB_HELDOUT_TASKS_PER_DOMAIN = 100

MIND2WEB_SCORE_FILE = "scores_all_data.pkl"
MIND2WEB_SCORE_SIZE = 245_190_981
MIND2WEB_SCORE_SHA256 = (
    "884c97cd9ae0544485d21ea39e0d46422aee0291969a7324e56df3a84466dbd7"
)

# ReMAS constructs and samples scenarios in this order.
MIND2WEB_DOMAINS: tuple[tuple[str, str], ...] = (
    ("travel", "Travel"),
    ("shopping", "Shopping"),
    ("entertainment", "Entertainment"),
)
MIND2WEB_DOMAIN_LABELS = dict(MIND2WEB_DOMAINS)


def canonical_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_protocol_splits(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce ReMAS's shared-RNG search and held-out sampling order."""

    records = list(records)
    search_rng = random.Random(MIND2WEB_SEARCH_SEED)
    search: list[dict[str, Any]] = []
    search_ids: set[str] = set()
    for scenario_id, domain in MIND2WEB_DOMAINS:
        pool = [record for record in records if record.get("domain") == domain]
        if len(pool) < MIND2WEB_SEARCH_TASKS_PER_DOMAIN:
            raise ValueError(
                f"Mind2Web domain {domain!r} has {len(pool)} tasks; the full "
                f"protocol requires {MIND2WEB_SEARCH_TASKS_PER_DOMAIN}."
            )
        selected = search_rng.sample(pool, MIND2WEB_SEARCH_TASKS_PER_DOMAIN)
        for record in selected:
            item = dict(record)
            item["scenario_id"] = scenario_id
            search.append(item)
            search_ids.add(str(item["id"]))

    heldout_rng = random.Random(MIND2WEB_HELDOUT_SEED)
    heldout: list[dict[str, Any]] = []
    for scenario_id, domain in MIND2WEB_DOMAINS:
        pool = [
            record
            for record in records
            if record.get("domain") == domain and str(record["id"]) not in search_ids
        ]
        if len(pool) < MIND2WEB_HELDOUT_TASKS_PER_DOMAIN:
            raise ValueError(
                f"Mind2Web domain {domain!r} has only {len(pool)} disjoint "
                "held-out tasks; the full protocol requires "
                f"{MIND2WEB_HELDOUT_TASKS_PER_DOMAIN}."
            )
        selected = heldout_rng.sample(pool, MIND2WEB_HELDOUT_TASKS_PER_DOMAIN)
        for record in selected:
            item = dict(record)
            item["scenario_id"] = scenario_id
            heldout.append(item)
    return search, heldout


def validate_split_records(records: Iterable[dict[str, Any]], split: str) -> None:
    records = list(records)
    expected_per_domain = {
        "search": MIND2WEB_SEARCH_TASKS_PER_DOMAIN,
        "heldout": MIND2WEB_HELDOUT_TASKS_PER_DOMAIN,
    }.get(split)
    if expected_per_domain is None:
        raise ValueError(f"Unknown Mind2Web split: {split!r}")

    expected_order = [scenario_id for scenario_id, _ in MIND2WEB_DOMAINS]
    actual_order: list[str] = []
    seen: set[str] = set()
    counts = {scenario_id: 0 for scenario_id in expected_order}
    for record in records:
        task_id = str(record.get("id", ""))
        scenario_id = str(record.get("scenario_id", ""))
        expected_domain = MIND2WEB_DOMAIN_LABELS.get(scenario_id)
        if not task_id or task_id in seen:
            raise ValueError(
                f"Duplicate or empty Mind2Web task ID in {split}: {task_id!r}"
            )
        if expected_domain is None or record.get("domain") != expected_domain:
            raise ValueError(
                f"Mind2Web task {task_id} has inconsistent scenario/domain values."
            )
        steps = record.get("steps")
        gold = record.get("gold_actions")
        if not isinstance(steps, list) or not steps or len(steps) != len(gold or []):
            raise ValueError(
                f"Mind2Web task {task_id} has invalid action trajectory data."
            )
        if int(record.get("num_steps", -1)) != len(steps):
            raise ValueError(
                f"Mind2Web task {task_id} has an incorrect num_steps value."
            )
        for index, (step, target) in enumerate(zip(steps, gold)):
            if int(step.get("step", -1)) != index or int(
                target.get("step", -1)
            ) != index:
                raise ValueError(
                    f"Mind2Web task {task_id} has non-causal step ordering."
                )
            if len(step.get("previous_actions") or []) != index:
                raise ValueError(
                    f"Mind2Web task {task_id} leaks or loses action history."
                )
        if not actual_order or actual_order[-1] != scenario_id:
            actual_order.append(scenario_id)
        counts[scenario_id] += 1
        seen.add(task_id)

    if actual_order != expected_order:
        raise ValueError(
            f"Mind2Web {split} scenario order drifted: expected {expected_order}, "
            f"received {actual_order}."
        )
    expected_total = expected_per_domain * len(MIND2WEB_DOMAINS)
    if len(records) != expected_total or any(
        count != expected_per_domain for count in counts.values()
    ):
        raise ValueError(
            f"Mind2Web {split} must contain {expected_per_domain} tasks per domain; "
            f"received {counts}."
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
            f"Mind2Web manifest is missing: {manifest_path}. "
            "Run `python dataset/build_mind2web_data.py`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = {
        "dataset": MIND2WEB_DATASET_ID,
        "dataset_revision": MIND2WEB_DATASET_REVISION,
        "top_k": MIND2WEB_TOP_K,
        "search_seed": MIND2WEB_SEARCH_SEED,
        "heldout_seed": MIND2WEB_HELDOUT_SEED,
    }
    for key, expected in expected_config.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Mind2Web manifest {key!r} drifted from {expected!r}."
            )

    expected_ids = manifest.get(f"{split}_ids")
    actual_ids = [str(record["id"]) for record in records]
    if actual_ids != expected_ids:
        raise ValueError(f"Mind2Web {split} task order drifted from its manifest.")
    expected_hashes = manifest.get(f"{split}_record_hashes")
    actual_hashes = {
        str(record["id"]): canonical_record_hash(record) for record in records
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            f"Mind2Web {split} record contents drifted from its manifest."
        )
