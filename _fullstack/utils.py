import json
import os
import random
import string

import numpy as np
import requests


DATASET_NAME = "full_stack_bench_en"


def random_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


def load_fullstack_examples(file_path: str) -> list[dict]:
    """Load JSONL. Each row: {id, content, category, difficulty, programming_language, raw_example}."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_task(example: dict) -> str:
    """Return the same task input used by AFlow and RobustMAS: raw content only."""
    return example["content"]


import threading

_debug_lock = threading.Lock()

def score_fullstack(prediction: str, raw_example: dict,
                    sandbox_endpoint: str = None,
                    compile_timeout: int = 120,
                    run_timeout: int = 120) -> float:
    """
    Submit raw model completion to SandboxFusion. Returns pass_rate (0.0–1.0).
    SandboxFusion handles code extraction — pass the raw string, do NOT pre-extract.
    """
    endpoint = sandbox_endpoint or os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080")
    # Language is nested inside 'labels' in the raw HF example
    labels = raw_example.get("labels", {})
    language = labels.get("programming_language", "python")
    try:
        response = requests.post(
            f"{endpoint}/submit",
            json={
                "dataset": DATASET_NAME,
                "id": raw_example["id"],
                "completion": prediction,
                "config": {
                    "dataset_type": "AutoEvalDataset",
                    "compile_timeout": compile_timeout,
                    "run_timeout": run_timeout,
                    "provided_data": raw_example,
                },
            },
            timeout=compile_timeout + run_timeout + 10,
        )
        
        # --- DEBUG LOGGING (Moved before raise_for_status) ---
        try:
            result = response.json()
        except:
            result = {"text": response.text}

        with _debug_lock:
            with open("fullstack_debug.jsonl", "a", encoding="utf-8") as f:
                debug_entry = {
                    "id": raw_example["id"],
                    "language": language,
                    "status_code": response.status_code,
                    "sent_to_sandbox": prediction,
                    "sandbox_response": result
                }
                f.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")
        # --------------------------------------------------

        response.raise_for_status()
        accepted = result.get("accepted", False)
        tests = result.get("tests", [])
        if tests:
            passed = sum(1 for t in tests if t.get("passed", False))
            return passed / len(tests)
        return 1.0 if accepted else 0.0
    except Exception as e:
        print(f"SandboxFusion error: {e}")
        return 0.0


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    data = np.array(data)
    bootstrap_means = np.mean(
        np.random.choice(data, size=(num_bootstrap_samples, len(data)), replace=True),
        axis=1
    )
    lower = (1.0 - confidence_level) / 2.0
    ci_lower = np.percentile(bootstrap_means, lower * 100) * 100
    ci_upper = np.percentile(bootstrap_means, (1 - lower) * 100) * 100
    median = np.median(bootstrap_means) * 100
    return f"95% Bootstrap Confidence Interval: ({ci_lower:.1f}%, {ci_upper:.1f}%), Median: {median:.1f}%"
