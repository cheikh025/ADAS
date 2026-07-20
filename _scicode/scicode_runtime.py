"""Shared SciCode trajectory runtime for ADAS search and held-out evaluation."""

from __future__ import annotations

import importlib.util
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

try:
    from .scicode_evaluator import SciCodeEvaluator
    from .scicode_spec import (
        SCICODE_FIELD_LABELS,
        SCICODE_REFERENCE_URL,
        SCICODE_SKIPPED_STEPS,
        validate_manifest_records,
    )
except ImportError:
    from scicode_evaluator import SciCodeEvaluator
    from scicode_spec import (
        SCICODE_FIELD_LABELS,
        SCICODE_REFERENCE_URL,
        SCICODE_SKIPPED_STEPS,
        validate_manifest_records,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5_PATH = ROOT / "data" / "scicode" / "test_data.h5"
DEFAULT_REFERENCE_DIR = ROOT / "data" / "scicode" / "reference_steps"

WITH_BACKGROUND_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""


def uses_local_thinking_controls(
    base_url: str,
    reasoning_backend: str = "auto",
) -> bool:
    """Resolve whether to use the ReMAS local-endpoint request schema."""

    if reasoning_backend not in {"auto", "openrouter", "local"}:
        raise ValueError(f"Unsupported reasoning backend: {reasoning_backend!r}")
    if reasoning_backend != "auto":
        return reasoning_backend == "local"
    url = str(base_url).lower()
    return any(
        marker in url
        for marker in (
            "ai4news.rnd.huawei.com",
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
        )
    )


def build_reasoning_extra_body(
    base_url: str,
    effort: str | None,
    *,
    thinking: bool | None,
    reasoning_backend: str = "auto",
) -> dict[str, Any]:
    """Return endpoint-specific reasoning controls matching ReMAS."""

    if not uses_local_thinking_controls(base_url, reasoning_backend):
        return {"reasoning": {"effort": effort}} if effort is not None else {}

    extra: dict[str, Any] = {}
    if effort not in (None, "none"):
        extra["reasoning_effort"] = effort
    if thinking is not None:
        extra["chat_template_kwargs"] = {"thinking": thinking}
    return extra


def extract_python_script(response: str) -> str:
    """Match ReMAS's first-code-block extraction and import stripping."""

    if "```" in response:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0]
        else:
            code = response.split("```", 1)[1].split("```", 1)[0]
    else:
        code = response
    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        code,
        flags=re.MULTILINE,
    ).strip()


def extract_agent_text(output: Any) -> str:
    """Extract a generated code string from common ADAS forward return shapes."""

    if output is None:
        return ""
    if hasattr(output, "content"):
        return str(output.content)
    if isinstance(output, dict):
        for key in ("code", "response", "answer", "solution"):
            if key in output:
                return extract_agent_text(output[key])
        return str(output)
    if isinstance(output, (list, tuple)):
        named_code = [
            item
            for item in output
            if str(getattr(item, "name", "")).lower() in {"code", "solution"}
        ]
        if named_code:
            return extract_agent_text(named_code[-1])
        return extract_agent_text(output[-1]) if output else ""
    return str(output)


def _step_text(step: dict[str, Any]) -> str:
    return (
        str(step.get("step_description_prompt") or "")
        + "\n"
        + str(step.get("step_background") or "")
    )


def build_step_prompt(
    problem: dict[str, Any],
    step_index: int,
    previous_code: list[str],
) -> tuple[str, str]:
    steps = problem["sub_steps"]
    if step_index < 0 or step_index >= len(steps):
        raise IndexError(f"Invalid SciCode step index: {step_index}")
    if len(previous_code) != step_index:
        raise ValueError(
            f"Step {step_index + 1} needs {step_index} previous code blocks; "
            f"received {len(previous_code)}."
        )

    previous_sections: list[str] = []
    for index in range(step_index):
        previous_sections.extend(
            [_step_text(steps[index]), previous_code[index], "------"]
        )
    problem_steps = "\n\n".join(previous_sections[:-1])

    current = steps[step_index]
    function_spec = (
        f"{current.get('function_header', '')}\n\n"
        f"{current.get('return_line', '')}"
    )
    next_step = "\n\n".join([_step_text(current), function_spec])
    dependencies = str(problem.get("required_dependencies") or "")
    prompt = WITH_BACKGROUND_TEMPLATE.format(
        problem_steps_str=problem_steps,
        next_step_str=next_step,
        dependencies=dependencies,
    )
    previous_code_text = "\n".join(previous_code)
    prefix = f"{dependencies}\n{previous_code_text}\n"
    return prompt, prefix


def load_scicode_records(path: str | Path, split: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SciCode {split} data is missing: {path}. "
            "Run `python dataset/build_scicode_data.py`."
        )
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    validate_manifest_records(records, split, path.with_name("scicode_manifest.json"))
    return records


def score_weight_for_problem(problem: dict[str, Any]) -> int:
    return sum(
        str(step.get("step_number")) not in SCICODE_SKIPPED_STEPS
        for step in problem.get("sub_steps", [])
    )


def _uses_scicode_helpers(records: Iterable[dict[str, Any]]) -> bool:
    return any(
        "scicode.compare" in str(test)
        for problem in records
        for step in problem.get("sub_steps", [])
        for test in step.get("test_cases", [])
    )


def ensure_runtime_ready(records: list[dict[str, Any]], evaluator: SciCodeEvaluator) -> None:
    evaluator.ensure_ready()
    try:
        helper_spec = importlib.util.find_spec("scicode.compare.cmp")
    except ModuleNotFoundError:
        helper_spec = None
    if _uses_scicode_helpers(records) and helper_spec is None:
        raise ModuleNotFoundError(
            "Selected official SciCode tests import `scicode.compare`. Install "
            "the pinned official helper package documented in how_to_run.md."
        )


def _reference_code(step_id: str, reference_dir: Path) -> str:
    if step_id not in SCICODE_SKIPPED_STEPS:
        raise ValueError(f"{step_id} is not an official skipped SciCode step.")
    reference_dir.mkdir(parents=True, exist_ok=True)
    target = reference_dir / f"{step_id}.txt"
    if not target.is_file():
        import requests

        response = requests.get(
            SCICODE_REFERENCE_URL.format(step_id=step_id), timeout=60
        )
        response.raise_for_status()
        target.write_text(response.text, encoding="utf-8")
    return target.read_text(encoding="utf-8")


@dataclass
class SciCodeEvaluation:
    fitness_score: float
    fitness: str
    metrics: dict[str, Any]
    records: list[dict[str, Any]]


def _zero_problem_result(problem: dict[str, Any], error: Exception) -> dict[str, Any]:
    total = score_weight_for_problem(problem)
    return {
        "field": str(problem.get("field", "unknown")),
        "problem_id": str(problem.get("problem_id", "")),
        "problem_name": str(problem.get("problem_name", "")),
        "passed_subproblems": 0,
        "total_subproblems": total,
        "main_problem_passed": False,
        "score": 0.0,
        "predictions": [],
        "steps": [],
        "infrastructure_error": str(error),
    }


def run_problem_trajectory(
    problem: dict[str, Any],
    agent_factory: Callable[[], Any],
    task_info_factory: Callable[[str], Any],
    evaluator: SciCodeEvaluator,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
) -> dict[str, Any]:
    agent = agent_factory()
    previous_code: list[str] = []
    step_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for index, step in enumerate(problem["sub_steps"]):
        step_id = str(step["step_number"])
        if step_id in SCICODE_SKIPPED_STEPS:
            previous_code.append(_reference_code(step_id, reference_dir))
            step_results.append({"step_id": step_id, "status": "skipped"})
            continue

        prompt, prefix = build_step_prompt(problem, index, previous_code)
        try:
            output = agent.forward(task_info_factory(prompt))
            response = extract_agent_text(output)
            current_code = extract_python_script(response)
            if not current_code:
                raise ValueError("Agent returned no executable Python code.")
            predictions.append(
                {
                    "step_id": step_id,
                    "response": response,
                    "extracted_code": current_code,
                }
            )
        except Exception as exc:
            current_code = ""
            evaluation = {
                "passed": False,
                "status": "generation_or_evaluator_error",
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
            }
            predictions.append(
                {
                    "step_id": step_id,
                    "response": None,
                    "extracted_code": "",
                    "error": str(exc),
                }
            )
        else:
            evaluation = evaluator.evaluate_step(step, f"{prefix}\n{current_code}")

        previous_code.append(current_code)
        step_results.append({"step_id": step_id, **evaluation})

    scored = [item for item in step_results if item["status"] != "skipped"]
    passed = sum(bool(item.get("passed")) for item in scored)
    total = len(scored)
    return {
        "field": str(problem["field"]),
        "problem_id": str(problem["problem_id"]),
        "problem_name": str(problem.get("problem_name", "")),
        "passed_subproblems": passed,
        "total_subproblems": total,
        "main_problem_passed": bool(total and passed == total),
        "score": passed / total if total else 0.0,
        "predictions": predictions,
        "steps": step_results,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    field_scores: dict[str, float] = {}
    field_counts: dict[str, dict[str, int]] = {}
    for field in SCICODE_FIELD_LABELS:
        subset = [item for item in results if item.get("field") == field]
        passed = sum(int(item.get("passed_subproblems", 0)) for item in subset)
        total = sum(int(item.get("total_subproblems", 0)) for item in subset)
        field_scores[field] = passed / total if total else 0.0
        field_counts[field] = {"passed": passed, "total": total}

    total_passed = sum(int(item.get("passed_subproblems", 0)) for item in results)
    total_subproblems = sum(int(item.get("total_subproblems", 0)) for item in results)
    passed_main = sum(bool(item.get("main_problem_passed")) for item in results)
    total_main = len(results)
    return {
        "fitness": sum(field_scores.values()) / len(SCICODE_FIELD_LABELS),
        "field_subproblem_pass_rates": field_scores,
        "field_subproblem_counts": field_counts,
        "global_subproblem_pass_rate": (
            total_passed / total_subproblems if total_subproblems else 0.0
        ),
        "passed_subproblems": total_passed,
        "total_subproblems": total_subproblems,
        "main_problem_resolve_rate": passed_main / total_main if total_main else 0.0,
        "passed_main_problems": passed_main,
        "total_main_problems": total_main,
        "infrastructure_errors": sum(
            bool(item.get("infrastructure_error")) for item in results
        ),
    }


def format_fitness(metrics: dict[str, Any]) -> str:
    fields = metrics["field_subproblem_pass_rates"]
    return (
        "Field-macro Subproblem Pass Rate: "
        f"{100 * metrics['fitness']:.2f}% "
        f"(Mathematics={100 * fields['mathematics']:.2f}%, "
        f"Physics={100 * fields['physics']:.2f}%, "
        f"Material Science={100 * fields['material_science']:.2f}%)"
    )


def evaluate_scicode(
    records: list[dict[str, Any]],
    agent_factory: Callable[[], Any],
    task_info_factory: Callable[[str], Any],
    h5py_file: str | os.PathLike[str] | None = None,
    max_workers: int = 3,
    evaluation_timeout: int = 1800,
    reference_dir: str | os.PathLike[str] = DEFAULT_REFERENCE_DIR,
    description: str = "Evaluating SciCode trajectories",
) -> SciCodeEvaluation:
    h5_path = Path(
        h5py_file or os.environ.get("SCICODE_H5_PATH", str(DEFAULT_H5_PATH))
    ).expanduser().resolve()
    evaluator = SciCodeEvaluator(h5_path, timeout=evaluation_timeout)
    ensure_runtime_ready(records, evaluator)
    reference_dir = Path(reference_dir).expanduser().resolve()

    results: list[dict[str, Any] | None] = [None] * len(records)

    def run(index: int) -> dict[str, Any]:
        problem = records[index]
        try:
            return run_problem_trajectory(
                problem,
                agent_factory,
                task_info_factory,
                evaluator,
                reference_dir,
            )
        except Exception as exc:
            return _zero_problem_result(problem, exc)

    worker_count = max(1, min(max_workers, len(records))) if records else 1
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(run, index): index for index in range(len(records))
        }
        for future in tqdm(
            as_completed(future_to_index), total=len(records), desc=description
        ):
            index = future_to_index[future]
            results[index] = future.result()

    complete = [item for item in results if item is not None]
    metrics = aggregate_results(complete)
    return SciCodeEvaluation(
        fitness_score=float(metrics["fitness"]),
        fitness=format_fitness(metrics),
        metrics=metrics,
        records=complete,
    )
