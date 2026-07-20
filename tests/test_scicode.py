from __future__ import annotations

import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import h5py

from _scicode.scicode_evaluator import SciCodeEvaluator
from _scicode.scicode_runtime import (
    aggregate_results,
    build_step_prompt,
    extract_python_script,
    run_problem_trajectory,
)
from _scicode.scicode_spec import (
    SCICODE_HELDOUT_IDS,
    SCICODE_SEARCH_IDS,
    ordered_pairs,
    validate_manifest_records,
)


ROOT = Path(__file__).resolve().parents[1]
Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])


class _FakeAgent:
    prompts: list[str] = []

    def forward(self, task):
        self.prompts.append(task.content)
        index = len(self.prompts)
        return Info(
            "code",
            "fake",
            f"```python\ndef generated_{index}():\n    return {index}\n```",
            -1,
        )


class _PassingEvaluator:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def evaluate_step(self, step, code):
        self.calls.append((str(step["step_number"]), code))
        return {
            "passed": True,
            "status": "pass",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }


class SciCodeSpecTests(unittest.TestCase):
    def test_frozen_ids_match_remas_and_do_not_overlap(self):
        self.assertEqual(
            ordered_pairs("search"),
            [
                ("mathematics", "40"),
                ("mathematics", "31"),
                ("mathematics", "18"),
                ("physics", "19"),
                ("physics", "73"),
                ("physics", "67"),
                ("material_science", "36"),
                ("material_science", "80"),
                ("material_science", "35"),
            ],
        )
        search = {item for values in SCICODE_SEARCH_IDS.values() for item in values}
        heldout = {item for values in SCICODE_HELDOUT_IDS.values() for item in values}
        self.assertFalse(search & heldout)
        self.assertEqual(len(heldout), 30)

    def test_materialized_data_matches_full_record_manifest(self):
        manifest = ROOT / "dataset/scicode_manifest.json"
        if not manifest.is_file():
            self.skipTest("Run dataset/build_scicode_data.py first.")
        for filename, split, expected_size in (
            ("scicode_validate.jsonl", "search", 9),
            ("scicode_test.jsonl", "heldout", 30),
        ):
            path = ROOT / "dataset" / filename
            with path.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), expected_size)
            validate_manifest_records(records, split, manifest)

    def test_field_macro_fitness_matches_remas(self):
        metrics = aggregate_results(
            [
                {
                    "field": "mathematics",
                    "passed_subproblems": 1,
                    "total_subproblems": 1,
                    "main_problem_passed": True,
                },
                {
                    "field": "physics",
                    "passed_subproblems": 0,
                    "total_subproblems": 9,
                    "main_problem_passed": False,
                },
                {
                    "field": "material_science",
                    "passed_subproblems": 1,
                    "total_subproblems": 2,
                    "main_problem_passed": False,
                },
            ]
        )
        self.assertAlmostEqual(metrics["fitness"], 0.5)
        self.assertAlmostEqual(metrics["global_subproblem_pass_rate"], 2 / 12)
        self.assertAlmostEqual(metrics["main_problem_resolve_rate"], 1 / 3)

    def test_code_extraction_removes_imports(self):
        response = (
            "before```python\nimport math\nfrom numpy import array\n"
            "def f():\n    return 3\n```after"
        )
        self.assertEqual(extract_python_script(response), "def f():\n    return 3")


class SciCodeEvaluatorTests(unittest.TestCase):
    def test_hdf5_fixture_executes_official_style_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "targets.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("900.1/test1/answer", data=10)
            evaluator = SciCodeEvaluator(h5_path, timeout=10)
            result = evaluator.evaluate_step(
                {
                    "step_number": "900.1",
                    "test_cases": ["assert square_plus_one(3) == target"],
                },
                "def square_plus_one(value):\n    return value * value + 1",
            )
            self.assertTrue(result["passed"], result["stderr"])


class SciCodeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        _FakeAgent.prompts = []

    def test_previous_code_propagates_but_tests_never_enter_prompt(self):
        problem = {
            "field": "mathematics",
            "problem_id": "40",
            "problem_name": "fixture",
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                {
                    "step_number": "40.1",
                    "step_description_prompt": "first description",
                    "step_background": "first background",
                    "function_header": "def generated_1():",
                    "return_line": "return result",
                    "test_cases": ["SECRET_TEST_ONE"],
                },
                {
                    "step_number": "40.2",
                    "step_description_prompt": "second description",
                    "step_background": "second background",
                    "function_header": "def generated_2():",
                    "return_line": "return result",
                    "test_cases": ["SECRET_TEST_TWO"],
                },
            ],
        }
        evaluator = _PassingEvaluator()
        result = run_problem_trajectory(
            problem,
            _FakeAgent,
            lambda prompt: Info("task", "User", prompt, -1),
            evaluator,
        )

        self.assertEqual(result["passed_subproblems"], 2)
        self.assertIn("def generated_1():", _FakeAgent.prompts[1])
        self.assertIn("def generated_1():", evaluator.calls[1][1])
        self.assertTrue(evaluator.calls[1][1].startswith("import numpy as np"))
        self.assertNotIn("SECRET_TEST", "\n".join(_FakeAgent.prompts))

    def test_prompt_requires_exact_number_of_previous_steps(self):
        problem = {
            "required_dependencies": "",
            "sub_steps": [
                {
                    "step_description_prompt": "description",
                    "step_background": "background",
                    "function_header": "def f():",
                    "return_line": "return 1",
                }
            ],
        }
        with self.assertRaises(ValueError):
            build_step_prompt(problem, 0, ["unexpected"])


if __name__ == "__main__":
    unittest.main()
