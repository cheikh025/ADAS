from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _mind2web.mind2web_dom import Mind2WebDOMError, format_pruned_html
from _mind2web.mind2web_runtime import (
    aggregate_results,
    build_action_prompt,
    build_reasoning_extra_body,
    evaluate_action,
    evaluate_mind2web,
    mind2web_provider_routing,
    official_token_f1,
    parse_action,
    run_task,
)
from _mind2web.mind2web_spec import (
    select_protocol_splits,
    validate_manifest_records,
)
from dataset.build_mind2web_data import build_step


ROOT = Path(__file__).resolve().parents[1]
Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])


def _step(index: int, previous: list[str], candidates: str) -> dict:
    return {
        "step": index,
        "previous_actions": previous,
        "page_context": f"page-{index}",
        "candidates": candidates,
        "letter_to_node_id": {"A": "10", "B": "20"},
    }


def _gold(index: int, acceptable: list[str], op: str, value: str = "") -> dict:
    return {
        "step": index,
        "candidate_letters": ["A", "B"],
        "acceptable_letters": acceptable,
        "acceptable_node_ids": ["10" if "A" in acceptable else "20"],
        "op": op,
        "value": value,
    }


class _PromptAgent:
    prompts: list[str] = []
    instances = 0
    lock = threading.Lock()

    def __init__(self):
        with self.lock:
            self.__class__.instances += 1

    def forward(self, task):
        with self.lock:
            self.__class__.prompts.append(task.content)
        if "step-two-marker" in task.content:
            action = {"element": "B", "operation": "TYPE", "value": "hello"}
        else:
            action = {"element": "A", "operation": "CLICK", "value": ""}
        return Info("action", "fake", action, -1)


class Mind2WebProtocolTests(unittest.TestCase):
    def test_reasoning_payload_matches_openrouter_and_local_backends(self):
        self.assertEqual(
            build_reasoning_extra_body(
                "https://openrouter.ai/api/v1", "none", thinking=False
            ),
            {"reasoning": {"effort": "none"}},
        )

        self.assertEqual(
            build_reasoning_extra_body(
                "https://ai4news.rnd.huawei.com/model/v1",
                "high",
                thinking=True,
            ),
            {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"thinking": True},
            },
        )
        self.assertEqual(
            build_reasoning_extra_body(
                "https://ai4news.rnd.huawei.com/model/v1",
                "none",
                thinking=False,
            ),
            {"chat_template_kwargs": {"thinking": False}},
        )

    def test_openrouter_routing_matches_remas_and_is_scoped(self):
        fallback = {"order": ["other"], "allow_fallbacks": True}
        self.assertEqual(
            mind2web_provider_routing("https://openrouter.ai/api/v1", fallback),
            {"only": ["deepseek"], "allow_fallbacks": False},
        )
        self.assertIs(
            mind2web_provider_routing("http://localhost:8000/v1", fallback),
            fallback,
        )

    def test_shared_rng_split_order_is_frozen_and_disjoint(self):
        records = [
            {"id": f"{domain[0]}-{index:03d}", "domain": domain}
            for domain in ("Travel", "Shopping", "Entertainment")
            for index in range(125)
        ]
        search, heldout = select_protocol_splits(records)
        self.assertEqual(
            [item["id"] for item in search[:8]],
            [
                "T-081",
                "T-014",
                "T-003",
                "T-094",
                "T-035",
                "T-031",
                "T-028",
                "T-017",
            ],
        )
        self.assertEqual(
            [item["id"] for item in search if item["scenario_id"] == "shopping"][:4],
            ["S-003", "S-071", "S-025", "S-091"],
        )
        self.assertEqual(len(search), 60)
        self.assertEqual(len(heldout), 300)
        self.assertFalse(
            {item["id"] for item in search}
            & {item["id"] for item in heldout}
        )

    def test_generated_files_match_full_content_manifest(self):
        manifest = ROOT / "dataset/mind2web_manifest.json"
        if not manifest.is_file():
            self.skipTest("Run dataset/build_mind2web_data.py first.")
        for filename, split, expected_size in (
            ("mind2web_validate.jsonl", "search", 60),
            ("mind2web_test.jsonl", "heldout", 300),
        ):
            with (ROOT / "dataset" / filename).open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), expected_size)
            validate_manifest_records(records, split, manifest)

    def test_builder_does_not_inject_ranker_missed_gold(self):
        action = {
            "action_uid": "a1",
            "cleaned_html": (
                '<html backend_node_id="1"><body backend_node_id="2">'
                '<button backend_node_id="10" title="buy"/>'
                '<button backend_node_id="20" title="cancel"/>'
                "</body></html>"
            ),
            "pos_candidates": [{"backend_node_id": "10"}],
            "neg_candidates": [{"backend_node_id": "20"}],
            "operation": {"op": "CLICK", "value": ""},
        }
        step, gold = build_step(
            action, 0, [], "task", {"task_a1": {"10": 99, "20": 0}}
        )
        self.assertEqual(set(step["letter_to_node_id"].values()), {"20"})
        self.assertEqual(gold["acceptable_letters"], [])


class Mind2WebDOMTests(unittest.TestCase):
    def test_official_pruning_keeps_candidate_neighborhood(self):
        buttons = "".join(
            f'<button backend_node_id="{index}" title="button {index}"/>'
            for index in range(10, 19)
        )
        page, candidates = format_pruned_html(
            '<html backend_node_id="1"><body backend_node_id="2">'
            + buttons
            + "</body></html>",
            ["14"],
        )
        self.assertIn("14", candidates)
        self.assertIn("button 14", candidates["14"])
        self.assertIn("button 11", page)
        self.assertIn("button 17", page)
        self.assertNotIn("button 10", page)
        self.assertNotIn("button 18", page)

    def test_missing_ranked_node_fails_closed(self):
        with self.assertRaises(Mind2WebDOMError):
            format_pruned_html('<html backend_node_id="1"/>', ["404"])


class Mind2WebScoringTests(unittest.TestCase):
    def test_parser_and_official_action_f1(self):
        parsed = parse_action(
            "```json\n{element: 'b', operation: 'type', value: 'red shoes'}\n```",
            {"A", "B"},
        )
        self.assertEqual(
            parsed,
            {"element": "B", "operation": "TYPE", "value": "red shoes"},
        )
        self.assertEqual(official_token_f1("TYPE shoes red", "TYPE red shoes"), 1.0)
        self.assertLess(
            official_token_f1("TYPE red shoes now", "TYPE red shoes"), 1.0
        )

    def test_step_requires_element_and_exact_action_f1(self):
        gold = _gold(0, ["A"], "TYPE", "red shoes")
        right = evaluate_action(
            {"element": "A", "operation": "TYPE", "value": "shoes red"}, gold
        )
        wrong_element = evaluate_action(
            {"element": "B", "operation": "TYPE", "value": "red shoes"}, gold
        )
        extra_value = evaluate_action(
            {"element": "A", "operation": "TYPE", "value": "red shoes now"},
            gold,
        )
        self.assertTrue(right["step_success"])
        self.assertFalse(wrong_element["step_success"])
        self.assertFalse(extra_value["step_success"])

    def test_domain_macro_uses_task_macro_step_success(self):
        results = [
            {"scenario_id": "travel", "element_acc": 1.0, "action_f1": 1.0, "step_success_rate": 1.0, "task_success": 1.0, "candidate_recall": 1.0, "num_steps": 1, "generation_errors": 0},
            {"scenario_id": "shopping", "element_acc": 0.0, "action_f1": 0.0, "step_success_rate": 0.0, "task_success": 0.0, "candidate_recall": 1.0, "num_steps": 100, "generation_errors": 0},
            {"scenario_id": "entertainment", "element_acc": 0.5, "action_f1": 0.5, "step_success_rate": 0.5, "task_success": 0.0, "candidate_recall": 1.0, "num_steps": 2, "generation_errors": 0},
        ]
        metrics = aggregate_results(results)
        self.assertEqual(metrics["fitness"], 0.5)
        self.assertNotAlmostEqual(metrics["fitness"], 2 / 103)


class Mind2WebTrajectoryTests(unittest.TestCase):
    def setUp(self):
        _PromptAgent.prompts = []
        _PromptAgent.instances = 0

    def _problem(self):
        return {
            "scenario_id": "travel",
            "domain": "Travel",
            "id": "fixture",
            "task": "perform fixture",
            "website": "example",
            "num_steps": 2,
            "steps": [
                _step(0, [], "A) first-marker\nB) unused"),
                _step(
                    1,
                    ["[button] first -> CLICK"],
                    "A) unused\nB) step-two-marker",
                ),
            ],
            "gold_actions": [
                _gold(0, ["A"], "CLICK"),
                _gold(1, ["B"], "TYPE", "hello"),
            ],
        }

    def test_actions_are_fresh_independent_and_teacher_forced(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            result = run_task(
                self._problem(),
                _PromptAgent,
                lambda prompt: Info("task", "User", prompt, -1),
                executor,
            )
        self.assertEqual(result["step_success_rate"], 1.0)
        self.assertEqual(_PromptAgent.instances, 2)
        first = next(p for p in _PromptAgent.prompts if "first-marker" in p)
        second = next(p for p in _PromptAgent.prompts if "step-two-marker" in p)
        self.assertIn("GOLD PREVIOUS ACTIONS", first)
        self.assertIn("None", first)
        self.assertNotIn("[button] first -> CLICK", first)
        self.assertIn("[button] first -> CLICK", second)
        self.assertNotIn('"value":"hello"', second)

    def test_candidate_miss_skips_agent_call(self):
        problem = self._problem()
        problem["steps"] = problem["steps"][:1]
        problem["gold_actions"] = [_gold(0, [], "CLICK")]
        problem["num_steps"] = 1
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = run_task(
                problem,
                _PromptAgent,
                lambda prompt: Info("task", "User", prompt, -1),
                executor,
            )
        self.assertEqual(result["step_success_rate"], 0.0)
        self.assertEqual(result["candidate_recall"], 0.0)
        self.assertEqual(_PromptAgent.instances, 0)

    def test_complete_evaluator_preserves_domain_macro(self):
        records = []
        for scenario, domain in (
            ("travel", "Travel"),
            ("shopping", "Shopping"),
            ("entertainment", "Entertainment"),
        ):
            problem = self._problem()
            problem.update(
                {"scenario_id": scenario, "domain": domain, "id": scenario}
            )
            records.append(problem)
        evaluation = evaluate_mind2web(
            records,
            _PromptAgent,
            lambda prompt: Info("task", "User", prompt, -1),
            max_task_workers=3,
            max_llm_calls=3,
            description="fixture",
        )
        self.assertEqual(evaluation.fitness_score, 1.0)
        self.assertEqual(evaluation.metrics["n_tasks"], 3)
        self.assertEqual(evaluation.metrics["n_steps"], 6)

    def test_prompt_does_not_include_internal_rank_mapping(self):
        step = _step(0, [], "A) candidate")
        prompt = build_action_prompt("task", step)
        self.assertNotIn("letter_to_node_id", prompt)
        self.assertNotIn('"10"', prompt)


if __name__ == "__main__":
    unittest.main()
