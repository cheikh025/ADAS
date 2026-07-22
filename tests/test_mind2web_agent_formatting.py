from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from _mind2web import search_ours
from _mind2web.mind2web_prompt import get_init_archive
from _mind2web.mind2web_runtime import evaluate_action, extract_agent_output


class Mind2WebAgentFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = search_ours.Info("task", "User", "fixture task", -1)

    def test_format_instruction_contains_valid_json(self):
        instruction = search_ours.FORMAT_INST(
            {
                "analysis": "Your analysis.",
                "action": {
                    "element": "A",
                    "operation": "CLICK",
                    "value": "",
                },
            }
        )
        json.loads(instruction.splitlines()[1])

    def test_flat_action_is_normalized_without_losing_it(self):
        agent = search_ours.LLMAgentBase(
            ["analysis", "action"], "Formatting fixture"
        )
        response = {"element": "B", "operation": "click", "value": ""}
        with patch.object(
            search_ours, "get_json_response_from_gpt", return_value=response
        ) as call:
            analysis, action = agent([self.task], "Choose the current action.")

        self.assertEqual(call.call_count, 1)
        self.assertEqual(analysis.content, "")
        self.assertEqual(
            action.content,
            {"element": "B", "operation": "CLICK", "value": ""},
        )

    def test_missing_fields_are_retried(self):
        agent = search_ours.LLMAgentBase(
            ["analysis", "action"], "Retry fixture"
        )
        responses = [
            {"unrelated": "value"},
            {
                "analysis": "checked",
                "action": {
                    "element": "A",
                    "operation": "TYPE",
                    "value": "hello",
                },
            },
        ]
        with patch.object(
            search_ours, "get_json_response_from_gpt", side_effect=responses
        ) as call:
            _, action = agent([self.task], "Choose the current action.")

        self.assertEqual(call.call_count, 2)
        self.assertEqual(action.content["operation"], "TYPE")

    def test_reviewer_failure_is_not_silently_replaced_with_empty_fields(self):
        reviewer = search_ours.LLMAgentBase(
            ["feedback", "correct"], "Reviewer fixture"
        )
        with patch.object(
            search_ours,
            "get_json_response_from_gpt",
            return_value={"code": "return the_wrong_thing"},
        ) as call:
            with self.assertRaisesRegex(ValueError, "failed to return"):
                reviewer([self.task], "Review the answer.")

        self.assertEqual(call.call_count, search_ours.EXEC_FORMAT_ATTEMPTS)

    def test_multi_agent_baselines_keep_a_valid_proposal_when_judge_fails(self):
        entries = {item["name"]: item for item in get_init_archive()}
        gold = {
            "acceptable_letters": ["A"],
            "candidate_letters": ["A", "B"],
            "op": "CLICK",
            "value": "",
        }
        for name in ("Web Action Self-Consistency", "Web Action Debate"):
            responses = [
                {
                    "analysis": f"proposal {index}",
                    "action": {
                        "element": "A",
                        "operation": "CLICK",
                        "value": "",
                    },
                }
                for index in range(3)
            ]
            responses.extend(
                [{"wrong": "judge output"}, {"still_wrong": "judge output"}]
            )
            namespace = {}
            exec(entries[name]["code"], vars(search_ours), namespace)
            with self.subTest(agent=name), patch.object(
                search_ours,
                "get_json_response_from_gpt",
                side_effect=responses,
            ):
                output = namespace["forward"](search_ours.AgentSystem(), self.task)
                evaluation = evaluate_action(extract_agent_output(output), gold)
                self.assertTrue(evaluation["step_success"])


if __name__ == "__main__":
    unittest.main()
