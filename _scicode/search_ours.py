"""ADAS architecture search for the frozen ReMAS-compatible SciCode split."""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import threading
from collections import namedtuple
from pathlib import Path

import backoff
import openai
from dotenv import load_dotenv
from json_repair import repair_json

try:
    from .scicode_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    from .scicode_runtime import (
        build_reasoning_extra_body,
        evaluate_scicode,
        load_scicode_records,
    )
except ImportError:
    from scicode_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    from scicode_runtime import (
        build_reasoning_extra_body,
        evaluate_scicode,
        load_scicode_records,
    )


load_dotenv(override=False)

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
FORMAT_INST = lambda fields: (
    "Reply EXACTLY with the following JSON format.\n"
    f"{fields}\n"
    "Do not omit fields. Return one well-formed JSON object."
)
ROLE_DESC = lambda role: f"You are a {role}."

SEARCHING_MODE = True
client = None
MODEL = "deepseek/deepseek-v4-flash"
SEARCH_TEMPERATURE = 0.8
EVAL_TEMPERATURE = 1.0
MAX_TOKENS = 32768
EXEC_MAX_TOKENS = 16324
PROVIDER_ROUTING = None
SEARCH_PROVIDER_ROUTING = None
EXEC_NO_THINKING = True
SEARCH_THINKING = "high"
EVAL_SEED = None
REASONING_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_BACKEND = "auto"

_search_input_tokens = 0
_search_output_tokens = 0
_exec_input_tokens = 0
_exec_output_tokens = 0
_total_subproblems_evaluated = 0
_exec_token_lock = threading.Lock()


def random_id() -> str:
    return secrets.token_hex(4)


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _completion_kwargs(extra: dict | None) -> dict:
    return {"extra_body": extra} if extra else {}


def configure_reasoning_backend(
    base_url: str,
    reasoning_backend: str = "auto",
) -> None:
    """Select the reasoning payload understood by the configured endpoint."""

    global REASONING_BASE_URL, REASONING_BACKEND
    REASONING_BASE_URL = str(base_url)
    REASONING_BACKEND = reasoning_backend


def reasoning_extra_body(
    effort: str | None,
    *,
    thinking: bool | None,
) -> dict:
    """Build OpenRouter or ReMAS-local thinking controls without mixing them."""

    return build_reasoning_extra_body(
        REASONING_BASE_URL,
        effort,
        thinking=thinking,
        reasoning_backend=REASONING_BACKEND,
    )


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=None):
    global _exec_input_tokens, _exec_output_tokens
    extra = {}
    if PROVIDER_ROUTING:
        extra["provider"] = PROVIDER_ROUTING
    if EXEC_NO_THINKING:
        extra.update(reasoning_extra_body("none", thinking=False))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=EVAL_TEMPERATURE if temperature is None else temperature,
        max_tokens=EXEC_MAX_TOKENS,
        stop=None,
        response_format={"type": "json_object"},
        timeout=300,
        **({"seed": EVAL_SEED} if EVAL_SEED is not None else {}),
        **_completion_kwargs(extra),
    )
    if response.usage:
        with _exec_token_lock:
            _exec_input_tokens += response.usage.prompt_tokens
            _exec_output_tokens += response.usage.completion_tokens
    parsed = repair_json(response.choices[0].message.content or "", return_objects=True)
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), {})
    return parsed if isinstance(parsed, dict) else {}


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(messages, model, temperature=None):
    global _search_input_tokens, _search_output_tokens
    extra = {}
    if SEARCH_PROVIDER_ROUTING:
        extra["provider"] = SEARCH_PROVIDER_ROUTING
    if SEARCH_THINKING is not None:
        extra.update(
            reasoning_extra_body(
                SEARCH_THINKING,
                thinking=SEARCH_THINKING != "none",
            )
        )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=SEARCH_TEMPERATURE if temperature is None else temperature,
        max_tokens=MAX_TOKENS,
        stop=None,
        response_format={"type": "json_object"},
        timeout=300,
        **_completion_kwargs(extra),
    )
    if response.usage:
        _search_input_tokens += response.usage.prompt_tokens
        _search_output_tokens += response.usage.completion_tokens
    parsed = repair_json(response.choices[0].message.content or "", return_objects=True)
    return parsed if isinstance(parsed, dict) else {}


class LLMAgentBase:
    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role: str = "helpful assistant",
        model=None,
        temperature=None,
    ) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model if model is not None else MODEL
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction):
        descriptions = {}
        for field in self.output_fields:
            if "code" in field.lower():
                descriptions[field] = (
                    "One fenced Python code block implementing only the requested "
                    "current function with its exact supplied signature. Do not include "
                    "imports, previous functions, examples, tests, or prose outside the block."
                )
            else:
                descriptions[field] = f"Your {field}."
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(descriptions)
        sections = []
        for info in input_infos:
            if not isinstance(info, Info):
                continue
            field_name, author, content, iteration_idx = info
            if author == self.__repr__():
                author += " (yourself)"
            if field_name == "task":
                sections.append(f"# Your Current SciCode Step:\n{content}")
            elif iteration_idx != -1:
                sections.append(
                    f"### {field_name} #{iteration_idx + 1} by {author}:\n{content}"
                )
            else:
                sections.append(f"### {field_name} by {author}:\n{content}")
        return system_prompt, "\n\n".join(sections + [instruction])

    def query(self, input_infos: list, instruction, iteration_idx=-1):
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = {}
        try:
            response_json = get_json_response_from_gpt(
                prompt, self.model, system_prompt, self.temperature
            )
            if any(field not in response_json for field in self.output_fields):
                raise ValueError("LLM response omitted requested output fields.")
        except Exception as exc:
            if "maximum context length" in str(exc).lower() and SEARCHING_MODE:
                raise AssertionError(
                    "The agent architecture exceeded the model context window."
                ) from exc
            for field in self.output_fields:
                response_json.setdefault(field, "")
        return [
            Info(field, self.__repr__(), response_json.get(field, ""), iteration_idx)
            for field in self.output_fields
        ]

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx)


class AgentSystem:
    pass


def _write_archive(path: Path, archive: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(archive, indent=2), encoding="utf-8")


def _attach_forward(forward_str: str) -> None:
    namespace: dict = {}
    exec(forward_str, globals(), namespace)
    forward = namespace.get("forward")
    callables = [value for value in namespace.values() if callable(value)]
    if forward is None or not callable(forward) or len(callables) != 1:
        raise AssertionError(
            "Agent code must define exactly one callable named forward and no other top-level code."
        )
    setattr(AgentSystem, "forward", forward)


def evaluate_forward_fn(args, forward_str):
    global _total_subproblems_evaluated
    _attach_forward(forward_str)
    records = load_scicode_records(args.data_filename, "search")
    evaluation = evaluate_scicode(
        records,
        AgentSystem,
        lambda prompt: Info("task", "User", prompt, -1),
        h5py_file=args.h5py_file,
        max_workers=args.max_workers,
        evaluation_timeout=args.evaluation_timeout,
        description="SciCode search evaluation",
    )
    _total_subproblems_evaluated += evaluation.metrics["total_subproblems"]
    print(evaluation.fitness)
    detail_path = Path(args.save_dir) / f"{args.expr_name}_latest_evaluation.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(
        json.dumps(
            {"metrics": evaluation.metrics, "records": evaluation.records},
            indent=2,
        ),
        encoding="utf-8",
    )
    return evaluation


def _evaluate_and_record(args, solution: dict) -> None:
    evaluation = evaluate_forward_fn(args, solution["code"])
    solution["fitness_score"] = evaluation.fitness_score
    solution["fitness"] = evaluation.fitness
    solution["metrics"] = evaluation.metrics


def search(args):
    archive_path = Path(args.save_dir) / f"{args.expr_name}_run_archive.json"
    if archive_path.is_file():
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
    else:
        archive = get_init_archive()

    for solution in archive:
        if isinstance(solution.get("fitness_score"), (int, float)):
            continue
        solution["generation"] = solution.get("generation", "initial")
        print(f"============ Initial Archive: {solution['name']} ============")
        try:
            _evaluate_and_record(args, solution)
        except Exception as exc:
            print(f"During initial archive evaluation: {exc}")
            continue
        _write_archive(archive_path, archive)

    integer_generations = [
        item["generation"]
        for item in archive
        if isinstance(item.get("generation"), int)
    ]
    generation = max(integer_generations, default=0) + 1
    while generation <= args.n_generation:
        total_tokens = (
            _search_input_tokens
            + _search_output_tokens
            + _exec_input_tokens
            + _exec_output_tokens
        )
        if args.total_token_budget is not None and total_tokens >= args.total_token_budget:
            print(
                f"Token budget reached: {total_tokens:,} >= "
                f"{args.total_token_budget:,}; stopping search."
            )
            break

        print(f"============ Generation {generation} ============")
        system_prompt, prompt = get_prompt(archive)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(
                messages, args.search_model
            )
            reflection_one, reflection_two = get_reflexion_prompt(
                archive[-1] if archive else None
            )
            for reflection_prompt in (reflection_one, reflection_two):
                messages.extend(
                    [
                        {"role": "assistant", "content": json.dumps(next_solution)},
                        {"role": "user", "content": reflection_prompt},
                    ]
                )
                next_solution = get_json_response_from_gpt_reflect(
                    messages, args.search_model
                )
        except Exception as exc:
            print(f"During architecture generation: {exc}")
            continue

        evaluation = None
        for _attempt in range(args.debug_max):
            try:
                if not all(key in next_solution for key in ("thought", "name", "code")):
                    raise ValueError("Architecture JSON must contain thought, name, and code.")
                evaluation = evaluate_forward_fn(args, next_solution["code"])
                if evaluation.fitness_score < 0.01:
                    raise RuntimeError("Architecture received all-zero SciCode fitness.")
                break
            except Exception as exc:
                print(f"During architecture evaluation: {exc}")
                messages.extend(
                    [
                        {"role": "assistant", "content": json.dumps(next_solution)},
                        {
                            "role": "user",
                            "content": (
                                "The architecture could not be evaluated: "
                                f"{exc}\nFix only the architecture code/API usage. "
                                "Do not request or infer benchmark tests. Return JSON with "
                                "thought, name, code, and debug_thought."
                            ),
                        },
                    ]
                )
                try:
                    next_solution = get_json_response_from_gpt_reflect(
                        messages, args.search_model
                    )
                except Exception as debug_exc:
                    print(f"During architecture debug: {debug_exc}")
                    break

        if evaluation is None or evaluation.fitness_score < 0.01:
            continue

        next_solution["fitness_score"] = evaluation.fitness_score
        next_solution["fitness"] = evaluation.fitness
        next_solution["metrics"] = evaluation.metrics
        next_solution["generation"] = generation
        next_solution.pop("debug_thought", None)
        next_solution.pop("reflection", None)
        archive.append(next_solution)
        _write_archive(archive_path, archive)
        generation += 1

    log_token_usage(args.save_dir, args.expr_name)


def log_token_usage(save_dir: str, expr_name: str) -> None:
    search_total = _search_input_tokens + _search_output_tokens
    exec_total = _exec_input_tokens + _exec_output_tokens
    average = (
        exec_total / _total_subproblems_evaluated
        if _total_subproblems_evaluated
        else 0.0
    )
    report = {
        "search": {
            "total_tokens": search_total,
            "input_tokens": _search_input_tokens,
            "output_tokens": _search_output_tokens,
        },
        "execution": {
            "total_tokens": exec_total,
            "input_tokens": _exec_input_tokens,
            "output_tokens": _exec_output_tokens,
        },
        "avg_execution_tokens_per_subproblem": round(average, 2),
        "total_subproblems_evaluated": _total_subproblems_evaluated,
    }
    path = Path(save_dir) / f"{expr_name}_token_usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Token usage: search={search_total:,}, execution={exec_total:,}, "
        f"average/subproblem={average:.1f}"
    )
    print(f"Token report saved to {path}")


def _resolve_api_key(args) -> str:
    if args.api_key:
        return args.api_key
    url = args.base_url.lower()
    if "groq" in url:
        key = os.environ.get("GROQ_API_KEY")
    elif "openrouter" in url:
        key = os.environ.get("OPENROUTER_API_KEY")
    elif "openai" in url:
        key = os.environ.get("OPENAI_API_KEY")
    else:
        key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    if not key:
        raise ValueError(
            "Provide --api_key or set OPENROUTER_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY."
        )
    return key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filename", default="dataset/scicode_validate.jsonl"
    )
    parser.add_argument(
        "--h5py_file",
        default=os.environ.get("SCICODE_H5_PATH"),
        help="Official SciCode test_data.h5 (or set SCICODE_H5_PATH).",
    )
    parser.add_argument("--evaluation_timeout", type=int, default=1800)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="Concurrent complete main-problem trajectories.",
    )
    parser.add_argument("--save_dir", default="results/")
    parser.add_argument("--expr_name", default="scicode_ours_results")
    parser.add_argument("--n_generation", type=int, default=20)
    parser.add_argument("--total_token_budget", type=int, default=None)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--search_model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--eval_model", default=None)
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--reasoning_backend",
        choices=["auto", "openrouter", "local"],
        default="auto",
        help=(
            "Reasoning request schema. Auto recognizes OpenRouter, localhost, "
            "and the ReMAS local endpoint."
        ),
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--exec_max_tokens", type=int, default=16324)
    parser.add_argument("--search_temperature", type=float, default=0.8)
    parser.add_argument("--eval_temperature", type=float, default=1.0)
    parser.add_argument("--provider_order", default="deepseek, alibaba")
    parser.add_argument("--no_exec_thinking", action="store_true", default=True)
    parser.add_argument("--search_provider_order", default="deepseek, alibaba")
    parser.add_argument(
        "--search_thinking",
        choices=["none", "medium", "high"],
        default="high",
    )
    return parser.parse_args()


def main() -> None:
    global client, MODEL, SEARCH_TEMPERATURE, EVAL_TEMPERATURE
    global MAX_TOKENS, EXEC_MAX_TOKENS, PROVIDER_ROUTING
    global SEARCH_PROVIDER_ROUTING, EXEC_NO_THINKING, SEARCH_THINKING

    args = parse_args()
    api_key = _resolve_api_key(args)
    configure_reasoning_backend(args.base_url, args.reasoning_backend)
    client = make_client(args.base_url, api_key)
    MODEL = args.eval_model or args.search_model
    SEARCH_TEMPERATURE = args.search_temperature
    EVAL_TEMPERATURE = args.eval_temperature
    MAX_TOKENS = args.max_tokens
    EXEC_MAX_TOKENS = args.exec_max_tokens
    EXEC_NO_THINKING = args.no_exec_thinking
    PROVIDER_ROUTING = (
        {
            "order": [item.strip() for item in args.provider_order.split(",")],
            "allow_fallbacks": True,
        }
        if args.provider_order
        else None
    )
    SEARCH_PROVIDER_ROUTING = (
        {
            "order": [
                item.strip() for item in args.search_provider_order.split(",")
            ],
            "allow_fallbacks": True,
        }
        if args.search_provider_order
        else None
    )
    SEARCH_THINKING = args.search_thinking
    search(args)


if __name__ == "__main__":
    main()
