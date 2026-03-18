"""
ADAS MATH search configured for our 4-subject comparison experiment.

Subjects: Prealgebra, Number Theory, Precalculus, Counting & Probability
Level 5 only, seed=42, 10 per subject = 40 total.

Usage:
    cd C:/Users/cheikh/Desktop/ADAS/_math
    python search_ours.py --search_model google/gemini-2.5-flash --base_url https://openrouter.ai/api/v1
    python search_ours.py --search_model gpt-4o --eval_model gpt-3.5-turbo --base_url https://api.openai.com/v1
"""

import argparse
import copy
import json
import os
import random
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from dotenv import load_dotenv
from json_repair import repair_json
from tqdm import tqdm
from typing import Union

# ── Token tracking ────────────────────────────────────────────────────────────
_search_input_tokens = 0
_search_output_tokens = 0
_exec_input_tokens = 0
_exec_output_tokens = 0
_total_questions_evaluated = 0
_exec_token_lock = threading.Lock()

load_dotenv(override=False)  # don't override already-set env vars

from math_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from utils import load_math_examples, score_math, random_id, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

client = None
MODEL = 'gpt-3.5-turbo-0125'  # overridden by args.model in main
SEARCH_TEMPERATURE = 0.8
EVAL_TEMPERATURE = 1.0
MAX_TOKENS = 32768
PROVIDER_ROUTING = None


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key=api_key)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=None):
    global _exec_input_tokens, _exec_output_tokens
    extra = {"provider": PROVIDER_ROUTING} if PROVIDER_ROUTING else {}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=EVAL_TEMPERATURE if temperature is None else temperature,
        max_tokens=MAX_TOKENS, stop=None, response_format={"type": "json_object"},
        extra_body=extra if extra else None,
    )
    if response.usage:
        with _exec_token_lock:
            _exec_input_tokens += response.usage.prompt_tokens
            _exec_output_tokens += response.usage.completion_tokens
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(msg_list, model, temperature=None):
    global _search_input_tokens, _search_output_tokens
    extra = {}
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=SEARCH_TEMPERATURE if temperature is None else temperature,
        max_tokens=MAX_TOKENS, stop=None, response_format={"type": "json_object"},
        extra_body=extra if extra else None,
    )
    if response.usage:
        _search_input_tokens += response.usage.prompt_tokens
        _search_output_tokens += response.usage.completion_tokens
    content = response.choices[0].message.content
    json_dict = repair_json(content, return_objects=True)
    assert json_dict is not None
    return json_dict


class LLMAgentBase():
    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model=None, temperature=None) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model if model is not None else MODEL
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        output_fields_and_description = {
            key: f"Your {key}." if 'answer' not in key
            else f"Your {key}. Provide the final answer in \\boxed{{...}} LaTeX format."
            for key in self.output_fields
        }
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)
        input_infos_text = ''
        for input_info in input_infos:
            if not isinstance(input_info, Info):
                continue
            (field_name, author, content, iteration_idx) = input_info
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
        return system_prompt, input_infos_text + instruction

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            for key in self.output_fields:
                if key not in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and key not in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            archive = json.load(f)
        start = archive[-1]['generation'] if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int) else 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue
        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:", e)
            continue
        solution['fitness'] = bootstrap_confidence_interval(acc_list)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(archive, f, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
            r1, r2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": r1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": r2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
        except Exception as e:
            print("During LLM generate new solution:", e)
            n -= 1
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:", e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nDebug and fix the code. Repeat your 'thought' and put debugging notes in 'debug_thought'."})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
                except Exception as e:
                    print("During LLM debug:", e)
        if not acc_list:
            n -= 1
            continue

        next_solution['fitness'] = bootstrap_confidence_interval(acc_list)
        next_solution['generation'] = n + 1
        next_solution.pop('debug_thought', None)
        next_solution.pop('reflection', None)
        archive.append(next_solution)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(archive, f, indent=4)

    log_token_usage(args.save_dir, args.expr_name)


def log_token_usage(save_dir: str, expr_name: str):
    """Log token usage summary to console and a JSON file."""
    search_total = _search_input_tokens + _search_output_tokens
    exec_total = _exec_input_tokens + _exec_output_tokens
    avg_exec_per_question = (
        exec_total / _total_questions_evaluated if _total_questions_evaluated > 0 else 0
    )

    print("\n" + "=" * 60)
    print("TOKEN USAGE REPORT")
    print("=" * 60)
    print(f"Search tokens  — total: {search_total:,}  |  input: {_search_input_tokens:,}  |  output: {_search_output_tokens:,}")
    print(f"Execution tokens — total: {exec_total:,}  |  input: {_exec_input_tokens:,}  |  output: {_exec_output_tokens:,}")
    print(f"Avg execution tokens per question: {avg_exec_per_question:.1f}  (over {_total_questions_evaluated:,} questions)")
    print("=" * 60 + "\n")

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
        "avg_execution_tokens_per_question": round(avg_exec_per_question, 2),
        "total_questions_evaluated": _total_questions_evaluated,
    }

    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"{expr_name}_token_usage.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Token usage report saved to: {report_path}")


def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = file_path.replace(".json", "_evaluate.json")
    with open(file_path, 'r') as f:
        archive = json.load(f)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_archive = json.load(f)

    current_idx = 0
    while current_idx < len(archive):
        with open(file_path, 'r') as f:
            archive = json.load(f)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        sol['test_fitness'] = bootstrap_confidence_interval(acc_list)
        eval_archive.append(sol)
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as f:
            json.dump(eval_archive, f, indent=4)


def evaluate_forward_fn(args, forward_str):
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    examples = load_math_examples(args.data_filename)
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [ex['problem'] for ex in examples]
    solutions = [ex['solution'] for ex in examples]

    global _total_questions_evaluated
    _total_questions_evaluated += len(examples)
    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = [Info('task', 'User', q, -1) for q in questions]
    agentSystem = AgentSystem()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    acc_list = []
    for q_idx, res in enumerate(results):
        try:
            prediction = res.content if isinstance(res, Info) else str(res)
            correct = score_math(solutions[q_idx], prediction)
        except Exception:
            acc_list.append(0)
            continue
        acc_list.append(1 if correct else 0)

    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="../dataset/math_4subjects.jsonl")
    parser.add_argument('--valid_size', type=int, default=120)  # 30 per subject x 4 subjects
    parser.add_argument('--test_size', type=int, default=0)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='../results/')
    parser.add_argument('--expr_name', type=str, default="math_ours_results")
    parser.add_argument('--n_generation', type=int, default=20)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--search_model', type=str, default='google/gemini-2.5-flash',
                        help='Meta-LLM used to generate new agent designs')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='LLM used inside forward() to answer questions (defaults to --search_model)')
    parser.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1',
                        help='API base URL')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (falls back to OPENROUTER_API_KEY / GROQ_API_KEY / OPENAI_API_KEY)')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Max tokens for all LLM calls')
    parser.add_argument('--search_temperature', type=float, default=0.8,
                        help='Temperature for the meta-LLM (search/reflexion calls)')
    parser.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Temperature for agent LLM calls during evaluation')
    parser.add_argument('--provider_order', type=str, default=None,
                        help='Comma-separated OpenRouter provider order, e.g. "Google Vertex,Together,Groq"')

    args = parser.parse_args()

    api_key = args.api_key
    if api_key is None:
        url = args.base_url.lower()
        if "groq" in url:
            api_key = os.environ.get("GROQ_API_KEY")
        elif "openrouter" in url:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        elif "openai" in url:
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = (os.environ.get("OPENROUTER_API_KEY")
                       or os.environ.get("GROQ_API_KEY")
                       or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise ValueError("Provide --api_key or set OPENROUTER_API_KEY / GROQ_API_KEY / OPENAI_API_KEY")

    client = make_client(base_url=args.base_url, api_key=api_key)
    MODEL = args.eval_model if args.eval_model else args.search_model
    SEARCH_TEMPERATURE = args.search_temperature
    EVAL_TEMPERATURE = args.eval_temperature
    MAX_TOKENS = args.max_tokens
    PROVIDER_ROUTING = {"order": [p.strip() for p in args.provider_order.split(",")], "allow_fallbacks": True} if args.provider_order else None

    SEARCHING_MODE = True
    search(args)

    if args.test_size > 0:
        SEARCHING_MODE = False
        evaluate(args)
