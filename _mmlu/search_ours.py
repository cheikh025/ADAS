"""
ADAS MMLU search configured for our 4-subject comparison experiment.

Key differences from the original search.py:
- Supports any OpenAI-compatible API (OpenRouter, Groq, etc.) via --base_url
- No model restriction (any model name accepted)
- Defaults to our 4-subject CSV (40 queries, seed=42)
- valid_size defaults to 120 (20 per subject x 6 subjects)
- expr_name defaults to mmlu_ours_results

Usage:
    cd C:/Users/cheikh/Desktop/ADAS/_mmlu
    # Same model for both search and eval:
    python search_ours.py --search_model google/gemini-2.5-flash-lite --base_url https://openrouter.ai/api/v1
    # Different models (strong meta-LLM, cheap eval-LLM):
    python search_ours.py --search_model gpt-4o --eval_model gpt-3.5-turbo --base_url https://api.openai.com/v1
    # Groq:
    python search_ours.py --search_model llama-3.1-8b-instant --base_url https://api.groq.com/openai/v1
  # Use same model for both (simplest)                                  
  python search_ours.py --search_model llama-3.1-8b-instant --base_url  
  https://api.groq.com/openai/v1                                                                                                                  # Strong meta-LLM to design agents, cheap model to evaluate             python search_ours.py --search_model gpt-4o --eval_model gpt-3.5-turbo   --base_url https://api.openai.com/v1                                    
  # OpenRouter: powerful search, fast eval                                python search_ours.py --search_model google/gemini-2.5-pro
  --eval_model google/gemini-2.5-flash-lite --base_url                    https://openrouter.ai/api/v1
    """

import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import backoff
import numpy as np
import openai
import pandas
from dotenv import load_dotenv
from json_repair import repair_json
from tqdm import tqdm

load_dotenv(override=False)  # don't override already-set env vars

from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt

from utils import format_multichoice_question, random_id, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

# Initialized in main based on args
client = None
MODEL = 'gpt-3.5-turbo-0125'  # overridden by args.model in main
SEARCH_TEMPERATURE = 0.8
EVAL_TEMPERATURE = 1.0
MAX_TOKENS = 32768
PROVIDER_ROUTING = None  # e.g. {"order": ["Google Vertex", "Together", "Groq"], "allow_fallbacks": True}


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key=api_key)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=None):
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
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(msg_list, model, temperature=None):
    extra = {"provider": PROVIDER_ROUTING} if PROVIDER_ROUTING else {}
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=SEARCH_TEMPERATURE if temperature is None else temperature,
        max_tokens=MAX_TOKENS, stop=None, response_format={"type": "json_object"},
        extra_body=extra if extra else None,
    )
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
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API Key" in str(e):
                raise RuntimeError(f"Authentication failed — check your API key: {e}")
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
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
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
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
            print("During evaluating initial archive:")
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
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
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.search_model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
        if not acc_list:
            n -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1
        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(file_path).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while current_idx < len(archive):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
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
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


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

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df = pandas.read_csv(args.data_filename)
    random.seed(args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [format_multichoice_question(example) for example in examples]
    answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()
    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res]
                print(res)
            elif 'A)' in res:
                predicted_idx = 0
            elif 'B)' in res:
                predicted_idx = 1
            elif 'C)' in res:
                predicted_idx = 2
            elif 'D)' in res:
                predicted_idx = 3
            elif isinstance(res, list):
                try_res = res[1]
                predicted_idx = LETTER_TO_INDEX[try_res.content]
            elif res.content in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res.content]
            elif 'A)' in res.content:
                predicted_idx = 0
            elif 'B)' in res.content:
                predicted_idx = 1
            elif 'C)' in res.content:
                predicted_idx = 2
            elif 'D)' in res.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}: got {repr(res)}")
                acc_list.append(0)
                continue
        except Exception as e:
            acc_list.append(0)
            continue

        if predicted_idx == answers[q_idx]:
            acc_list.append(1)
        else:
            acc_list.append(0)

    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="../dataset/mmlu_4subjects.csv")
    parser.add_argument('--valid_size', type=int, default=120)  # 20 per subject x 6 subjects
    parser.add_argument('--test_size', type=int, default=0)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='../results/')
    parser.add_argument('--expr_name', type=str, default="mmlu_ours_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    # Model / API config — supports any OpenAI-compatible endpoint
    parser.add_argument('--search_model', type=str, default='google/gemini-2.5-flash',
                        help='Meta-LLM used to generate new agent designs')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='LLM used inside forward() to answer questions (defaults to --search_model)')
    parser.add_argument('--base_url', type=str,
                        default='https://openrouter.ai/api/v1',
                        help='API base URL (OpenAI, OpenRouter, Groq, etc.)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (falls back to OPENROUTER_API_KEY / GROQ_API_KEY / OPENAI_API_KEY env vars)')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Max tokens for all LLM calls')
    parser.add_argument('--search_temperature', type=float, default=0.8,
                        help='Temperature for the meta-LLM (search/reflexion calls)')
    parser.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Temperature for agent LLM calls during evaluation')
    parser.add_argument('--provider_order', type=str, default=None,
                        help='Comma-separated OpenRouter provider order, e.g. "Google Vertex,Together,Groq"')

    args = parser.parse_args()

    # Resolve API key — auto-select based on base_url if not explicitly provided
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

    # Search
    SEARCHING_MODE = True
    search(args)

    # Evaluate (only if test_size > 0)
    if args.test_size > 0:
        SEARCHING_MODE = False
        evaluate(args)
