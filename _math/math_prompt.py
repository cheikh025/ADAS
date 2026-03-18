import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
"""
}

COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{...}."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], cot_instruction)
    return answer
"""
}

COT_SC = {
    "thought": "Multiple CoT agents with majority voting to improve reliability.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{...}."
    N = 5
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]

    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    answer = majority_voting(possible_answers)
    return answer
"""
}

Reflexion = {
    "thought": "An iterative refinement approach where the agent reflects on its own previous answer and improves it.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{...}."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], cot_instruction)

    for i in range(3):
        critic_instruction = "Review your previous answer. If it is incorrect or can be improved, provide a revised answer in \\\\boxed{...}. If it is correct, restate it."
        critic_agent = LLMAgentBase(['reflection', 'answer'], 'Critic Agent')
        reflection, new_answer = critic_agent([taskInfo, thinking, answer], critic_instruction, iteration_idx=i)
        answer = new_answer

    return answer
"""
}

LLM_debate = {
    "thought": "Multiple agents debate the solution, then a judge selects the best answer.",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    solve_instruction = "Please think step by step and solve the math problem. Put your answer in \\\\boxed{...}."
    N = 3
    agents = [LLMAgentBase(['thinking', 'answer'], f'Solver Agent {i}', temperature=0.8) for i in range(N)]

    all_answers = []
    for agent in agents:
        thinking, answer = agent([taskInfo], solve_instruction)
        all_answers.extend([thinking, answer])

    judge_instruction = "Given the math problem and several proposed solutions, select the most likely correct answer. Return it in \\\\boxed{...}."
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Judge Agent', temperature=0.1)
    thinking, answer = judge_agent([taskInfo] + all_answers, judge_instruction)
    return answer
"""
}

Take_a_step_back = {
    "thought": "First derive the general principle or formula, then apply it to solve the specific problem.",
    "name": "Step-Back Abstraction",
    "code": """def forward(self, taskInfo):
    principle_instruction = "What is the general mathematical principle, theorem, or formula needed to solve this type of problem? Do not solve it yet."
    principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
    thinking, principle = principle_agent([taskInfo], principle_instruction)

    solve_instruction = "Using the principle identified, solve the math problem step by step. Put your final answer in \\\\boxed{...}."
    solve_agent = LLMAgentBase(['thinking', 'answer'], 'Solver Agent')
    thinking, answer = solve_agent([taskInfo, principle], solve_instruction)
    return answer
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the MATH benchmark, specifically Level 5 competition mathematics problems across Prealgebra, Number Theory, Precalculus, and Counting & Probability.

## An example question from MATH:

Solve the following competition math problem.

How many integers between 1 and 200 are divisible by both 3 and 5?

(Answer: 13, written as \\boxed{13})

# The utility code:

```python
from collections import namedtuple
from typing import Union
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\\n{str(request_keys)}\\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        output_fields_and_description = {
            key: f"Your {key}." if 'answer' not in key
            else f"Your {key}. Provide the final answer in \\\\boxed{{...}} LaTeX format."
            for key in self.output_fields
        }
        system_prompt = ROLE_DESC(self.role) + "\\n\\n" + FORMAT_INST(output_fields_and_description)
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\\n{content}\\n\\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\\n{content}\\n\\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\\n{content}\\n\\n'
        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    def forward(self, taskInfo) -> Union[Info, str]:
        pass
```

# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
Finally, the last key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture.
Also, it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Infos.content.
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
if len(feedback_info) < 3:
    return 'Error: Feedback info incomplete'
```
First, you should never return an error message. You should always return the best answer you can get.
Second, you should never print anything in the code.
Lastly, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent in agents:
    outputs = agent([taskInfo], instruction)
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)
aggregated = '\\n'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. Put those Info objects into a list and use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo], reasoning_instruction)
for info in response_infos:
    if info.name == 'final_answer':
        return info
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
You should not extract the final answer by yourself. You SHOULD directly unpack and return the answer Info.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo], reasoning_instruction)
return answer
```

5. This is WRONG: ```
import sympy as sp
import re

def forward(self, taskInfo):
    ...
```
Do NOT add any import statements or top-level code outside the forward() function body. All necessary imports (json, collections, etc.) are already available. If you need collections.Counter, import it inside the function as shown in the archive examples.

6. This is WRONG: ```
def forward(self, taskInfo) -> Union[Info, str]:
    ...
```
Do NOT use return type annotations that require imports. Just write `def forward(self, taskInfo):` with no return type hint.

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new agents.
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX.
"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings.
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back]


def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2
