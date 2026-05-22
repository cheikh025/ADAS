import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return code_solution
"""
}

COT = {
    "thought": "Think through the algorithm before writing code to reduce errors in complex problems.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    instruction = "First analyze the problem carefully and plan your approach step by step, then write a complete, correct code solution."
    coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent', temperature=0.3)
    thinking, code = coder([taskInfo], instruction)
    return code
"""
}

COT_SC = {
    "thought": "Generate multiple independent code solutions with reasoning, then use a low-temperature judge to compare the approaches and select the most robust complete solution.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    solve_instruction = "Please think step by step about the algorithm, edge cases, input/output format, and implementation details. Then write a complete, correct code solution for the given programming problem."
    N = 5
    coders = [LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Coder Agent', temperature=0.8) for _ in range(N)]

    possible_solutions = []
    for coder in coders:
        thinking, code = coder([taskInfo], solve_instruction)
        possible_solutions.extend([thinking, code])

    judge_instruction = "Given multiple independently reasoned candidate code solutions for the same programming problem, compare their algorithms, edge-case handling, and completeness. Select the most likely correct solution. Return only the complete code solution."
    judge = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    thinking, code = judge([taskInfo] + possible_solutions, judge_instruction)
    return code
"""
}

Reflexion = {
    "thought": "Generate code, then self-critique and refine it iteratively to catch bugs before submission.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    solve_initial_instruction = "Write a complete, correct code solution for the given programming problem."
    solve_reflect_instruction = "Given previous attempts and feedback, carefully consider where the code could be wrong. Using insights from previous attempts, write an improved code solution."
    coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent', temperature=0.3)

    critic_instruction = "Please review the code solution above and criticize on where it might be wrong or incomplete. If you are absolutely sure it is correct, output 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')

    N_max = 5

    coder_inputs = [taskInfo]
    thinking, code = coder(coder_inputs, solve_initial_instruction, 0)

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking, code], critic_instruction, i)
        if correct.content == 'True':
            break
        coder_inputs.extend([thinking, code, feedback])
        thinking, code = coder(coder_inputs, solve_reflect_instruction, i + 1)
    return code
"""
}

LLM_debate = {
    "thought": "Multiple agents propose and debate code solutions across rounds, with a judge picking the best.",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    debate_initial_instruction = "Write a complete, correct code solution for the given programming problem. Explain your approach briefly."
    debate_instruction = "Given code solutions to the problem from other agents, consider their approaches as additional advice. Please think carefully and provide an updated code solution."

    debate_agents = [LLMAgentBase(['thinking', 'code'], 'Debate Agent', temperature=0.8, role=role) for role in ['Algorithm and Data Structures Expert', 'Scientific Computing Expert', 'Data Analysis Expert', 'General Programming Expert']]

    final_decision_instruction = "Given all the above thinking and code solutions, select the most correct and complete one. Return only that code solution."
    final_decision_agent = LLMAgentBase(['code'], 'Final Decision Agent', temperature=0.1)

    max_round = 2
    all_thinking = [[] for _ in range(max_round)]
    all_code = [[] for _ in range(max_round)]

    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, code = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, code = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_code[r].append(code)

    code, = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_code[max_round-1], final_decision_instruction)
    return code
"""
}

Take_a_step_back = {
    "thought": "Let LLM first identify the key algorithms, data structures, and implementation strategy before writing code. By abstracting the problem first, the model can write more correct and efficient solutions.",
    "name": "Step-back Abstraction",
    "code": """def forward(self, taskInfo):
        principle_instruction = "What are the key algorithms, data structures, and implementation steps needed to solve this programming problem? First think step by step. Then list all involved concepts and explain them."
        cot_instruction = "Given the programming problem and the identified algorithms and implementation strategy, write a complete, correct code solution."

        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent', temperature=0.3)

        thinking, principle = principle_agent([taskInfo], principle_instruction)
        thinking, code = coder([taskInfo, thinking, principle], cot_instruction)
        return code
"""
}

QD = {
    "thought": "Similar to Quality-Diversity methods, generate multiple diverse code solutions using different algorithms or approaches, then select the best one.",
    "name": "Quality-Diversity",
    "code": """def forward(self, taskInfo):
    cot_initial_instruction = "Write a complete, correct code solution for the given programming problem."
    qd_instruction = "Given previous code solutions, try to come up with another interesting approach to solve the problem. Use a different algorithm or data structure if possible."
    coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent', temperature=0.8)

    final_decision_instruction = "Given all the above code solutions, select the most correct and complete one. Return only that solution."
    final_decision_agent = LLMAgentBase(['code'], 'Final Decision Agent', temperature=0.1)

    N_max = 3

    coder_inputs = [taskInfo]
    possible_solutions = []
    thinking, code = coder(coder_inputs, cot_initial_instruction, 0)
    possible_solutions.extend([thinking, code])

    for i in range(N_max):
        coder_inputs.extend([thinking, code])
        thinking, code = coder(coder_inputs, qd_instruction, i + 1)
        possible_solutions.extend([thinking, code])

    code, = final_decision_agent([taskInfo] + possible_solutions, final_decision_instruction)
    return code
"""
}

Role_Assignment = {
    "thought": "Similar to Auto-GPT and expert prompting, route the coding task to the most relevant domain expert based on the problem type.",
    "name": "Dynamic Assignment of Roles",
    "code": """def forward(self, taskInfo):
        cot_instruction = "Write a complete, correct code solution for the given programming problem."
        expert_agents = [LLMAgentBase(['thinking', 'code'], 'Expert Agent', role=role) for role in ['Algorithm and Data Structures Expert', 'Scientific Computing Expert', 'Data Analysis Expert', 'Web and Desktop Development Expert', 'General Programming Expert']]

        routing_instruction = "Given the programming task, choose the most relevant expert. Choose from: Algorithm and Data Structures, Scientific Computing, Data Analysis, Web and Desktop Development Expert, or General Programming Expert."
        routing_agent = LLMAgentBase(['choice'], 'Routing Agent')

        choice = routing_agent([taskInfo], routing_instruction)[0]

        if 'algorithm' in choice.content.lower() or 'data structure' in choice.content.lower():
            expert_id = 0
        elif 'scientific' in choice.content.lower():
            expert_id = 1
        elif 'data analysis' in choice.content.lower() or 'analysis' in choice.content.lower():
            expert_id = 2
        elif 'web' in choice.content.lower() or 'desktop' in choice.content.lower():
            expert_id = 3
        else:
            expert_id = 4

        thinking, code = expert_agents[expert_id]([taskInfo], cot_instruction)
        return code
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on FullStackBench, a challenging code generation benchmark covering hard programming problems across multiple languages and application domains (Advanced Programming, Scientific Computing, Data Analysis, Desktop and Web Development).

## An example task from FullStackBench:

Implement a function `merge_intervals(intervals)` that takes a list of intervals [start, end] and returns a new list with all overlapping intervals merged.

Example: merge_intervals([[1,3],[2,6],[8,10],[15,18]]) → [[1,6],[8,10],[15,18]]

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
            key: f"Your {key}." if 'code' not in key
            else f"Your {key}. Return the complete, correct code solution as a plain string."
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

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list:
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
        # The output of the LLM is a list of Info. If you are only querying one output, unpack with [0] or tuple unpacking.
        # It is good practice to always include 'thinking' in the output alongside 'code'.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    def forward(self, taskInfo) -> Union[Info, str]:
        pass
```

# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the pass rate on a validation set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
Finally, the last key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture.
Also, it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Info.content.
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.
The agent's forward() must return the raw code Info or string — SandboxFusion extracts and executes it directly. Do NOT pre-extract or post-process the code.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
review, code = reviewer([taskInfo, code], review_instruction, i)
code_info = fixer([taskInfo, Info('code', 'Reviewer', code, 0)], fix_instruction)
```
It is wrong to use "Info('code', 'Reviewer', code, 0)". The returned "code" from LLMAgentBase is already an Info object. Pass it directly.

2. This is WRONG: ```
# Debugging: Log the generated code
print('Generated code:', ...)
if len(results) < 2:
    return 'Error: incomplete results'
```
First, you should never return an error message. Always return the best code you can produce.
Second, you should never print anything in the code.
Lastly, DO NOT CREATE Info objects by yourself.

3. This is WRONG: ```
solutions = []
for coder in coders:
    outputs = coder([taskInfo], instruction)
    solutions.append(outputs[0].content)
aggregated = '\\n\\n'.join(solutions)
```
You SHOULD NOT extract the content from the Info object by yourself. Use the Info object directly. Put those Info objects into a list and pass the list as input to the next LLM agent.

4. This is WRONG: ```
coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent')
response_infos = coder([taskInfo], instruction)
for info in response_infos:
    if info.name == 'code':
        return info
return Info('code', 'Coder Agent', '', 0)
```
You should not search for fields by name manually. Unpack directly and return the code Info.
CORRECT example: ```
coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent')
thinking, code = coder([taskInfo], instruction)
return code
```

5. This is WRONG: ```
import re
import subprocess

def forward(self, taskInfo):
    ...
```
Do NOT add any import statements or top-level code outside the forward() function body. All necessary imports (json, collections, etc.) are already available. If you need something like collections.Counter, import it inside the function.

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
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


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
