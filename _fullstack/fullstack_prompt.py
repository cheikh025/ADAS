import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return code_solution
"""
}

DirectCode = {
    "thought": "The simplest approach: directly ask the LLM to write a complete code solution. The raw output is passed to SandboxFusion which handles code extraction.",
    "name": "Direct Code Generation",
    "code": """def forward(self, taskInfo):
    instruction = "Write a complete, correct code solution for the given programming problem. Output only the code."
    coder = LLMAgentBase(['code'], 'Coder Agent', temperature=0.3)
    code, = coder([taskInfo], instruction)
    return code
"""
}

COT_Code = {
    "thought": "Think through the algorithm before writing code to reduce errors in complex problems.",
    "name": "Chain-of-Thought Code Generation",
    "code": """def forward(self, taskInfo):
    instruction = "First analyze the problem carefully and plan your approach step by step, then write a complete, correct code solution."
    coder = LLMAgentBase(['thinking', 'code'], 'Coder Agent', temperature=0.3)
    thinking, code = coder([taskInfo], instruction)
    return code
"""
}

SC_Code = {
    "thought": "Generate multiple solutions and select the most consistent one — useful when the correct approach is ambiguous.",
    "name": "Self-Consistency Code Generation",
    "code": """def forward(self, taskInfo):
    solve_instruction = "Write a complete, correct code solution for the given programming problem. Output only the code."
    N = 3
    coders = [LLMAgentBase(['code'], 'Coder Agent', temperature=0.8) for _ in range(N)]

    solutions = []
    for coder in coders:
        code, = coder([taskInfo], solve_instruction)
        solutions.append(code.content)

    judge_instruction = "Given multiple candidate code solutions for the same problem, select the one most likely to be correct and complete. Return only that solution."
    judge = LLMAgentBase(['code'], 'Judge Agent', temperature=0.1)
    code, = judge([taskInfo] + [Info('solution', 'Coder', s, i) for i, s in enumerate(solutions)], judge_instruction)
    return code
"""
}

Reflexion_Code = {
    "thought": "Generate code, then self-critique and refine it iteratively to catch bugs before submission.",
    "name": "Self-Refine Code Generation",
    "code": """def forward(self, taskInfo):
    solve_instruction = "Write a complete, correct code solution for the given programming problem. Output only the code."
    coder = LLMAgentBase(['code'], 'Coder Agent', temperature=0.3)
    code, = coder([taskInfo], solve_instruction)

    for i in range(2):
        review_instruction = "Review the code solution for correctness, edge cases, and potential bugs. If you find issues, provide an improved version. If the code is correct, return it unchanged."
        reviewer = LLMAgentBase(['review', 'code'], 'Reviewer Agent', temperature=0.2)
        review, code = reviewer([taskInfo, code], review_instruction, iteration_idx=i)

    return code
"""
}

Debate_Code = {
    "thought": "Multiple agents propose solutions and debate their correctness, with a judge picking the best.",
    "name": "Multi-Agent Code Debate",
    "code": """def forward(self, taskInfo):
    solve_instruction = "Write a complete, correct code solution for the given programming problem. Explain your approach briefly."
    N = 3
    coders = [LLMAgentBase(['approach', 'code'], f'Coder {i}', temperature=0.7) for i in range(N)]

    all_outputs = []
    for coder in coders:
        approach, code = coder([taskInfo], solve_instruction)
        all_outputs.extend([approach, code])

    judge_instruction = "Given multiple proposed solutions, select the most correct and complete one. Return only the chosen code."
    judge = LLMAgentBase(['code'], 'Judge Agent', temperature=0.1)
    code, = judge([taskInfo] + all_outputs, judge_instruction)
    return code
"""
}


def get_init_archive():
    return [DirectCode, COT_Code, SC_Code, Reflexion_Code, Debate_Code]


def get_prompt(archive):
    archive_str = ",\n".join([json.dumps(sol) for sol in archive])
    prompt = f"""You are an expert at designing agentic systems for competitive programming and software engineering tasks (FullStackBench).

# Archive of Discovered Agents
[{archive_str}]

# Task Description
- Each problem is a hard programming challenge in various languages (Python, Go, Java, etc.).
- The taskInfo content is formatted as: "Programming Language: <lang>\\n\\nProblem:\\n<description>"
- The agent's forward() must return the raw code string. SandboxFusion will extract and execute it.
- Do NOT pre-extract code — return the full model output directly.
- Agents are built from LLMAgentBase objects.

# What You Should Do
1. Analyze the archive and identify patterns that work well for code generation.
2. Design a **novel** agent that improves over existing ones.
3. Return JSON with fields: "thought", "name", "code".

The "code" must define exactly one Python function: `def forward(self, taskInfo)`.
Available: `LLMAgentBase`, `Info`, standard Python libraries.

{json.dumps(EXAMPLE)}
"""
    system_prompt = "You are an expert AI researcher designing novel agentic systems for code generation tasks."
    return system_prompt, prompt


def get_reflexion_prompt(last_solution):
    prompt1 = """Reflect on your proposed agent design. Consider:
1. Does the agent correctly handle multi-language problems?
2. Are there prompt improvements that could reduce syntax errors or logic bugs?
3. Would a different multi-agent strategy (debate, self-repair, etc.) work better?

Provide an improved design. Return the same JSON format: {"thought": ..., "name": ..., "code": ...}"""

    if last_solution:
        prompt2 = f"""The previous best solution was:
{json.dumps(last_solution)}

With fitness: {last_solution.get('fitness', 'unknown')}

Design an agent that addresses its weaknesses. Return JSON: {{"thought": ..., "name": ..., "code": ...}}"""
    else:
        prompt2 = """Design a final, polished version of your agent. Return JSON: {"thought": ..., "name": ..., "code": ...}"""

    return prompt1, prompt2
