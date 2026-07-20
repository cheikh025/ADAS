"""Search prompts and seed architectures for SciCode ADAS."""

from __future__ import annotations

import copy
import json


EXAMPLE = {
    "thought": (
        "**Insights:** What the archive suggests should be tried next.\n"
        "**Overall Idea:** Why the design should improve scientific-code accuracy.\n"
        "**Implementation:** The control flow and prompts used by the design."
    ),
    "name": "Name of the proposed agent",
    "code": """def forward(self, taskInfo):
    coder = LLMAgentBase(['thinking', 'code'], 'Scientific Coder', temperature=0.2)
    thinking, code = coder([taskInfo], 'Analyze the numerical method, then implement only the requested function.')
    return code
""",
}


INITIAL_ARCHIVE = [
    {
        "thought": "Plan the mathematics and numerical edge cases before implementing the requested function.",
        "name": "Scientific Chain-of-Thought",
        "code": """def forward(self, taskInfo):
    instruction = "Analyze the scientific problem, equations, numerical method, array shapes, units, and edge cases step by step. Then implement only the requested current function, preserving its exact signature. Return executable Python code with no imports, usage examples, or tests."
    coder = LLMAgentBase(['thinking', 'code'], 'Scientific Coder', temperature=0.2)
    thinking, code = coder([taskInfo], instruction)
    return code
""",
    },
    {
        "thought": "Generate independent implementations and let a careful verifier select the most reliable one.",
        "name": "Scientific Self-Consistency",
        "code": """def forward(self, taskInfo):
    instruction = "Independently derive the numerical method and implement only the requested function. Preserve the exact header; return code without imports, examples, or tests."
    candidates = []
    for i in range(3):
        coder = LLMAgentBase(['thinking', 'code'], 'Independent Scientific Coder', temperature=0.7)
        thinking, code = coder([taskInfo], instruction, i)
        candidates.extend([thinking, code])
    judge_instruction = "Compare the candidate implementations for mathematical correctness, numerical stability, shape handling, and compliance with the exact function header. Return the best complete implementation of only the requested function, with no imports or tests."
    judge = LLMAgentBase(['thinking', 'code'], 'Scientific Code Judge', temperature=0.1)
    thinking, code = judge([taskInfo] + candidates, judge_instruction)
    return code
""",
    },
    {
        "thought": "Use a critic to find mathematical and implementation defects, then revise the code.",
        "name": "Scientific Reflexion",
        "code": """def forward(self, taskInfo):
    coder = LLMAgentBase(['thinking', 'code'], 'Scientific Coder', temperature=0.25)
    critic = LLMAgentBase(['feedback', 'correct'], 'Numerical Methods Reviewer', temperature=0.1)
    thinking, code = coder([taskInfo], "Derive the method carefully and implement only the requested function. Preserve the exact signature and omit imports, examples, and tests.", 0)
    for i in range(2):
        feedback, correct = critic([taskInfo, thinking, code], "Audit the proposed implementation against the task. Check equations, conventions, indexing, dimensions, numerical stability, return values, and use of earlier-step functions. Put True in correct only if no defect remains.", i)
        if correct.content.strip().lower() == 'true':
            break
        thinking, code = coder([taskInfo, thinking, code, feedback], "Use the review to produce a corrected implementation of only the requested function. Return code without imports, examples, or tests.", i + 1)
    return code
""",
    },
    {
        "thought": "Separate symbolic derivation, numerical review, and implementation before synthesis.",
        "name": "Triangulated Scientific Experts",
        "code": """def forward(self, taskInfo):
    theorist = LLMAgentBase(['analysis'], 'Mathematical Physicist', temperature=0.3)
    numerics = LLMAgentBase(['analysis'], 'Numerical Methods Expert', temperature=0.3)
    implementation = LLMAgentBase(['analysis'], 'Scientific Python Expert', temperature=0.3)
    theory, = theorist([taskInfo], "Derive the exact equations and conventions needed by the current function. Identify inputs, outputs, assumptions, and invariants.")
    numerical, = numerics([taskInfo], "Analyze the stable numerical algorithm, boundary cases, array shapes, tolerances, and failure modes for the current function.")
    coding, = implementation([taskInfo], "Analyze how to implement the current function with the allowed dependencies and existing previous-step functions. Preserve the supplied interface.")
    synthesizer = LLMAgentBase(['thinking', 'code'], 'Lead Scientific Programmer', temperature=0.15)
    thinking, code = synthesizer([taskInfo, theory, numerical, coding], "Synthesize the expert analyses into the complete implementation of only the requested function. Preserve the exact function header. Output no imports, examples, or tests.")
    return code
""",
    },
    {
        "thought": "First abstract the governing principles, then turn them into precise code.",
        "name": "Scientific Step-Back",
        "code": """def forward(self, taskInfo):
    principle_agent = LLMAgentBase(['principles'], 'Scientific Principle Analyst', temperature=0.25)
    principles, = principle_agent([taskInfo], "Identify the governing equations, conventions, invariants, and numerical strategy behind the requested function. Resolve ambiguities using the supplied background and earlier steps.")
    coder = LLMAgentBase(['thinking', 'code'], 'Scientific Python Implementer', temperature=0.15)
    thinking, code = coder([taskInfo, principles], "Implement only the requested current function from the derived principles. Preserve its exact signature and expected return. Do not include imports, examples, tests, or earlier functions.")
    return code
""",
    },
    {
        "thought": "Explore meaningfully different numerical implementations and select after explicit verification.",
        "name": "Scientific Quality-Diversity",
        "code": """def forward(self, taskInfo):
    candidates = []
    approaches = [
        "Use the most direct mathematically faithful formulation.",
        "Prefer a vectorized, numerically stable formulation with careful shape handling.",
        "Use an alternative derivation and focus on boundary conditions and conventions.",
    ]
    for i, approach in enumerate(approaches):
        coder = LLMAgentBase(['thinking', 'code'], 'Diverse Scientific Coder', temperature=0.55)
        thinking, code = coder([taskInfo], approach + " Implement only the requested function with its exact signature; omit imports, examples, and tests.", i)
        candidates.extend([thinking, code])
    selector = LLMAgentBase(['thinking', 'code'], 'Scientific Verification Lead', temperature=0.1)
    thinking, code = selector([taskInfo] + candidates, "Select or repair the implementation most consistent with the equations, numerical requirements, prior-step interfaces, and expected return. Return only the current function code, with no imports or tests.")
    return code
""",
    },
    {
        "thought": "Route each step to a domain specialist, then retain a common numerical verification pass.",
        "name": "Dynamic Scientific Specialist",
        "code": """def forward(self, taskInfo):
    router = LLMAgentBase(['field'], 'Scientific Field Router', temperature=0.1)
    field, = router([taskInfo], "Classify this current subproblem as primarily mathematics, physics, materials science, or general scientific computing.")
    roles = {
        'mathematics': 'Applied Mathematician and Numerical Analyst',
        'physics': 'Computational Physicist',
        'materials': 'Computational Materials Scientist',
    }
    choice = field.content.lower()
    role = roles['mathematics'] if 'math' in choice else roles['physics'] if 'physics' in choice else roles['materials'] if 'material' in choice else 'Scientific Computing Expert'
    specialist = LLMAgentBase(['thinking', 'code'], 'Domain Specialist', role=role, temperature=0.2)
    thinking, code = specialist([taskInfo], "Derive and implement only the requested function. Preserve the exact header and return contract; omit imports, examples, tests, and earlier functions.")
    verifier = LLMAgentBase(['thinking', 'code'], 'Numerical Verification Expert', temperature=0.1)
    final_thinking, final_code = verifier([taskInfo, thinking, code], "Check the implementation for equation, convention, dimension, stability, and interface errors, and return a corrected implementation of only the current function. Include no imports or tests.")
    return final_code
""",
    },
]


SYSTEM_PROMPT = "You are an expert agent-architecture researcher. Return one well-formed JSON object."


BASE_PROMPT = r"""# Objective
Design the next ADAS agent architecture for SciCode, a benchmark of realistic scientific-computing problems in mathematics, physics, and materials science. Each benchmark item is a main problem made of sequential subproblems. The forward function is called once for the current subproblem; its taskInfo already includes the background, the supplied function header, allowed dependencies, and all earlier generated step code. The returned code is combined with those dependencies and earlier functions before the official tests run.

# Evaluation protocol
The search split is frozen to the same nine main problems used by ReMAS: three per field. Every main problem is run as a trajectory, so later steps receive the code generated for earlier steps. Tests are never provided to the agent. A subproblem passes only when all of its official tests pass. Fitness is the mean of three values: pooled subproblem pass rate for mathematics, pooled subproblem pass rate for physics, and pooled subproblem pass rate for materials science. Higher fitness is better.

# Available runtime API
The submitted architecture is executed in a module that provides:

```python
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

class LLMAgentBase:
    def __init__(self, output_fields, agent_name, role='helpful assistant', model=None, temperature=None): ...
    def __call__(self, input_infos, instruction, iteration_idx=-1):
        # returns one Info per requested output field, in the requested order
        ...

class AgentSystem:
    def forward(self, taskInfo): ...
```

An Info object should be passed directly to later agents. Read `.content` only for genuine control-flow decisions. All agents automatically receive JSON formatting instructions. A `code` output is instructed to contain one fenced Python block implementing only the requested current function, with the exact supplied header and no imports or tests.

# Architecture archive
[ARCHIVE]

# Required output
Return exactly a JSON object with keys `thought`, `name`, and `code`. `code` must contain exactly one complete `def forward(self, taskInfo): ...` function.

[EXAMPLE]

# Constraints
- Put no imports, decorators, classes, top-level assignments, type annotations requiring imports, or other top-level code outside `forward`.
- Always pass `taskInfo` to any agent that needs the problem.
- Do not construct Info objects manually. Unpack LLMAgentBase results in output-field order.
- Do not print, access files, inspect tests, call evaluation code, or return an error message.
- The final result must be a code Info object or code string for the current function only—not a plan and not the whole trajectory.
- Prompts must tell the coder to preserve the exact function signature, omit imports/tests/examples, use supplied earlier functions where appropriate, and respect allowed dependencies.
- Keep context and calls purposeful: scientific implementations can be long, and later trajectory steps already carry earlier code.

Propose a creative architecture that learns from the archive and is likely to improve field-macro subproblem pass rate.
"""


REFLECTION_ONE = """Review the proposed architecture before it is evaluated. Identify concrete failure modes for sequential SciCode steps, including loss of the exact signature, returning prose instead of code, numerical/convention mistakes, excessive context, and mishandling earlier functions. Return JSON with `reflection`, `thought`, `name`, and a corrected `code`."""

REFLECTION_TWO = """Perform one final implementation audit. The code must define exactly one top-level forward function and use only the provided runtime API. Ensure the final agent returns only a current-step implementation, never imports/tests/examples or the whole trajectory. Return JSON with `reflection`, `thought`, `name`, and the final corrected `code`."""


def get_init_archive() -> list[dict]:
    return copy.deepcopy(INITIAL_ARCHIVE)


def _archive_view(archive: list[dict], limit: int = 8) -> str:
    compact = []
    for item in archive[-limit:]:
        compact.append(
            {
                "generation": item.get("generation"),
                "name": item.get("name"),
                "thought": item.get("thought"),
                "code": item.get("code"),
                "fitness": item.get("fitness"),
                "fitness_score": item.get("fitness_score"),
            }
        )
    return json.dumps(compact, indent=2, ensure_ascii=False)


def get_prompt(archive: list[dict]) -> tuple[str, str]:
    prompt = BASE_PROMPT.replace("[ARCHIVE]", _archive_view(archive))
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE, indent=2))
    return SYSTEM_PROMPT, prompt


def get_reflexion_prompt(_previous: dict | None = None) -> tuple[str, str]:
    return REFLECTION_ONE, REFLECTION_TWO
