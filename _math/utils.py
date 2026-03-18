import json
import random
import string
from math import isclose

import numpy as np
import regex


def random_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract the last \\boxed{...} expression, or fall back to last sentence."""
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    matches = regex.findall(pattern, text, regex.DOTALL)
    if matches:
        return matches[-1].strip()
    # fallback: last sentence
    sentences = [s.strip() for s in regex.split(r"(?<!\d)[.!?]\s+", text) if s.strip()]
    return sentences[-1] if sentences else ""


def _parse_digits(num) -> float | None:
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1].rstrip("\\")
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def _symbolic_equal(a, b) -> bool:
    try:
        from sympy import simplify, N
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr

        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        a_p, b_p = _parse(a), _parse(b)
        try:
            if simplify(a_p - b_p) == 0:
                return True
        except Exception:
            pass
        try:
            if isclose(float(N(a_p)), float(N(b_p)), abs_tol=1e-3):
                return True
        except Exception:
            pass
    except ImportError:
        pass
    return False


def math_equal(prediction, reference) -> bool:
    if str(prediction) == str(reference):
        return True
    if str(prediction).replace(" ", "") == str(reference).replace(" ", ""):
        return True
    a_d = _parse_digits(prediction)
    b_d = _parse_digits(reference)
    if a_d is not None and b_d is not None:
        try:
            if isclose(a_d, b_d, abs_tol=1e-3):
                return True
        except Exception:
            pass
    try:
        if _symbolic_equal(prediction, reference):
            return True
    except Exception:
        pass
    return False


def score_math(solution: str, prediction: str) -> bool:
    """Compare prediction against ground-truth solution via \\boxed{} extraction."""
    gt = extract_answer(solution)
    pred = extract_answer(str(prediction))
    return math_equal(pred, gt)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_math_examples(file_path: str) -> list[dict]:
    """Load JSONL file. Each row: {subject, problem, solution, level, type}."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── Statistics ────────────────────────────────────────────────────────────────

def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    data = np.array(data)
    bootstrap_means = np.mean(
        np.random.choice(data, size=(num_bootstrap_samples, len(data)), replace=True),
        axis=1
    )
    lower = (1.0 - confidence_level) / 2.0
    ci_lower = np.percentile(bootstrap_means, lower * 100) * 100
    ci_upper = np.percentile(bootstrap_means, (1 - lower) * 100) * 100
    median = np.median(bootstrap_means) * 100
    return f"95% Bootstrap Confidence Interval: ({ci_lower:.1f}%, {ci_upper:.1f}%), Median: {median:.1f}%"
