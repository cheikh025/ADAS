import json
import random
import string
from math import isclose

import numpy as np
import regex


def random_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


# ── Answer extraction (identical to MAS_pro) ─────────────────────────────────

def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    left = "\\boxed{"
    assert s[:len(left)] == left and s[-1] == "}"
    return s[len(left):-1]


def extract_answer(text: str) -> str:
    boxed = _last_boxed_only_string(text)
    if boxed:
        try:
            return _remove_boxed(boxed)
        except Exception:
            pass
    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = regex.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""


# ── String normalization (MAS_pro strategy) ───────────────────────────────────

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _parse_digits(num) -> float | None:
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def _try_parse_numeric(s) -> float | None:
    """Parse a numeric value, also handling LaTeX \\frac{a}{b} patterns."""
    val = _parse_digits(s)
    if val is not None:
        return val
    m = regex.match(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', str(s).strip())
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def math_equal(prediction, reference) -> bool:
    """
    Two-stage comparison matching MAS_pro strategy:
      1. Symbolic string equivalence after LaTeX normalization (_strip_string)
      2. Numeric comparison with abs_tol=1e-5 (also handles \\frac{a}{b})
    No SymPy — ensures identical scoring to MAS_pro.
    """
    try:
        if _strip_string(str(prediction)) == _strip_string(str(reference)):
            return True
    except Exception:
        if str(prediction) == str(reference):
            return True
    try:
        p = _try_parse_numeric(prediction)
        r = _try_parse_numeric(reference)
        if p is not None and r is not None:
            return isclose(p, r, abs_tol=1e-5)
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
