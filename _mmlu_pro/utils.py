import random
import string
from collections import namedtuple

import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def format_multichoice_question(row):
    options = ""
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        val = row.get(letter)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            options += f"\n({letter}) {val}"
    return f"Answer the following multiple choice question.\n\n{row['Question']}{options}"


def random_id(length=4):
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.

    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages.
    """
    data = np.array(data)
    bootstrap_means = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    bootstrap_means = np.array(bootstrap_means)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)
    median = np.median(bootstrap_means)

    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
