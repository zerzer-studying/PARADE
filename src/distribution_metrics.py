"""Distribution-level metrics for ordinal survey responses.

The metrics are computed per (domain, question) and macro-averaged so that
questions with more respondents do not dominate the result. Predictions are
hard choices; their empirical frequencies across respondents form the model
distribution.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy / math.log2(len(probabilities))


def _kl_divergence(
    target: list[float], prediction: list[float], epsilon: float
) -> float:
    smoothed_prediction = [max(value, epsilon) for value in prediction]
    normalizer = sum(smoothed_prediction)
    smoothed_prediction = [value / normalizer for value in smoothed_prediction]
    return sum(
        p * math.log2(p / q)
        for p, q in zip(target, smoothed_prediction)
        if p > 0
    )


def _js_divergence(target: list[float], prediction: list[float]) -> float:
    midpoint = [(p + q) / 2.0 for p, q in zip(target, prediction)]

    def kl_to_midpoint(values: list[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0
        )

    return 0.5 * (kl_to_midpoint(target) + kl_to_midpoint(prediction))


def _normalized_emd(target: list[float], prediction: list[float]) -> float:
    if len(target) <= 1:
        return 0.0
    target_cdf = 0.0
    prediction_cdf = 0.0
    distance = 0.0
    for p, q in zip(target[:-1], prediction[:-1]):
        target_cdf += p
        prediction_cdf += q
        distance += abs(target_cdf - prediction_cdf)
    return distance / (len(target) - 1)


def compute_distribution_metrics(
    records: Iterable[Mapping], *, epsilon: float = 1e-8
) -> dict[str, float | int]:
    """Compute macro distribution metrics from survey prediction records.

    Each record must contain ``question_id``, ``target`` and ``prediction``.
    ``domain`` and ``valid_options`` are optional. Invalid predictions and
    response codes outside the caller's ordinal core are excluded from these
    metrics and reflected by the caller's existing sample-level metrics.
    """

    groups: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for record in records:
        if record.get("include_in_distribution") is False:
            continue
        prediction = record.get("prediction")
        if prediction is None:
            continue
        key = (
            str(record.get("domain", "")),
            str(record.get("question_id", "")),
        )
        groups[key].append(record)

    per_question = []
    sample_count = 0
    for key, items in groups.items():
        options = set()
        for item in items:
            options.update(int(value) for value in item.get("valid_options", ()))
            options.add(int(item["target"]))
            options.add(int(item["prediction"]))
        ordered_options = sorted(options)
        if not ordered_options:
            continue

        target_counts = {option: 0 for option in ordered_options}
        prediction_counts = {option: 0 for option in ordered_options}
        for item in items:
            target_counts[int(item["target"])] += 1
            prediction_counts[int(item["prediction"])] += 1

        count = len(items)
        sample_count += count
        target = [target_counts[option] / count for option in ordered_options]
        prediction = [
            prediction_counts[option] / count for option in ordered_options
        ]
        target_entropy = _normalized_entropy(target)
        prediction_entropy = _normalized_entropy(prediction)
        per_question.append(
            {
                "domain": key[0],
                "question_id": key[1],
                "sample_count": count,
                "js_divergence": _js_divergence(target, prediction),
                "kl_divergence": _kl_divergence(target, prediction, epsilon),
                "target_entropy": target_entropy,
                "prediction_entropy": prediction_entropy,
                "entropy": abs(target_entropy - prediction_entropy),
                "emd": _normalized_emd(target, prediction),
            }
        )

    metric_keys = (
        "js_divergence",
        "kl_divergence",
        "target_entropy",
        "prediction_entropy",
        "entropy",
        "emd",
    )
    if not per_question:
        return {
            **{key: 0.0 for key in metric_keys},
            "distribution_group_count": 0,
            "distribution_sample_count": 0,
        }

    return {
        **{
            key: sum(float(item[key]) for item in per_question)
            / len(per_question)
            for key in metric_keys
        },
        "distribution_group_count": len(per_question),
        "distribution_sample_count": sample_count,
    }
