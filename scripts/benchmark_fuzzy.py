"""Benchmark fuzzy normalization across synthetic catalog sizes.

Run from the repository root, for example:
    python -m scripts.benchmark_fuzzy --sizes 100 1000 5000 --iterations 30

This is a performance/behavior probe, not an accuracy corpus. It reports when
the normalizer's configured safety ceilings prevent a fuzzy scan; such a run
must not be interpreted as a fast successful match.
"""

import argparse
import json
import math
import platform
from statistics import median
from time import perf_counter

from custom_components.morpholizer.matching import (
    MAX_CANDIDATES,
    MAX_COMPARISONS,
    MAX_QUERY_TOKENS,
    MAX_TEXT_CHARS,
    FuzzySettings,
)
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer

TARGET = "Стиральная машинка"
TYPO_QUERY = "Включи стиралная машинка"


def _catalog(size: int, *, ambiguous: bool = False) -> tuple[NameCandidate, ...]:
    candidates = [NameCandidate("switch.washer", TARGET)]
    if ambiguous:
        # Same observed name on a different entity must remain unresolved.
        candidates.append(NameCandidate("switch.washer_guest", TARGET))
    candidates.extend(
        NameCandidate(f"light.fixture_{i}", f"Светильник {i}")
        for i in range(size - len(candidates))
    )
    return tuple(candidates)


def _percentile95(samples: list[float]) -> float:
    return sorted(samples)[max(0, math.ceil(0.95 * len(samples)) - 1)]


def _as_ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _decision_summary(result: object) -> list[dict[str, object]]:
    return [
        {
            "reason": decision.reason,
            "method": decision.method,
            "source": decision.source,
            "score": decision.score,
            "runner_up_score": decision.runner_up_score,
            "candidate_entity_ids": [candidate.entity_id for candidate in decision.candidates],
        }
        for decision in result.decisions
    ]


def _workloads() -> tuple[tuple[str, str, bool, str], ...]:
    return (
        ("typo_hit", TYPO_QUERY, False, "fuzzy_replacement"),
        ("no_hit", "Включи телевизор", False, "unchanged"),
        ("ambiguous", TYPO_QUERY, True, "unchanged_ambiguous"),
    )


def _run_case(
    *,
    size: int,
    iterations: int,
    enabled: bool,
    workload: str,
    query: str,
    ambiguous: bool,
    expected: str,
) -> None:
    candidates = _catalog(size, ambiguous=ambiguous)
    before_morphology = perf_counter()
    morphology = Morphology()
    morphology_init_ms = _as_ms(perf_counter() - before_morphology)
    normalizer = Normalizer(morphology, fuzzy=FuzzySettings(enabled=enabled))

    first_started = perf_counter()
    first_result = normalizer.normalize(query, candidates)
    first_ms = _as_ms(perf_counter() - first_started)

    samples: list[float] = []
    last_result = first_result
    for _ in range(iterations):
        started = perf_counter()
        last_result = normalizer.normalize(query, candidates)
        samples.append(perf_counter() - started)

    token_count = len(query.split())
    budget_skipped = enabled and (
        len(query) > MAX_TEXT_CHARS
        or token_count > MAX_QUERY_TOKENS
        or len(candidates) > MAX_CANDIDATES
        or token_count * len(candidates) > MAX_COMPARISONS
    )
    replacements = [
        {
            "reason": decision.reason,
            "method": decision.method,
            "source": decision.source,
            "replacement": decision.candidates[0].name if decision.candidates else None,
            "candidate_entity_ids": [candidate.entity_id for candidate in decision.candidates],
        }
        for decision in last_result.replacements
    ]
    if budget_skipped:
        outcome = "budget_skipped"
    elif workload == "typo_hit" and enabled:
        outcome = "matched" if last_result.text != query else "no_replacement"
    elif workload == "ambiguous" and enabled:
        outcome = (
            "ambiguous_preserved"
            if any(
                d.reason == "ambiguous_candidates" and d.method == "fuzzy"
                for d in last_result.decisions
            )
            else "no_replacement"
        )
    else:
        outcome = "unchanged" if last_result.text == query else "unexpected_replacement"

    print(
        json.dumps(
            {
                "catalog_size": size,
                "workload": workload,
                "fuzzy_enabled": enabled,
                "iterations": iterations,
                "morphology_init_ms": morphology_init_ms,
                "first_query_ms": first_ms,
                "warm_median_ms": _as_ms(median(samples)),
                "warm_p95_ms": _as_ms(_percentile95(samples)),
                "expected_behavior": expected,
                "outcome": outcome,
                "input": query,
                "output": last_result.text,
                "replacements": replacements,
                "decisions": _decision_summary(last_result),
                "budget": {
                    "skipped_by_configured_ceiling": budget_skipped,
                    "candidate_limit": MAX_CANDIDATES,
                    "comparison_limit": MAX_COMPARISONS,
                    "estimated_token_candidate_pairs": token_count * len(candidates),
                },
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if any(size < 2 or size > 10000 for size in args.sizes):
        parser.error("each catalog size must be between 2 and 10000")
    if not 1 <= args.iterations <= 10000:
        parser.error("iterations must be between 1 and 10000")

    print(
        json.dumps(
            {
                "benchmark": "fuzzy_normalizer",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "iterations": args.iterations,
                "sizes": args.sizes,
            },
            ensure_ascii=False,
        )
    )
    for size in args.sizes:
        for workload, query, ambiguous, expected in _workloads():
            for enabled in (False, True):
                _run_case(
                    size=size,
                    iterations=args.iterations,
                    enabled=enabled,
                    workload=workload,
                    query=query,
                    ambiguous=ambiguous,
                    expected=expected,
                )


if __name__ == "__main__":
    main()
