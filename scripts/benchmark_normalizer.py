"""Synthetic benchmark: python -m scripts.benchmark_normalizer --entities 1000."""

import argparse
import json
import os
import resource
from pathlib import Path
from statistics import median
from time import perf_counter

from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer


def _rss_mib() -> float | None:
    """Current resident memory on Linux (not the lifetime peak)."""
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    pages = int(statm.read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.entities <= 10000 or not 1 <= args.iterations <= 10000:
        parser.error("entities and iterations must be between 1 and 10000")

    before = _rss_mib()
    start = perf_counter()
    normalizer = Normalizer(Morphology())
    load_ms = (perf_counter() - start) * 1000
    after = _rss_mib()
    candidates = (NameCandidate("switch.washer", "Стиральная машинка"),) + tuple(
        NameCandidate(f"light.lamp_{i}", f"Лампа {i}") for i in range(args.entities - 1)
    )
    query = "Что со стиральной машинкой?"
    start = perf_counter()
    result = normalizer.normalize(query, candidates)
    first_ms = (perf_counter() - start) * 1000
    assert result.text == "Что со Стиральная машинка?", result.text
    samples = []
    for _ in range(args.iterations):
        start = perf_counter()
        result = normalizer.normalize(query, candidates)
        samples.append((perf_counter() - start) * 1000)
    ordered = sorted(samples)
    print(
        json.dumps(
            {
                "entities": args.entities,
                "iterations": args.iterations,
                "cold_load_ms": round(load_ms, 3),
                "rss_before_mib": before,
                "rss_after_load_mib": after,
                "rss_after_queries_mib": _rss_mib(),
                "first_query_ms": round(first_ms, 3),
                "warm_median_ms": round(median(samples), 3),
                "warm_p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
                "peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
