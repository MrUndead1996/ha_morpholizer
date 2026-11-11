#!/usr/bin/env python3
"""Evaluate the frozen synthetic fuzzy corpus without changing its thresholds.

These curated typo/STT-like cases are not real user STT data (no recordings
were available). Holdout is reported separately and must not be used to tune.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_components.morpholizer.matching import FuzzySettings
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer


def evaluate(name: str, filename: str, normalizer: Normalizer) -> None:
    data = json.loads((ROOT / "tests" / "fixtures" / filename).read_text(encoding="utf-8"))
    cases = data["cases"]
    correct_replacements = correct_refusals = false_replacements = 0
    abstentions = exact_text_matches = eligible_positive = 0
    fuzzy_replacements = 0
    for case in cases:
        result = normalizer.normalize(
            case["input"], tuple(NameCandidate(*row) for row in case["candidates"])
        )
        changed = result.text != case["input"]
        fuzzy_replacements += sum(d.method == "fuzzy" for d in result.replacements)
        # expected text is authoritative; eligible distinguishes intended
        # positive correction opportunities from expected no-op examples.
        truth_positive = bool(case["eligible"] and case["expected"] != case["input"])
        if result.text == case["input"]:
            exact_text_matches += 1
        if truth_positive:
            eligible_positive += 1
            if result.text == case["expected"]:
                correct_replacements += 1
            elif not changed:
                abstentions += 1
            else:
                false_replacements += 1
        elif changed:
            false_replacements += 1
        else:
            correct_refusals += 1
    attempted_replacements = correct_replacements + false_replacements
    precision = correct_replacements / attempted_replacements if attempted_replacements else 0.0
    recall = correct_replacements / eligible_positive if eligible_positive else 0.0
    print(
        f"{name}: size={len(cases)} correct_replacements={correct_replacements} "
        f"correct_refusals={correct_refusals} false_replacements={false_replacements} "
        f"abstentions={abstentions} exact_text_matches={exact_text_matches} "
        f"precision={precision:.3f} ({correct_replacements}/{attempted_replacements}) "
        f"recall={recall:.3f} ({correct_replacements}/{eligible_positive}) "
        f"eligible_positives={eligible_positive}"
        f" fuzzy_span_replacements={fuzzy_replacements} (metrics above cover the full pipeline)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cutoff", type=float, default=90.0)
    parser.add_argument("--margin", type=float, default=8.0)
    parser.add_argument("--split", choices=("development", "holdout", "both"), default="both")
    args = parser.parse_args()
    normalizer = Normalizer(
        Morphology(),
        fuzzy=FuzzySettings(enabled=True, score_cutoff=args.score_cutoff, margin=args.margin),
    )
    if args.split in ("development", "both"):
        evaluate("development", "fuzzy_development.json", normalizer)
    if args.split in ("holdout", "both"):
        evaluate("holdout (report only; do not tune)", "fuzzy_holdout.json", normalizer)


if __name__ == "__main__":
    main()
