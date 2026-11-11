"""Adversarial regression tests for opt-in fuzzy entity-name matching."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("rapidfuzz")
pytest.importorskip("pymorphy3")

from custom_components.morpholizer.matching import FuzzySettings
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer

FIXTURES = Path(__file__).parent / "fixtures"
DEVELOPMENT = json.loads((FIXTURES / "fuzzy_development.json").read_text(encoding="utf-8"))["cases"]
HOLDOUT = json.loads((FIXTURES / "fuzzy_holdout.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def fuzzy_normalizer() -> Normalizer:
    return Normalizer(
        Morphology(), fuzzy=FuzzySettings(enabled=True, score_cutoff=90.0, margin=8.0)
    )


@pytest.mark.parametrize("case", DEVELOPMENT + HOLDOUT, ids=lambda case: case["id"])
def test_adversarial_fuzzy_corpora(fuzzy_normalizer: Normalizer, case: dict) -> None:
    candidates = tuple(NameCandidate(*row) for row in case["candidates"])
    result = fuzzy_normalizer.normalize(case["input"], candidates)
    assert result.original_text == case["input"]
    if case["eligible"] and case["expected"] != case["input"]:
        # A conservative matcher may abstain on a difficult positive. It must
        # never produce a third, unsupported transformation.
        assert result.text in (case["input"], case["expected"])
    else:
        assert result.text == case["expected"]


def test_holdout_is_frozen() -> None:
    expected = (FIXTURES / "fuzzy_holdout.sha256").read_text().split()[0]
    assert hashlib.sha256((FIXTURES / "fuzzy_holdout.json").read_bytes()).hexdigest() == expected


def test_fuzzy_is_disabled_by_default() -> None:
    normalizer = Normalizer(Morphology())
    result = normalizer.normalize(
        "стиралная машинка", (NameCandidate("washer", "Стиральная машинка"),)
    )
    assert result.text == result.original_text


@pytest.mark.parametrize("text", ["что со стиралной машинкой?", "Включи стиралную машинку"])
def test_typo_and_inflection_are_combined(fuzzy_normalizer: Normalizer, text: str) -> None:
    candidates = (NameCandidate("washer", "Стиральная машинка"),)
    result = fuzzy_normalizer.normalize(text, candidates)
    expected = (
        "что со Стиральная машинка?" if text.startswith("что") else "Включи Стиральная машинка"
    )
    assert result.text == expected
    assert len(result.replacements) == 1
    decision = result.replacements[0]
    assert decision.method == "fuzzy"
    assert text[slice(*decision.span)] == decision.source
    assert fuzzy_normalizer.normalize(result.text, candidates).text == expected
    assert Normalizer(Morphology()).normalize(text, candidates).text == text


def test_long_one_token_typo_records_original_span_and_is_idempotent(
    fuzzy_normalizer: Normalizer,
) -> None:
    original = "Пожалуйста, стиралная машинка"
    candidates = (NameCandidate("washer", "Стиральная машинка"),)
    result = fuzzy_normalizer.normalize(original, candidates)
    fuzzy = [d for d in result.replacements if d.method == "fuzzy"]
    assert len(fuzzy) == 1
    start, end = fuzzy[0].span
    assert original[start:end] == fuzzy[0].source == "стиралная машинка"
    assert fuzzy[0].method == "fuzzy"
    assert fuzzy_normalizer.normalize(result.text, candidates).text == result.text


def test_repeated_mentions_keep_separate_original_spans(fuzzy_normalizer: Normalizer) -> None:
    original = "стиралная машинка и стиралная машинка"
    result = fuzzy_normalizer.normalize(original, (NameCandidate("washer", "Стиральная машинка"),))
    assert result.text == "Стиральная машинка и Стиральная машинка"
    assert len(result.replacements) == 2
    assert all(original[a:b] == d.source for d in result.replacements for a, b in [d.span])
