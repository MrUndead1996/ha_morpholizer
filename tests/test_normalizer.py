"""Regression corpus for conservative Russian entity-name normalization."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("pymorphy3")

from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "normalization.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def normalizer() -> Normalizer:
    return Normalizer(Morphology())


@pytest.mark.parametrize("case", CORPUS, ids=lambda case: case["id"])
def test_normalization_corpus(normalizer: Normalizer, case: dict[str, object]) -> None:
    candidates = tuple(NameCandidate(*candidate) for candidate in case["candidates"])

    result = normalizer.normalize(case["input"], candidates)

    assert result.original_text == case["input"]
    assert result.text == case["expected"]
    assert len(result.replacements) == case["replacement_count"]
    if "reason" in case:
        assert [decision.reason for decision in result.decisions] == [case["reason"]]
        assert result.decisions[0].method == "morphology"


def test_morphology_replacement_has_original_span_and_source(normalizer: Normalizer) -> None:
    original = "Что со стиральной машинкой?"
    result = normalizer.normalize(original, (NameCandidate("switch.washer", "Стиральная машинка"),))

    decision = result.replacements[0]
    start, end = decision.span
    assert original[start:end] == decision.source == "стиральной машинкой"
    assert decision.method == "morphology"
    assert decision.reason == "matched"
    assert decision.candidates == (NameCandidate("switch.washer", "Стиральная машинка"),)


@pytest.mark.parametrize("case", CORPUS, ids=lambda case: case["id"])
def test_normalized_corpus_is_idempotent(normalizer: Normalizer, case: dict[str, object]) -> None:
    candidates = tuple(NameCandidate(*candidate) for candidate in case["candidates"])
    first = normalizer.normalize(case["input"], candidates)
    second = normalizer.normalize(first.text, candidates)

    assert second.text == first.text
    assert second.replacements == ()


def test_word_cache_is_bounded_and_owned_by_the_analyzer() -> None:
    morphology = Morphology()
    with patch.object(morphology._analyzer, "parse", wraps=morphology._analyzer.parse) as parse:
        first = morphology.analyze("лампой")
        assert morphology.analyze("лампой") is first
        parse.assert_called_once()
    assert morphology._cached_analyze.cache_info().maxsize == 4096
    assert Morphology()._cached_analyze.cache_info().currsize == 0


def test_concurrent_candidate_generations_do_not_mix(normalizer: Normalizer) -> None:
    def run(index: int) -> str:
        name = "Лампа" if index % 2 else "Лампы"
        candidates = (NameCandidate(f"light.entity_{index % 2}", name),)
        return normalizer.normalize("Включи лампой", candidates).text

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(run, range(20)))
    assert outputs == ["Включи Лампа" if i % 2 else "Включи Лампы" for i in range(20)]


def test_empty_catalog_never_uses_morphology(normalizer: Normalizer) -> None:
    with patch.object(normalizer._morphology, "analyze") as analyze:
        result = normalizer.normalize("Не включай ничего, 25!", ())
    analyze.assert_not_called()
    assert result.text == result.original_text
    assert not result.replacements
