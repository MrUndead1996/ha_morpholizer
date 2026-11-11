"""Independent safety regressions for fuzzy entity selection."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

pytest.importorskip("pymorphy3")
pytest.importorskip("rapidfuzz")

from custom_components.morpholizer.matching import FuzzySettings
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer


@pytest.fixture(scope="module")
def fuzzy() -> Normalizer:
    return Normalizer(Morphology(), fuzzy=FuzzySettings(enabled=True))


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Вкллючи свет", "Включи"),
        ("Не выключй свет", "Выключи"),
        ("Включи термостта 2", "Термостат"),
        ("Включи термостта Xiaomi", "Термостат"),
        ("Включи термостта-02", "Термостат-2"),
        ("Включи термостта Xiaomi2", "Термостат Xiaomi3"),
        ("Включи машнку", "Машинка"),
        ("Включи лампа 1", "Лампа 2"),
        ("Включи лампа", "Лампа спальни"),
        ("Включи машину", "Машинка"),
    ],
)
def test_dangerous_or_underspecified_mentions_are_not_guessed(
    fuzzy: Normalizer, text: str, name: str
) -> None:
    result = fuzzy.normalize(text, (NameCandidate("switch.target", name),))
    assert result.text == text
    assert not result.replacements


def test_candidate_order_never_resolves_independent_targets(fuzzy: Normalizer) -> None:
    candidates = (
        NameCandidate("switch.a", "Стиральная машинка"),
        NameCandidate("switch.b", "Стиральная машинка"),
        NameCandidate("switch.a", "Стиральная машина"),
    )
    text = "Включи стриальная машинка"
    for ordered in permutations(candidates):
        assert fuzzy.normalize(text, ordered).text == text


def test_disabled_fuzzy_has_identical_results_to_morphology() -> None:
    morphology = Morphology()
    plain = Normalizer(morphology)
    disabled = Normalizer(morphology, fuzzy=FuzzySettings(enabled=False))
    names = (NameCandidate("switch.washer", "Стиральная машинка"),)
    for text in ("Включи стиральную машинку", "Включи стиральная машынка", "Что нового?"):
        assert disabled.normalize(text, names) == plain.normalize(text, names)


def test_settings_are_immutable() -> None:
    settings = FuzzySettings(enabled=True)
    with pytest.raises(FrozenInstanceError):
        settings.enabled = False


def test_concurrent_catalogs_do_not_leak_targets(fuzzy: Normalizer) -> None:
    text = "Включи стиральня машинка"

    def normalize(index: int) -> str:
        names = (NameCandidate("switch.washer", "Стиральная машинка"),) if index % 2 else ()
        return fuzzy.normalize(text, names).text

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(normalize, range(24)))
    assert results == ["Включи Стиральная машинка" if i % 2 else text for i in range(24)]


def test_aliases_of_one_entity_do_not_compete(fuzzy: Normalizer) -> None:
    names = (
        NameCandidate("switch.a", "Стиральная машинка"),
        NameCandidate("switch.a", "Стиральная машинкаа"),
    )
    for ordered in permutations(names):
        result = fuzzy.normalize("стиралная машинка", ordered)
        assert result.text == "Стиральная машинка"
        assert result.replacements[0].runner_up_score is None


def test_crossing_fuzzy_mentions_abstain_independent_of_order(fuzzy: Normalizer) -> None:
    names = (
        NameCandidate("switch.a", "Стиральная машинка"),
        NameCandidate("switch.b", "Машинка духовка"),
    )
    text = "стиралная машинка духвка"
    for ordered in permutations(names):
        result = fuzzy.normalize(text, ordered)
        assert result.text == text
        assert any(d.reason == "overlapping_matches" for d in result.decisions)


def test_runner_below_threshold_still_blocks_small_margin(fuzzy: Normalizer) -> None:
    result = fuzzy.normalize(
        "стиралная машинка",
        (
            NameCandidate("switch.a", "Стиральная машинка"),
            NameCandidate("switch.b", "Стиральная машинкаа"),
        ),
    )
    assert result.text == result.original_text
    decision = next(d for d in result.decisions if d.method == "fuzzy")
    assert decision.reason == "ambiguous_candidates"
    assert decision.runner_up_score < 90 <= decision.score


@pytest.mark.parametrize("kind", ["catalog", "query"])
def test_resource_limits_are_explained(fuzzy: Normalizer, kind: str) -> None:
    from custom_components.morpholizer.matching import MAX_CANDIDATES, MAX_TEXT_CHARS

    names = (NameCandidate("switch.a", "Стиральная машинка"),)
    text = "стиралная машинка"
    if kind == "catalog":
        names += tuple(
            NameCandidate(f"light.{i}", f"Светильник {i}") for i in range(MAX_CANDIDATES)
        )
    else:
        text += " " * MAX_TEXT_CHARS
    result = fuzzy.normalize(text, names)
    assert result.text == text
    assert any(d.reason == "resource_limit" and d.method == "fuzzy" for d in result.decisions)
