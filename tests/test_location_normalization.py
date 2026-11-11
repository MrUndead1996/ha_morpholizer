"""Pure regression coverage for named area/floor targets."""

from itertools import permutations

import pytest

pytest.importorskip("pymorphy3")
pytest.importorskip("rapidfuzz")

from custom_components.morpholizer.matching import FuzzySettings
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import NameCandidate, Normalizer


@pytest.fixture(scope="module")
def normalizer() -> Normalizer:
    return Normalizer(Morphology(), fuzzy=FuzzySettings(enabled=True))


@pytest.mark.parametrize(
    ("kind", "name", "source", "expected"),
    [
        ("area", "Кухня", "Включи свет в кухне", "Включи свет в Кухня"),
        ("area", "Спальня", "Температура в спальне", "Температура в Спальня"),
        ("floor", "Второй этаж", "Выключи свет на втором этаже", "Выключи свет на Второй этаж"),
        ("floor", "Верхний этаж", "Свет на верхнем этаже", "Свет на Верхний этаж"),
        ("area", "Гостевая комната", "Свет в гостеваая комната", "Свет в Гостевая комната"),
        ("floor", "Мансардный этаж", "Свет на мансарднном этаже", "Свет на Мансардный этаж"),
    ],
)
def test_location_forms_and_typos(normalizer, kind, name, source, expected) -> None:
    targets = (NameCandidate("target", name, kind=kind),)
    result = normalizer.normalize(source, targets)
    assert result.text == expected
    assert result.replacements
    assert all(source[slice(*d.span)] == d.source for d in result.replacements)
    assert normalizer.normalize(expected, targets).text == expected


def test_multiple_target_types_share_original_offsets(normalizer) -> None:
    names = (
        NameCandidate("light.a", "Настольная лампа"),
        NameCandidate("bedroom", "Спальня", kind="area"),
        NameCandidate("upper", "Второй этаж", kind="floor"),
    )
    source = "Не включай настольную лампу в спальне на втором этаже"
    result = normalizer.normalize(source, names)
    assert result.text == "Не включай Настольная лампа в Спальня на Второй этаж"
    assert len(result.replacements) == 3
    assert [d.candidates[0].kind for d in result.replacements] == ["entity", "area", "floor"]
    assert all(source[slice(*d.span)] == d.source for d in result.replacements)


@pytest.mark.parametrize("source", ["в гостевой комнате", "в гостеваая комната"])
def test_same_id_in_different_registries_is_still_ambiguous(normalizer, source) -> None:
    targets = (
        NameCandidate("same", "Гостевая комната", kind="area"),
        NameCandidate("same", "Гостевая комната", kind="floor"),
    )
    for ordered in permutations(targets):
        result = normalizer.normalize(source, ordered)
        assert result.text == source
        assert any(d.reason == "ambiguous_candidates" for d in result.decisions)


def test_entity_and_area_collision_abstains(normalizer) -> None:
    names = (
        NameCandidate("light.bedroom", "Спальня"),
        NameCandidate("bedroom", "Спальня", kind="area"),
    )
    assert normalizer.normalize("в спальне", names).text == "в спальне"


def test_candidate_cache_includes_registry_kind(normalizer) -> None:
    for kind in ("area", "floor", "entity", "area"):
        result = normalizer.normalize("в спальне", (NameCandidate("same", "Спальня", kind=kind),))
        assert result.replacements[0].candidates[0].kind == kind


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("на первом этаже", "Второй этаж"),
        ("на этаже 1", "Этаж 2"),
        ("в кухне 1", "Кухня 2"),
        ("на этаж 01", "Этаж 1"),
    ],
)
def test_distinct_location_numbers_are_preserved(normalizer, source, name) -> None:
    assert (
        normalizer.normalize(source, (NameCandidate("floor", name, kind="floor"),)).text == source
    )
