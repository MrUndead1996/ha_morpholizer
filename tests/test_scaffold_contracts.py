"""Lightweight repository scaffold contracts; these are not HA runtime tests."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "morpholizer"


def test_integration_manifest_declares_expected_domain_and_minimum_ha() -> None:
    manifest_path = INTEGRATION / "manifest.json"
    assert manifest_path.is_file(), "custom_components/morpholizer/manifest.json is required"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["domain"] == "morpholizer"
    assert manifest["name"] == "Morpholizer"
    assert manifest["version"]
    assert isinstance(manifest.get("requirements", []), list)
    minimum_ha = manifest.get("homeassistant")
    if minimum_ha is not None:
        assert tuple(map(int, minimum_ha.split("."))) <= (2026, 11, 1)


def test_english_and_russian_translation_keys_match() -> None:
    translations = INTEGRATION / "translations"
    english_path = translations / "en.json"
    russian_path = translations / "ru.json"
    assert english_path.is_file(), "English translation is required"
    assert russian_path.is_file(), "Russian translation is required"

    def leaf_paths(value: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        if isinstance(value, dict):
            return {
                path for key, child in value.items() for path in leaf_paths(child, (*prefix, key))
            }
        return {prefix}

    english = leaf_paths(json.loads(english_path.read_text(encoding="utf-8")))
    russian = leaf_paths(json.loads(russian_path.read_text(encoding="utf-8")))
    assert english == russian


def test_integration_python_sources_compile() -> None:
    sources = sorted(INTEGRATION.rglob("*.py"))
    assert sources, "Integration Python sources are required"
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")


def test_manifest_has_no_placeholder_urls_or_owners() -> None:
    manifest_path = INTEGRATION / "manifest.json"
    assert manifest_path.is_file(), "custom_components/morpholizer/manifest.json is required"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("documentation", "issue_tracker", "codeowners"):
        value = manifest.get(field)
        if isinstance(value, str):
            assert not re.search(
                r"example\.(com|org)|your[-_ ]?(name|user|org)", value, re.IGNORECASE
            )
