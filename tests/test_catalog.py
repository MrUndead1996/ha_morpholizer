"""Tests for the Home Assistant entity catalog."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components.homeassistant import exposed_entities
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import ExposedEntities
from homeassistant.config_entries import ConfigEntries, ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from custom_components.morpholizer import catalog as catalog_module
from custom_components.morpholizer.catalog import EntityCatalog


def _run(tmp_path: Path, check) -> None:
    async def run() -> None:
        hass = HomeAssistant(str(tmp_path))
        hass.config_entries = ConfigEntries(hass, {})
        try:
            await hass.config_entries.async_initialize()
            dr.async_setup(hass)
            await dr.async_load(hass, load_empty=True)
            await ar.async_load(hass, load_empty=True)
            await er.async_load(hass, load_empty=True)
            exposure = ExposedEntities(hass)
            hass.data[DATA_EXPOSED_ENTITIES] = exposure
            await exposure.async_initialize()
            await check(hass)
        finally:
            await hass.async_stop(force=True)

    asyncio.run(run())


def _entity(hass, entity_id: str, *, name: str | None = None, **kwargs):
    registry = er.async_get(hass)
    domain, _ = entity_id.split(".", 1)
    entry = registry.async_get_or_create(
        domain,
        "test",
        entity_id,
        suggested_object_id=entity_id.split(".", 1)[1],
        original_name=name,
        **kwargs,
    )
    if name:
        entry = registry.async_update_entity(entry.entity_id, name=name)
    return entry


async def _start_with_exposure(hass, catalog, exposed: set[str]) -> None:
    """Use HA's exposure API and registry options, with no store loader involved."""
    for state in hass.states.async_all():
        exposed_entities.async_expose_entity(
            hass, "conversation", state.entity_id, state.entity_id in exposed
        )
    await catalog.async_start()


def test_catalog_indexes_exposed_entity_names_and_preserves_collisions(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        first = _entity(hass, "light.first", name="Kitchen lamp")
        second = _entity(hass, "light.second", name="Kitchen lamp")
        hidden = _entity(hass, "light.hidden", name="Not exposed")
        for entity_id in (first.entity_id, second.entity_id, hidden.entity_id):
            hass.states.async_set(entity_id, "off")

        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {first.entity_id, second.entity_id})
        snapshot = catalog.snapshot

        assert {entry.entity_id for entry in snapshot.entries} == {
            first.entity_id,
            second.entity_id,
        }
        candidates = snapshot.by_name["kitchen lamp"]
        assert {candidate.entity_id for candidate in candidates} == {
            first.entity_id,
            second.entity_id,
        }
        assert hidden.entity_id not in {entry.entity_id for entry in snapshot.entries}
        with pytest.raises(AttributeError):
            snapshot.entries.append(object())
        catalog.async_stop()

    _run(tmp_path, check)


def test_catalog_requires_state_not_registry_and_keeps_unavailable(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        no_state = _entity(hass, "sensor.no_state", name="No state")
        unavailable = _entity(hass, "sensor.unavailable", name="Offline")
        missing_registry_state = "sensor.registry_missing"
        hass.states.async_set(unavailable.entity_id, "unavailable")
        hass.states.async_set(missing_registry_state, "on")

        catalog = EntityCatalog(hass)
        await _start_with_exposure(
            hass, catalog, {no_state.entity_id, unavailable.entity_id, missing_registry_state}
        )
        assert {entry.entity_id for entry in catalog.snapshot.entries} == {
            unavailable.entity_id,
            missing_registry_state,
        }
        catalog.async_stop()

    _run(tmp_path, check)


def test_rebuild_coalesces_registry_events_and_publishes_immutable_snapshot(
    tmp_path: Path,
) -> None:
    async def check(hass: HomeAssistant) -> None:
        first = _entity(hass, "light.first", name="First")
        hass.states.async_set(first.entity_id, "off")
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {first.entity_id})
        old = catalog.snapshot

        # HA registry events use action plus entity_id and changes for updated records.
        added = _entity(hass, "light.added", name="Added")
        hass.states.async_set(added.entity_id, "on")
        hass.bus.async_fire(
            er.EVENT_ENTITY_REGISTRY_UPDATED,
            {"action": "create", "entity_id": added.entity_id, "changes": {}},
        )
        await hass.async_block_till_done()

        assert added.entity_id in {entry.entity_id for entry in catalog.snapshot.entries}
        assert added.entity_id not in {entry.entity_id for entry in old.entries}
        catalog.async_stop()

    _run(tmp_path, check)


def test_entity_area_context_is_resolved_from_real_registry(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        area = ar.async_get(hass).async_create("Kitchen")
        entry = _entity(hass, "sensor.kitchen", name="Temperature")
        hass.states.async_set(entry.entity_id, "20")
        er.async_get(hass).async_update_entity(entry.entity_id, area_id=area.id)
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        record = next(
            item for item in catalog.snapshot.entries if item.entity_id == entry.entity_id
        )
        assert record.area_name == "Kitchen"
        assert record.area_id == area.id
        catalog.async_stop()

    _run(tmp_path, check)


def test_failed_rebuild_discards_stale_snapshot(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entity(hass, "light.kitchen", name="Kitchen")
        hass.states.async_set(entry.entity_id, "on")
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        assert catalog.snapshot.entries

        hass.bus.async_fire(
            er.EVENT_ENTITY_REGISTRY_UPDATED,
            {"action": "update", "entity_id": entry.entity_id, "changes": {"name": "Changed"}},
        )
        with (
            patch.object(catalog, "_build_snapshot", side_effect=RuntimeError("rebuild failed")),
            pytest.raises(RuntimeError, match="rebuild failed"),
        ):
            _ = catalog.snapshot
        assert catalog.snapshot.entries == ()
        catalog.async_stop()

    _run(tmp_path, check)


def test_aliases_follow_native_rules_and_update_without_restart(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entity(hass, "light.kitchen", name="Свет")
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Не имя из registry"})
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        await hass.async_block_till_done()
        old = catalog.snapshot
        assert "свет" in old.by_name
        assert "не имя из registry" not in old.by_name

        with patch.object(catalog, "_build_snapshot", wraps=catalog._build_snapshot) as build:
            er.async_get(hass).async_update_entity(entry.entity_id, name="Люстра")
            er.async_get(hass).async_update_entity(
                entry.entity_id, aliases=[er.COMPUTED_NAME, "ЛАМПА", "лампа", "  "]
            )
            await hass.async_block_till_done()
            new = catalog.snapshot
            assert build.call_count == 1
        assert new.by_name["лампа"][0].names == ("Люстра", "ЛАМПА", "лампа")
        assert len(new.by_name["лампа"]) == 1
        assert "люстра" in new.by_name and "свет" not in new.by_name
        assert "свет" in old.by_name
        with pytest.raises(TypeError):
            new.by_name["injected"] = ()

        er.async_get(hass).async_update_entity(entry.entity_id, aliases=[])
        await hass.async_block_till_done()
        assert not catalog.snapshot.entries
        catalog.async_stop()

    _run(tmp_path, check)


def test_real_state_events_skip_value_updates_and_track_name_and_deletion(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entity_id = "light.legacy"
        hass.states.async_set(entity_id, "off", {"friendly_name": "Свет"})
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entity_id})
        await hass.async_block_till_done()
        old = catalog.snapshot
        with patch.object(catalog, "_build_snapshot", wraps=catalog._build_snapshot) as build:
            hass.states.async_set(entity_id, "on", {"friendly_name": "Свет", "brightness": 42})
            await hass.async_block_till_done()
            assert catalog.snapshot is old
            build.assert_not_called()
            hass.states.async_set(entity_id, "on", {"friendly_name": "Лампа"})
            await hass.async_block_till_done()
            assert "лампа" in catalog.snapshot.by_name
            assert "свет" not in catalog.snapshot.by_name
            build.assert_called_once()
        hass.states.async_remove(entity_id)
        await hass.async_block_till_done()
        assert not catalog.snapshot.entries
        catalog.async_stop()

    _run(tmp_path, check)


def test_exposure_revocation_is_visible_before_debounce_and_stop_cancels_work(
    tmp_path: Path,
) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entity(hass, "light.kitchen", name="Свет")
        hass.states.async_set(entry.entity_id, "on")
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        await hass.async_block_till_done()
        assert catalog.snapshot.entries
        exposed_entities.async_expose_entity(hass, "conversation", entry.entity_id, False)
        assert not catalog.snapshot.entries
        exposed_entities.async_expose_entity(hass, "conversation", entry.entity_id, True)
        assert catalog.snapshot.entries

        with patch.object(catalog, "_build_snapshot", wraps=catalog._build_snapshot) as build:
            exposed_entities.async_expose_entity(hass, "conversation", entry.entity_id, False)
            catalog.async_stop()
            exposed_entities.async_expose_entity(hass, "conversation", entry.entity_id, True)
            hass.states.async_set("light.new", "on")
            await hass.async_block_till_done()
            build.assert_not_called()
            assert not catalog.snapshot.entries

    _run(tmp_path, check)


def test_expose_new_preference_applies_to_new_states_only(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        catalog = EntityCatalog(hass)
        await catalog.async_start()
        exposure = hass.data[DATA_EXPOSED_ENTITIES]
        exposure.async_set_expose_new_entities("conversation", False)
        hass.states.async_set("light.unexposed", "on", {"friendly_name": "Первый"})
        await hass.async_block_till_done()
        assert not catalog.snapshot.entries
        exposure.async_set_expose_new_entities("conversation", True)
        hass.states.async_set("light.exposed", "on", {"friendly_name": "Второй"})
        await hass.async_block_till_done()
        assert {entry.entity_id for entry in catalog.snapshot.entries} == {"light.exposed"}
        catalog.async_stop()

    _run(tmp_path, check)


def test_child_device_inherits_parent_area_and_tracks_area_rename(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        config_entry = ConfigEntry(
            domain="test",
            title="Test",
            data={},
            options={},
            source="user",
            version=1,
            minor_version=1,
            unique_id=None,
            discovery_keys={},
            subentries_data=(),
        )
        with patch.object(hass.config_entries, "async_setup", new=AsyncMock(return_value=True)):
            await hass.config_entries.async_add(config_entry)
        areas = ar.async_get(hass)
        kitchen = areas.async_create("Кухня")
        devices = dr.async_get(hass)
        parent = devices.async_get_or_create(
            config_entry_id=config_entry.entry_id, identifiers={("test", "parent")}, name="Parent"
        )
        devices.async_update_device(parent.id, area_id=kitchen.id)
        child = devices.async_get_or_create_child(
            config_entry_id=config_entry.entry_id,
            identifiers={("test", "child")},
            parent_device_id=parent.id,
            name="Child",
        )
        entry = _entity(hass, "light.child", name="Свет", device_id=child.id)
        hass.states.async_set(entry.entity_id, "on")
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        assert catalog.snapshot.entries[0].area_id == kitchen.id
        assert catalog.snapshot.entries[0].area_name == "Кухня"
        areas.async_update(kitchen.id, name="Столовая")
        await hass.async_block_till_done()
        assert catalog.snapshot.entries[0].area_name == "Столовая"
        room = areas.async_create("Спальня")
        er.async_get(hass).async_update_entity(entry.entity_id, area_id=room.id)
        await hass.async_block_till_done()
        assert catalog.snapshot.entries[0].area_id == room.id
        catalog.async_stop()

    _run(tmp_path, check)


def test_partial_subscription_failure_does_not_leak_listeners(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        catalog = EntityCatalog(hass)
        before = hass.bus.async_listeners()
        with (
            patch.object(
                catalog_module, "async_listen_entity_updates", side_effect=RuntimeError("not ready")
            ),
            pytest.raises(RuntimeError),
        ):
            await catalog.async_start()
        assert hass.bus.async_listeners() == before
        assert not catalog.snapshot.entries
        with patch.object(catalog, "_build_snapshot") as build:
            hass.states.async_set("light.late", "on")
            await hass.async_block_till_done()
            build.assert_not_called()

    _run(tmp_path, check)


def test_entity_id_rename_and_registry_removal_refresh_candidates(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entity(hass, "light.old", name="Registered name")
        registry = er.async_get(hass)
        registry.async_update_entity(entry.entity_id, aliases=["Alias"])
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Legacy name"})
        catalog = EntityCatalog(hass)
        await _start_with_exposure(hass, catalog, {entry.entity_id})
        assert catalog.snapshot.by_name["alias"][0].entity_id == entry.entity_id
        registry.async_update_entity(entry.entity_id, new_entity_id="light.new")
        hass.states.async_remove(entry.entity_id)
        hass.states.async_set("light.new", "on", {"friendly_name": "Legacy name"})
        await hass.async_block_till_done()
        assert catalog.snapshot.by_name["alias"][0].entity_id == "light.new"

        # A still-present state without its registry record follows legacy naming.
        registry.async_remove("light.new")
        await hass.async_block_till_done()
        assert "alias" not in catalog.snapshot.by_name
        assert catalog.snapshot.by_name["legacy name"][0].entity_id == "light.new"
        hass.states.async_remove("light.new")
        await hass.async_block_till_done()
        assert not catalog.snapshot.entries
        catalog.async_stop()

    _run(tmp_path, check)
