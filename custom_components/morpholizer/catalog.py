"""Immutable catalog of exposed entities and HA areas/floors."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from homeassistant.components.homeassistant.exposed_entities import (
    async_listen_entity_updates,
    async_should_expose,
)
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers.intent import async_get_entity_aliases


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One exposed entity and its usable names/context."""

    entity_id: str
    names: tuple[str, ...]
    domain: str
    area_id: str | None = None
    area_name: str | None = None


@dataclass(frozen=True, slots=True)
class LocationEntry:
    """One area or floor, preserving its registry name and aliases."""

    kind: Literal["area", "floor"]
    id: str
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """A complete immutable catalog generation."""

    entries: tuple[CatalogEntry, ...]
    by_name: Mapping[str, tuple[CatalogEntry, ...]]
    locations: tuple[LocationEntry, ...] = ()


def _key(name: str) -> str:
    """Normalize whitespace and casing, without linguistic transformations."""
    return " ".join(name.split()).casefold()


class EntityCatalog:
    """Maintain exposed entities and all locations, like HA's local agent."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._snapshot = CatalogSnapshot((), MappingProxyType({}))
        self._unsubscribers: list[CALLBACK_TYPE] = []
        self._debounce: CALLBACK_TYPE | None = None
        self._started = False

    @property
    def snapshot(self) -> CatalogSnapshot:
        """Return a complete generation, flushing any already notified changes."""
        if self._debounce is not None:
            self._debounce()
            self._debounce = None
            self._flush()
        return self._snapshot

    async def async_start(self) -> None:
        """Subscribe and build the initial snapshot; roll back a partial start."""
        if self._started:
            return
        self._started = True
        try:
            for event_type in (
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                ar.EVENT_AREA_REGISTRY_UPDATED,
                dr.EVENT_DEVICE_REGISTRY_UPDATED,
                fr.EVENT_FLOOR_REGISTRY_UPDATED,
            ):
                self._unsubscribers.append(
                    self.hass.bus.async_listen(event_type, self._on_registry)
                )
            # The indexed state-change helper treats "*" as a literal entity ID.
            self._unsubscribers.append(
                self.hass.bus.async_listen(EVENT_STATE_CHANGED, self._on_state_change)
            )
            self._unsubscribers.append(
                async_listen_entity_updates(self.hass, "conversation", self._on_exposure_change)
            )
            self._build_snapshot()
        except BaseException:
            self.async_stop()
            raise

    @callback
    def async_stop(self) -> None:
        """Cancel pending rebuild and remove all subscriptions."""
        if self._debounce:
            self._debounce()
            self._debounce = None
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._started = False
        self._snapshot = CatalogSnapshot((), MappingProxyType({}))

    @callback
    def _on_registry(self, event: Event) -> None:
        self._schedule_rebuild()

    @callback
    def _on_exposure_change(self) -> None:
        self._schedule_rebuild()

    @callback
    def _on_state_change(self, event: Event) -> None:
        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if (old_state.name if old_state else None) != (new_state.name if new_state else None):
            self._schedule_rebuild()

    @callback
    def _schedule_rebuild(self) -> None:
        if self._started and self._debounce is None:
            self._debounce = self.hass.loop.call_soon(self._flush).cancel

    @callback
    def _flush(self) -> None:
        self._debounce = None
        if self._started:
            try:
                self._build_snapshot()
            except Exception:
                # A stale generation may still contain newly unexposed names.
                # Fail closed rather than continuing to offer those candidates.
                self._snapshot = CatalogSnapshot((), MappingProxyType({}))
                raise

    @callback
    def _build_snapshot(self) -> None:
        states = self.hass.states.async_all()
        entity_registry = er.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        floor_registry = fr.async_get(self.hass)
        entries: list[CatalogEntry] = []
        names_index: defaultdict[str, list[CatalogEntry]] = defaultdict(list)

        for state in states:
            entity_id = state.entity_id
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue
            registry_entry = entity_registry.async_get(entity_id)
            # Use HA's own computed-name semantics (including registry/state
            # precedence and aliases), rather than independently merging fields.
            candidates = async_get_entity_aliases(self.hass, registry_entry, state=state)
            names: list[str] = []
            seen_names: set[str] = set()
            for candidate in candidates:
                key = _key(candidate)
                if key and candidate not in seen_names:
                    names.append(candidate)
                    seen_names.add(candidate)

            if not names:
                continue

            area_id = registry_entry.area_id if registry_entry else None
            if not area_id and registry_entry and registry_entry.device_id:
                device = device_registry.async_get(registry_entry.device_id)
                if device:
                    area_id = dr.async_get_effective_area_id(self.hass, device)
            area = area_registry.async_get_area(area_id) if area_id else None
            entry = CatalogEntry(
                entity_id,
                tuple(names),
                entity_id.split(".", 1)[0],
                area_id,
                area.name if area else None,
            )
            entries.append(entry)
            for key in dict.fromkeys(_key(name) for name in names):
                names_index[key].append(entry)

        # HA's local agent offers every registered area/floor, including empty
        # ones. Exposure remains an entity-level check at intent execution.
        locations = tuple(
            LocationEntry(
                kind,
                location.id if kind == "area" else location.floor_id,
                tuple(
                    dict.fromkeys(
                        name.strip()
                        for name in (location.name, *sorted(location.aliases))
                        if name.strip()
                    )
                ),
            )
            for kind, registry_entries in (
                ("area", area_registry.async_list_areas()),
                ("floor", floor_registry.async_list_floors()),
            )
            for location in registry_entries
        )
        self._snapshot = CatalogSnapshot(
            tuple(entries),
            MappingProxyType({key: tuple(value) for key, value in names_index.items()}),
            locations,
        )
