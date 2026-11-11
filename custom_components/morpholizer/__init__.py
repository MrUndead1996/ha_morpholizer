"""Morpholizer integration runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import ConfigType

    from .catalog import EntityCatalog


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Morpholizer integration."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Set up the conversation platform and reload when options change."""
    from homeassistant.const import Platform

    from .catalog import EntityCatalog

    catalog = EntityCatalog(hass)
    try:
        await catalog.async_start()
        entry.runtime_data = catalog
        await hass.config_entries.async_forward_entry_setups(entry, (Platform.CONVERSATION,))
    except BaseException:
        catalog.async_stop()
        entry.runtime_data = None
        raise
    entry.async_on_unload(entry.add_update_listener(_async_update_options))


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Unload the conversation platform, raising if it fails."""
    from homeassistant.const import Platform

    await hass.config_entries.async_unload_platforms_or_raise(entry, (Platform.CONVERSATION,))
    catalog: EntityCatalog = entry.runtime_data
    catalog.async_stop()
    entry.runtime_data = None


async def _async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the integration after options change."""
    await hass.config_entries.async_reload(entry.entry_id)
