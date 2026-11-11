"""Morpholizer integration runtime."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Morpholizer integration."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Set up the conversation platform and reload when options change."""
    await hass.config_entries.async_forward_entry_setups(entry, (Platform.CONVERSATION,))
    entry.async_on_unload(entry.add_update_listener(_async_update_options))


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Unload the conversation platform, raising if it fails."""
    await hass.config_entries.async_unload_platforms_or_raise(entry, (Platform.CONVERSATION,))


async def _async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the integration after options change."""
    await hass.config_entries.async_reload(entry.entry_id)
