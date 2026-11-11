"""Config and options flows for Morpholizer."""

from typing import Any

import probatio
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import EntitySelector

from .const import CONF_FALLBACK_AGENT, DOMAIN


def _fallback_schema(hass: HomeAssistant) -> probatio.Schema:
    """Build a required conversation entity selector excluding our own entities."""
    registry = er.async_get(hass)
    own_entities = [
        entity.entity_id for entity in registry.entities.values() if entity.platform == DOMAIN
    ]
    return probatio.Schema(
        {
            probatio.Required(CONF_FALLBACK_AGENT): EntitySelector(
                {"domain": ["conversation"], "exclude_entities": own_entities}
            )
        }
    )


def _valid_fallback(hass: HomeAssistant, fallback_id: str) -> bool:
    """Check that the selected entity is a loaded, available conversation agent."""
    registry_entry = er.async_get(hass).async_get(fallback_id)
    if registry_entry is None or registry_entry.platform == DOMAIN:
        return False
    if registry_entry.entity_id.partition(".")[0] != "conversation":
        return False
    if conversation.async_get_agent(hass, fallback_id) is None:
        return False
    state = hass.states.get(fallback_id)
    # A conversation entity's state is its last activity, not readiness. A newly
    # loaded agent may be "unknown" until its first request and is still usable.
    return state is not None and state.state != STATE_UNAVAILABLE


class MorpholizerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure one Morpholizer instance and its fallback agent."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Select an available conversation entity as fallback."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            fallback_id = user_input[CONF_FALLBACK_AGENT]
            if not _valid_fallback(self.hass, fallback_id):
                return self.async_show_form(
                    step_id="user",
                    data_schema=_fallback_schema(self.hass),
                    errors={"base": "invalid_agent"},
                )
            return self.async_create_entry(
                title="Morpholizer",
                data={CONF_FALLBACK_AGENT: fallback_id},
            )

        return self.async_show_form(
            step_id="user",
            data_schema=_fallback_schema(self.hass),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow handler."""
        return MorpholizerOptionsFlow()


class MorpholizerOptionsFlow(config_entries.OptionsFlow):
    """Change the selected fallback conversation entity."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and save the selected fallback entity."""
        if user_input is not None:
            fallback_id = user_input[CONF_FALLBACK_AGENT]
            if not _valid_fallback(self.hass, fallback_id):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_fallback_schema(self.hass),
                    errors={"base": "invalid_agent"},
                )
            return self.async_create_entry(title="", data=user_input)

        current = self.config_entry.options.get(
            CONF_FALLBACK_AGENT,
            self.config_entry.data.get(CONF_FALLBACK_AGENT),
        )
        schema = _fallback_schema(self.hass)
        if current and _valid_fallback(self.hass, current):
            schema = self.add_suggested_values_to_schema(schema, {CONF_FALLBACK_AGENT: current})
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )
