"""Config and options flows for Morpholizer."""

import math
from typing import Any

import probatio
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import (
    CONF_FALLBACK_AGENT,
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_MARGIN,
    CONF_FUZZY_SCORE_CUTOFF,
    DOMAIN,
    FUZZY_DEFAULT_ENABLED,
    FUZZY_DEFAULT_MARGIN,
    FUZZY_DEFAULT_SCORE_CUTOFF,
    FUZZY_MAX_MARGIN,
    FUZZY_MAX_SCORE_CUTOFF,
    FUZZY_MIN_MARGIN,
    FUZZY_MIN_SCORE_CUTOFF,
)
from .matching import FuzzySettings


def _fuzzy_values(values: dict[str, Any]) -> dict[str, Any] | None:
    """Validate optional values (including calls that bypass the UI schema)."""
    enabled = values.get(CONF_FUZZY_ENABLED, FUZZY_DEFAULT_ENABLED)
    score = values.get(CONF_FUZZY_SCORE_CUTOFF, FUZZY_DEFAULT_SCORE_CUTOFF)
    margin = values.get(CONF_FUZZY_MARGIN, FUZZY_DEFAULT_MARGIN)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        return None
    try:
        score = float(score)
        margin = float(margin)
        if not math.isfinite(score) or not math.isfinite(margin):
            return None
        settings = FuzzySettings(enabled=enabled, score_cutoff=score, margin=margin)
    except TypeError, ValueError, OverflowError:
        return None
    return {
        CONF_FUZZY_ENABLED: settings.enabled,
        CONF_FUZZY_SCORE_CUTOFF: settings.score_cutoff,
        CONF_FUZZY_MARGIN: settings.margin,
    }


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
            fuzzy = _fuzzy_values(user_input)
            if fuzzy is None:
                return self.async_show_form(
                    step_id="init",
                    data_schema=_options_schema(self.hass),
                    errors={"base": "invalid_fuzzy_settings"},
                )
            fallback_id = user_input.get(CONF_FALLBACK_AGENT)
            if not _valid_fallback(self.hass, fallback_id):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_options_schema(self.hass),
                    errors={"base": "invalid_agent"},
                )
            return self.async_create_entry(title="", data={**user_input, **fuzzy})

        current = self.config_entry.options.get(
            CONF_FALLBACK_AGENT,
            self.config_entry.data.get(CONF_FALLBACK_AGENT),
        )
        schema = _options_schema(self.hass)
        if current and _valid_fallback(self.hass, current):
            schema = self.add_suggested_values_to_schema(schema, {CONF_FALLBACK_AGENT: current})
        fuzzy = _fuzzy_values(self.config_entry.options)
        schema = self.add_suggested_values_to_schema(
            schema,
            fuzzy
            or {
                CONF_FUZZY_ENABLED: FUZZY_DEFAULT_ENABLED,
                CONF_FUZZY_SCORE_CUTOFF: FUZZY_DEFAULT_SCORE_CUTOFF,
                CONF_FUZZY_MARGIN: FUZZY_DEFAULT_MARGIN,
            },
        )
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )


def _options_schema(hass: HomeAssistant) -> probatio.Schema:
    """Fallback selector plus opt-in, conservatively bounded fuzzy settings."""
    return probatio.Schema(
        {
            **_fallback_schema(hass).schema,
            probatio.Optional(CONF_FUZZY_ENABLED): BooleanSelector(),
            probatio.Optional(CONF_FUZZY_SCORE_CUTOFF): NumberSelector(
                NumberSelectorConfig(
                    min=FUZZY_MIN_SCORE_CUTOFF,
                    max=FUZZY_MAX_SCORE_CUTOFF,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            probatio.Optional(CONF_FUZZY_MARGIN): NumberSelector(
                NumberSelectorConfig(
                    min=FUZZY_MIN_MARGIN,
                    max=FUZZY_MAX_MARGIN,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
        }
    )
