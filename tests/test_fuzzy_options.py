"""Fuzzy options validation and schema behavior."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

pytest.importorskip("homeassistant")

from custom_components.morpholizer.config_flow import (
    MorpholizerOptionsFlow,
    _fuzzy_values,
    _options_schema,
)
from custom_components.morpholizer.const import (
    CONF_FALLBACK_AGENT,
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_MARGIN,
    CONF_FUZZY_SCORE_CUTOFF,
)


def test_old_options_get_conservative_fuzzy_defaults() -> None:
    assert _fuzzy_values({}) == {
        CONF_FUZZY_ENABLED: False,
        CONF_FUZZY_SCORE_CUTOFF: 90.0,
        CONF_FUZZY_MARGIN: 8.0,
    }


@pytest.mark.parametrize(
    "values",
    [
        {CONF_FUZZY_ENABLED: 1},
        {CONF_FUZZY_SCORE_CUTOFF: True},
        {CONF_FUZZY_SCORE_CUTOFF: math.nan},
        {CONF_FUZZY_SCORE_CUTOFF: math.inf},
        {CONF_FUZZY_SCORE_CUTOFF: 84.99},
        {CONF_FUZZY_SCORE_CUTOFF: 100.01},
        {CONF_FUZZY_MARGIN: False},
        {CONF_FUZZY_MARGIN: 4.99},
        {CONF_FUZZY_MARGIN: 30.01},
    ],
)
def test_invalid_fuzzy_options_are_rejected(values: dict[str, object]) -> None:
    assert _fuzzy_values(values) is None


def test_fuzzy_options_schema_contains_all_optional_controls() -> None:
    # Schema construction is HA-facing; a minimal hass stub is sufficient because
    # the registry is only read to populate the fallback selector's exclusions.
    class Registry:
        def __init__(self) -> None:
            self.entities: dict[str, object] = {}

    class Hass:
        @staticmethod
        def data() -> dict[str, object]:
            return {}

    # Exercise the schema via a mocked registry lookup without requiring a running HA.
    from unittest.mock import patch

    with patch("custom_components.morpholizer.config_flow.er.async_get", return_value=Registry()):
        schema = _options_schema(Hass())
    assert CONF_FUZZY_ENABLED in {str(key) for key in schema.schema}
    assert CONF_FUZZY_SCORE_CUTOFF in {str(key) for key in schema.schema}
    assert CONF_FUZZY_MARGIN in {str(key) for key in schema.schema}


def test_options_roundtrip_and_validation() -> None:
    async def check() -> None:
        values = {
            CONF_FALLBACK_AGENT: "conversation.llm",
            CONF_FUZZY_ENABLED: True,
            CONF_FUZZY_SCORE_CUTOFF: 94,
            CONF_FUZZY_MARGIN: 12,
        }
        entry = SimpleNamespace(options=values, data={})
        flow = MorpholizerOptionsFlow()
        flow.hass = object()
        flow.async_show_form = Mock(side_effect=lambda **kwargs: kwargs)
        flow.async_create_entry = Mock(side_effect=lambda **kwargs: kwargs)
        with (
            patch.object(
                MorpholizerOptionsFlow,
                "config_entry",
                new_callable=PropertyMock,
                return_value=entry,
            ),
            patch(
                "custom_components.morpholizer.config_flow.er.async_get",
                return_value=SimpleNamespace(entities={}),
            ),
            patch("custom_components.morpholizer.config_flow._valid_fallback", return_value=True),
        ):
            form = await flow.async_step_init()
            suggested = {
                str(k): k.description.get("suggested_value") for k in form["data_schema"].schema
            }
            assert suggested == values
            result = await flow.async_step_init(values)
            assert result["data"] == values
            invalid = await flow.async_step_init({**values, CONF_FUZZY_MARGIN: 0})
            assert invalid["errors"] == {"base": "invalid_fuzzy_settings"}

    asyncio.run(check())
