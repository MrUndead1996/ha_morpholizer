"""Runtime fuzzy setting loading remains defensive for old/corrupted entries."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("homeassistant")

from custom_components.morpholizer.const import (
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_MARGIN,
    CONF_FUZZY_SCORE_CUTOFF,
)
from custom_components.morpholizer.conversation import _entry_fuzzy_settings, async_setup_entry
from custom_components.morpholizer.matching import FuzzySettings


def test_entry_settings_forward_valid_values() -> None:
    settings = _entry_fuzzy_settings(
        {
            CONF_FUZZY_ENABLED: True,
            CONF_FUZZY_SCORE_CUTOFF: 95,
            CONF_FUZZY_MARGIN: 12,
        }
    )
    assert (settings.enabled, settings.score_cutoff, settings.margin) == (True, 95.0, 12.0)


@pytest.mark.parametrize(
    "options",
    [
        {CONF_FUZZY_ENABLED: 1},
        {CONF_FUZZY_SCORE_CUTOFF: True},
        {CONF_FUZZY_SCORE_CUTOFF: math.nan},
        {CONF_FUZZY_MARGIN: 31},
    ],
)
def test_corrupted_entry_settings_fail_closed(options: dict[str, object]) -> None:
    settings = _entry_fuzzy_settings(options)
    assert (settings.enabled, settings.score_cutoff, settings.margin) == (False, 90.0, 8.0)


@pytest.mark.parametrize("enabled", [False, True])
def test_setup_forwards_options_to_executor(enabled: bool) -> None:
    async def check() -> None:
        normalizer = object()
        hass = SimpleNamespace(async_add_executor_job=AsyncMock(return_value=normalizer))
        entry = SimpleNamespace(options={CONF_FUZZY_ENABLED: enabled})
        add = Mock()
        with patch(
            "custom_components.morpholizer.conversation.MorpholizerConversationEntity"
        ) as entity:
            await async_setup_entry(hass, entry, add)
            entity.assert_called_once_with(entry, normalizer)
            add.assert_called_once_with([entity.return_value])
        args = hass.async_add_executor_job.await_args.args
        assert args[0].__name__ == "_load_normalizer"
        assert args[1:] == ((FuzzySettings(enabled=True),) if enabled else ())

    asyncio.run(check())
