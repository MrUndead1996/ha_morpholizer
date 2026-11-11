"""Exercise local Home Assistant intent routing with the real default agent."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components import conversation
from homeassistant.components.conversation.agent_manager import get_agent_manager
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.components.conversation.default_agent import DefaultAgent
from homeassistant.components.homeassistant.exposed_entities import (
    DATA_EXPOSED_ENTITIES,
    ExposedEntities,
)
from homeassistant.components.intent import OnOffIntentHandler
from homeassistant.config_entries import SOURCE_USER, ConfigEntries, ConfigEntry
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import (
    area_registry,
    device_registry,
    entity_registry,
    floor_registry,
    intent,
)
from homeassistant.helpers.intent import IntentResponse

from custom_components.morpholizer.const import CONF_FALLBACK_AGENT, DOMAIN
from custom_components.morpholizer.conversation import MorpholizerConversationEntity


def _entry() -> ConfigEntry:
    return ConfigEntry(
        domain=DOMAIN,
        title="Morpholizer",
        data={CONF_FALLBACK_AGENT: "conversation.test_agent"},
        options={},
        source=SOURCE_USER,
        version=1,
        minor_version=1,
        unique_id="local-intent-probe",
        discovery_keys={},
        subentries_data=(),
    )


class _Registry:
    def __init__(self) -> None:
        self.entries: dict[str, SimpleNamespace] = {}

    def async_get(self, entity_id: str) -> SimpleNamespace | None:
        return self.entries.get(entity_id)


class _DeviceRegistry:
    def async_get_device(self, **kwargs: object) -> None:
        return None


def _run(tmp_path: Path, check: Callable[[HomeAssistant], object]) -> None:
    async def run() -> None:
        hass = HomeAssistant(str(tmp_path))
        hass.config_entries = ConfigEntries(hass, {})
        hass.config.language = "ru"
        try:
            registry = _Registry()
            exposed_entities = ExposedEntities(hass)
            hass.data[DATA_EXPOSED_ENTITIES] = exposed_entities
            await exposed_entities.async_initialize()

            # Keep entity/area/device registry boundaries minimal; intent
            # recognition, exposure policy, target matching and processing stay real.
            empty_areas = SimpleNamespace(async_list_areas=list)
            empty_floors = SimpleNamespace(async_list_floors=list)
            with (
                patch.object(entity_registry, "async_get", return_value=registry),
                patch.object(area_registry, "async_get", return_value=empty_areas),
                patch.object(floor_registry, "async_get", return_value=empty_floors),
                patch.object(device_registry, "async_get", return_value=_DeviceRegistry()),
            ):
                await check(hass)
        finally:
            await hass.async_stop(force=True)

    asyncio.run(run())


async def _setup_default_agent(hass: HomeAssistant) -> None:
    agent = DefaultAgent(hass)
    await get_agent_manager(hass).async_setup_default_agent(agent)
    intent.async_register(
        hass,
        OnOffIntentHandler(intent.INTENT_TURN_ON, "homeassistant", "turn_on"),
    )


def _chat_log(hass: HomeAssistant) -> ChatLog:
    return ChatLog(hass=hass, conversation_id="local-intent-test")


def _user_input(text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id="local-intent-test",
        device_id=None,
        satellite_id=None,
        language="ru",
        agent_id="conversation.morpholizer",
    )


def _assistant_messages(chat_log: ChatLog) -> list[str]:
    return [
        item.content
        for item in chat_log.content
        if item.role == "assistant" and item.content is not None
    ]


def test_real_russian_local_intent_executes_once_without_fallback(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        await _setup_default_agent(hass)
        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        hass.states.async_set("light.kitchen", "off", {"friendly_name": "кухонный свет"})

        service_calls: list[ServiceCall] = []

        async def turn_on(call: ServiceCall) -> None:
            service_calls.append(call)
            hass.states.async_set("light.kitchen", "on", {"friendly_name": "кухонный свет"})

        hass.services.async_register("light", "turn_on", turn_on)
        hass.services.async_register("homeassistant", "turn_on", turn_on)
        fallback = AsyncMock()
        chat_log = _chat_log(hass)
        with patch.object(conversation, "async_converse", new=fallback):
            result = await entity._async_handle_message(
                _user_input("включи кухонный свет"), chat_log
            )

        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
        assert result.response.error_code is None
        assert result.response.speech["plain"]["speech"] == "Свет включен"
        assert hass.states.get("light.kitchen").state == "on"
        assert len(service_calls) == 1
        assert service_calls[0].data["entity_id"] == "light.kitchen"
        fallback.assert_not_awaited()
        assert len(_assistant_messages(chat_log)) == 1

    _run(tmp_path, check)


def test_local_intent_service_error_does_not_fallback(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        await _setup_default_agent(hass)
        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        hass.states.async_set("light.kitchen", "off", {"friendly_name": "кухонный свет"})

        service_calls = 0

        async def failed_turn_on(call: ServiceCall) -> None:
            nonlocal service_calls
            service_calls += 1
            raise RuntimeError("simulated light service failure")

        hass.services.async_register("light", "turn_on", failed_turn_on)
        hass.services.async_register("homeassistant", "turn_on", failed_turn_on)
        fallback = AsyncMock()
        chat_log = _chat_log(hass)
        with patch.object(conversation, "async_converse", new=fallback):
            result = await entity._async_handle_message(
                _user_input("включи кухонный свет"), chat_log
            )

        assert service_calls == 1
        assert hass.states.get("light.kitchen").state == "off"
        assert result.response.error_code is not None
        fallback.assert_not_awaited()
        assert len(_assistant_messages(chat_log)) == 1

    _run(tmp_path, check)


def test_unmatched_text_falls_back_once(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        await _setup_default_agent(hass)
        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        fallback_registry_entry = SimpleNamespace(
            platform="test_platform",
            options={"conversation": {"should_expose": False}},
        )
        registry = entity_registry.async_get(hass)
        registry.entries["conversation.test_agent"] = fallback_registry_entry
        hass.states.async_set("conversation.test_agent", "unknown")
        downstream_result = conversation.ConversationResult(
            response=IntentResponse(language="ru"),
            conversation_id="local-intent-test",
        )
        fallback = AsyncMock(return_value=downstream_result)
        chat_log = _chat_log(hass)
        with (
            patch.object(conversation, "async_converse", new=fallback),
            patch.object(conversation, "async_get_agent", return_value=object()),
        ):
            result = await entity._async_handle_message(
                _user_input("zzqv random blargle"), chat_log
            )

        assert result is downstream_result
        fallback.assert_awaited_once()
        assert fallback.await_args.kwargs["agent_id"] == "conversation.test_agent"

    _run(tmp_path, check)
