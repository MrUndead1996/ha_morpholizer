"""Optional Home Assistant smoke tests (not a substitute for the HA test suite)."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components import conversation
from homeassistant.config_entries import SOURCE_USER, ConfigEntries, ConfigEntry
from homeassistant.core import Context, HomeAssistant, State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.intent import IntentResponse

from custom_components.morpholizer import (
    async_setup_entry,
    async_unload_entry,
)
from custom_components.morpholizer.config_flow import (
    MorpholizerConfigFlow,
    MorpholizerOptionsFlow,
)
from custom_components.morpholizer.const import (
    CONF_FALLBACK_AGENT,
    DOMAIN,
)
from custom_components.morpholizer.conversation import (
    MorpholizerConversationEntity,
)


def _run(tmp_path: Path, check: Callable[[HomeAssistant], object]) -> None:
    async def run() -> None:
        hass = HomeAssistant(str(tmp_path))
        hass.config_entries = ConfigEntries(hass, {})
        try:
            await check(hass)
        finally:
            await hass.async_stop(force=True)

    asyncio.run(run())


def _entry(fallback: str = "conversation.test_agent") -> ConfigEntry:
    return ConfigEntry(
        domain=DOMAIN,
        title="Morpholizer",
        data={CONF_FALLBACK_AGENT: fallback},
        options={},
        source=SOURCE_USER,
        version=1,
        minor_version=1,
        unique_id="morpholizer",
        discovery_keys={},
        subentries_data=(),
    )


def test_message_forwarding_uses_chat_log_id_and_preserves_result(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entry = _entry()
        entity = MorpholizerConversationEntity(entry)
        entity.hass = hass
        context = Context()
        user_input = conversation.ConversationInput(
            text="Turn on the kitchen light",
            context=context,
            conversation_id=None,
            device_id="device-1",
            satellite_id="satellite-1",
            language="ru",
            agent_id="conversation.morpholizer",
            extra_system_prompt="keep this metadata",
        )
        log = chat_log.ChatLog(hass=hass, conversation_id="existing-chat-session")
        result = conversation.ConversationResult(
            response=IntentResponse(language="ru"),
            conversation_id="downstream-session",
            continue_conversation=True,
        )

        registry = _registry_stub("conversation.test_agent", "test_platform")
        hass.states.async_set("conversation.test_agent", "on")
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(
                conversation, "async_converse", new=AsyncMock(return_value=result)
            ) as forward,
        ):
            actual = await entity._async_handle_message(user_input, log)

        assert actual is result
        assert forward.await_args.kwargs == {
            "hass": hass,
            "text": "Turn on the kitchen light",
            "conversation_id": "existing-chat-session",
            "context": context,
            "language": "ru",
            "agent_id": "conversation.test_agent",
            "device_id": "device-1",
            "satellite_id": "satellite-1",
            "extra_system_prompt": "keep this metadata",
        }

    _run(tmp_path, check)


def test_real_dispatch_preserves_session_without_duplicate_messages(tmp_path: Path) -> None:
    """Exercise the real dispatcher and chat logs, not a mocked async_converse."""

    class TestAgent(conversation.ConversationEntity):
        @property
        def supported_languages(self):
            return ["ru"]

        async def _async_handle_message(self, user_input, chat_log):
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Тестовый ответ")
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content="Тестовый ответ")
            )
            return conversation.ConversationResult(
                response=response, conversation_id=chat_log.conversation_id
            )

    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS

        proxy = MorpholizerConversationEntity(_entry())
        downstream = TestAgent()
        proxy.entity_id = "conversation.morpholizer"
        downstream.entity_id = "conversation.test_agent"
        entities = {entity.entity_id: entity for entity in (proxy, downstream)}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(downstream.entity_id, "unknown")
        registry = _registry_stub(downstream.entity_id, "test_platform")
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(TestAgent, "async_write_ha_state"),
        ):
            first = await conversation.async_converse(
                hass, "Первый вопрос", None, Context(), language="ru", agent_id=proxy.entity_id
            )
            second = await conversation.async_converse(
                hass,
                "Второй вопрос",
                first.conversation_id,
                Context(),
                language="ru",
                agent_id=proxy.entity_id,
            )

        assert first.conversation_id is not None
        assert second.conversation_id == first.conversation_id
        messages = hass.data[DATA_CHAT_LOGS][first.conversation_id].content
        assert [message.role for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [message.content for message in messages if message.role == "user"] == [
            "Первый вопрос",
            "Второй вопрос",
        ]

    _run(tmp_path, check)


@pytest.mark.parametrize("fallback", [None, "conversation.morpholizer", "conversation.test_agent"])
def test_missing_or_self_fallback_returns_error(tmp_path: Path, fallback: str | None) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log
        from homeassistant.helpers.intent import IntentResponseErrorCode

        entry = _entry(fallback or "")
        entity = MorpholizerConversationEntity(entry)
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        if fallback == "conversation.test_agent":
            registry = _registry_stub(fallback, "test_platform")
            hass.states.async_set(fallback, "unavailable")
        else:
            registry = _registry_stub("conversation.test_agent", DOMAIN)
        user_input = conversation.ConversationInput(
            text="hello",
            context=Context(),
            conversation_id=None,
            device_id=None,
            satellite_id=None,
            language="ru",
            agent_id=entity.entity_id,
        )
        log = chat_log.ChatLog(hass=hass, conversation_id="chat-id")

        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=AsyncMock()) as forward,
        ):
            result = await entity._async_handle_message(user_input, log)

        assert result.response.error_code == IntentResponseErrorCode.FAILED_TO_HANDLE
        assert result.conversation_id == "chat-id"
        forward.assert_not_awaited()

    _run(tmp_path, check)


def test_setup_registers_forwarding_and_reload_listener_and_unloads(
    tmp_path: Path,
) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.const import Platform

        entry = _entry()
        forward = AsyncMock()
        unload = AsyncMock()
        reload_entry = AsyncMock()
        with (
            patch.object(hass.config_entries, "async_forward_entry_setups", forward),
            patch.object(hass.config_entries, "async_unload_platforms_or_raise", unload),
            patch.object(hass.config_entries, "async_reload", reload_entry),
        ):
            await async_setup_entry(hass, entry)
            forward.assert_awaited_once_with(entry, (Platform.CONVERSATION,))
            assert entry._on_unload  # listener is registered for entry lifetime

            await async_unload_entry(hass, entry)
            unload.assert_awaited_once_with(entry, (Platform.CONVERSATION,))

            # Invoke the registered callback to prove options changes reload this entry.
            callbacks = entry.update_listeners
            assert callbacks
            await callbacks[0](hass, entry)
            reload_entry.assert_awaited_once_with(entry.entry_id)

    _run(tmp_path, check)


def test_options_flow_constructor_does_not_assign_readonly_config_entry() -> None:
    flow = MorpholizerOptionsFlow()
    assert isinstance(flow, MorpholizerOptionsFlow)


def test_config_flow_rejects_registry_entity_from_its_own_integration(
    tmp_path: Path,
) -> None:
    async def check(hass: HomeAssistant) -> None:
        from unittest.mock import Mock

        own_id = "conversation.morpholizer"
        registry = _registry_stub(own_id, DOMAIN, entities={})
        hass.states.async_set(own_id, "on")
        flow = MorpholizerConfigFlow()
        flow.hass = hass
        flow.async_set_unique_id = AsyncMock()
        flow._abort_if_unique_id_configured = Mock()
        flow.async_show_form = Mock(side_effect=lambda **kwargs: kwargs)

        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=object()) as get_agent,
        ):
            result = await flow.async_step_user({CONF_FALLBACK_AGENT: own_id})

        assert result["errors"]["base"] == "invalid_agent"
        get_agent.assert_not_called()

    _run(tmp_path, check)


def _registry_stub(entity_id: str, platform: str, *, entities: dict | None = None):
    """Provide only the registry records queried by the integration under test."""
    entry = SimpleNamespace(entity_id=entity_id, platform=platform)
    return SimpleNamespace(
        async_get=lambda requested: entry if requested == entity_id else None,
        entities=entities or {},
    )


def test_unknown_state_is_usable_when_registry_and_agent_exist(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entry()
        entity = MorpholizerConversationEntity(entry)
        entity.hass = hass
        fallback_id = "conversation.test_agent"
        registry = _registry_stub(fallback_id, "test_platform")
        hass.states.async_set(fallback_id, "unknown")
        assert isinstance(hass.states.get(fallback_id), State)
        user_input = conversation.ConversationInput(
            text="hello",
            context=Context(),
            conversation_id=None,
            device_id=None,
            satellite_id=None,
            language="ru",
            agent_id="conversation.morpholizer",
        )
        from homeassistant.components.conversation import chat_log

        log = chat_log.ChatLog(hass=hass, conversation_id="unknown-state-chat")
        result = conversation.ConversationResult(
            response=IntentResponse(language="ru"),
            conversation_id="downstream",
        )

        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(
                conversation, "async_converse", new=AsyncMock(return_value=result)
            ) as forward,
        ):
            actual = await entity._async_handle_message(user_input, log)

        assert actual is result
        forward.assert_awaited_once()

    _run(tmp_path, check)
