"""Routing contracts for Morpholizer's local-first conversation agent."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components import conversation
from homeassistant.config_entries import SOURCE_USER, ConfigEntries, ConfigEntry
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.intent import IntentResponse, IntentResponseErrorCode

from custom_components.morpholizer.const import CONF_FALLBACK_AGENT, DOMAIN
from custom_components.morpholizer.conversation import MorpholizerConversationEntity


def _run(tmp_path: Path, check) -> None:
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
        unique_id="morpholizer-routing",
        discovery_keys={},
        subentries_data=(),
    )


def _registry(entity_id: str):
    record = SimpleNamespace(entity_id=entity_id, platform="test_platform")
    return SimpleNamespace(async_get=lambda requested: record if requested == entity_id else None)


def _input() -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text="Turn on the kitchen light",
        context=Context(),
        conversation_id="chat-id",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.morpholizer",
    )


@pytest.mark.parametrize("local_error", [False, True])
def test_any_local_result_is_terminal_even_failed_to_handle(
    tmp_path: Path, local_error: bool
) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="chat-id")
        local = IntentResponse(language="en")
        if local_error:
            local.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE, "local miss")
        fallback = AsyncMock()
        with (
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=local)),
            patch.object(er, "async_get", return_value=_registry("conversation.test_agent")),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=fallback),
        ):
            result = await entity._async_handle_message(_input(), log)

        assert result.response is local
        assert result.conversation_id == "chat-id"
        fallback.assert_not_awaited()

    _run(tmp_path, check)


def test_local_exception_returns_safe_error_without_fallback(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="chat-id")
        fallback = AsyncMock()
        with (
            patch.object(
                conversation,
                "async_handle_intents",
                new=AsyncMock(side_effect=RuntimeError("local internal details")),
            ),
            patch.object(er, "async_get", return_value=_registry("conversation.test_agent")),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=fallback),
        ):
            result = await entity._async_handle_message(_input(), log)

        assert result.response.error_code == IntentResponseErrorCode.FAILED_TO_HANDLE
        assert "local internal details" not in result.response.speech.get("plain", {}).get(
            "speech", ""
        )
        fallback.assert_not_awaited()

    _run(tmp_path, check)


def test_local_success_does_not_require_available_fallback(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entity = MorpholizerConversationEntity(_entry(""))
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="chat-id")
        local = IntentResponse(language="en")
        local.async_set_speech("Done")
        with patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=local)):
            result = await entity._async_handle_message(_input(), log)

        assert result.response is local
        assert result.response.error_code is None

    _run(tmp_path, check)


@pytest.mark.parametrize(
    ("continue_conversation", "assistant_text", "local_calls_after_second"),
    [(True, "Tell me more.", 1), (False, "Would you like to continue?", 2)],
)
def test_real_session_followup_uses_explicit_continue_state(
    tmp_path: Path,
    continue_conversation: bool,
    assistant_text: str,
    local_calls_after_second: int,
) -> None:
    class Agent(conversation.ConversationEntity):
        def __init__(self):
            self.turns = 0

        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            self.turns += 1
            response = IntentResponse(language=user_input.language)
            response.async_set_speech(assistant_text)
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content=assistant_text)
            )
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=continue_conversation and self.turns == 1,
            )

    async def check(hass: HomeAssistant) -> None:
        proxy = MorpholizerConversationEntity(_entry())
        downstream = Agent()
        proxy.entity_id = "conversation.morpholizer"
        downstream.entity_id = "conversation.test_agent"
        entities = {proxy.entity_id: proxy, downstream.entity_id: downstream}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(downstream.entity_id, "unknown")
        local = AsyncMock(return_value=None)
        registry = _registry(downstream.entity_id)
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(Agent, "async_write_ha_state"),
        ):
            first = await conversation.async_converse(
                hass, "First", None, Context(), language="en", agent_id=proxy.entity_id
            )
            second = await conversation.async_converse(
                hass,
                "Second",
                first.conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )

        assert second.conversation_id == first.conversation_id
        assert local.await_count == local_calls_after_second
        if continue_conversation:
            with (
                patch.object(er, "async_get", return_value=registry),
                patch.object(conversation, "async_get_agent", return_value=downstream),
                patch.object(conversation, "async_handle_intents", new=local),
                patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
                patch.object(Agent, "async_write_ha_state"),
            ):
                third = await conversation.async_converse(
                    hass,
                    "Third",
                    second.conversation_id,
                    Context(),
                    language="en",
                    agent_id=proxy.entity_id,
                )
            assert third.conversation_id == first.conversation_id
            assert local.await_count == 2
        from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS

        log = hass.data[DATA_CHAT_LOGS][first.conversation_id]
        assert [item.content for item in log.content if item.role == "user"] == [
            "First",
            "Second",
            *(["Third"] if continue_conversation else []),
        ]
        assert sum(item.role == "assistant" for item in log.content) == (
            3 if continue_conversation else 2
        )

    _run(tmp_path, check)


def test_simultaneous_sessions_keep_continue_state_isolated(tmp_path: Path) -> None:
    class Agent(conversation.ConversationEntity):
        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Continue")
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content="Continue")
            )
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=user_input.text != "B",
            )

    async def check(hass: HomeAssistant) -> None:
        proxy = MorpholizerConversationEntity(_entry())
        downstream = Agent()
        proxy.entity_id = "conversation.morpholizer"
        downstream.entity_id = "conversation.test_agent"
        entities = {proxy.entity_id: proxy, downstream.entity_id: downstream}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(downstream.entity_id, "unknown")
        gate = asyncio.Event()
        both_started = asyncio.Event()
        start_count = 0
        local = AsyncMock(return_value=None)

        async def blocked_local(*args, **kwargs):
            nonlocal start_count
            start_count += 1
            if start_count == 2:
                both_started.set()
            await gate.wait()

        local.side_effect = blocked_local
        registry = _registry(downstream.entity_id)
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(Agent, "async_write_ha_state"),
        ):
            first_task = asyncio.create_task(
                conversation.async_converse(
                    hass, "A", None, Context(), language="en", agent_id=proxy.entity_id
                )
            )
            second_task = asyncio.create_task(
                conversation.async_converse(
                    hass, "B", None, Context(), language="en", agent_id=proxy.entity_id
                )
            )
            await asyncio.wait_for(both_started.wait(), timeout=2)
            gate.set()
            first, second = await asyncio.gather(first_task, second_task)

        assert first.conversation_id != second.conversation_id
        assert local.await_count == 2
        # A's explicit continuation suppresses local matching; B's false state
        # permits it, proving the state is isolated per chat session.
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=downstream),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(Agent, "async_write_ha_state"),
        ):
            await conversation.async_converse(
                hass,
                "A followup",
                first.conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )
            await conversation.async_converse(
                hass,
                "B followup",
                second.conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )
        assert local.await_count == 3

    _run(tmp_path, check)


def test_cancelled_local_and_fallback_are_propagated(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="cancel-test")

        async def wait_forever(*args, **kwargs):
            await asyncio.Event().wait()

        async def cancelled_by_caller(mock_target, *args):
            started = asyncio.Event()

            async def run_target():
                started.set()
                await mock_target(*args)

            task = asyncio.create_task(run_target())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with patch.object(
            conversation, "async_handle_intents", new=AsyncMock(side_effect=wait_forever)
        ):
            await cancelled_by_caller(
                lambda user_input, chat_log: entity._async_handle_message(user_input, chat_log),
                _input(),
                log,
            )

        registry = _registry("conversation.test_agent")
        hass.states.async_set("conversation.test_agent", "unknown")
        with (
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=AsyncMock(side_effect=wait_forever)),
        ):
            await cancelled_by_caller(
                lambda user_input, chat_log: entity._async_handle_message(user_input, chat_log),
                _input(),
                log,
            )

    _run(tmp_path, check)


def test_normalized_text_reaches_local_and_fallback_with_original_context(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        context = Context()
        user_input = conversation.ConversationInput(
            text="Original words",
            context=context,
            conversation_id="normalize-chat",
            device_id="device-x",
            satellite_id="satellite-x",
            language="ru",
            agent_id="conversation.morpholizer",
            extra_system_prompt="metadata",
        )
        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="normalize-chat")
        local = AsyncMock(return_value=None)
        fallback_result = conversation.ConversationResult(
            response=IntentResponse(language="ru"), conversation_id="normalize-chat"
        )
        forward = AsyncMock(return_value=fallback_result)
        registry = _registry("conversation.test_agent")
        hass.states.async_set("conversation.test_agent", "unknown")
        with (
            patch.object(entity, "_normalize_text", return_value="Normalized words"),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=forward),
        ):
            result = await entity._async_handle_message(user_input, log)

        assert result is fallback_result
        local_input = local.await_args.args[1]
        assert local_input.text == "Normalized words"
        assert local_input.context is context
        assert local_input.device_id == "device-x"
        assert local_input.satellite_id == "satellite-x"
        assert local_input.language == "ru"
        assert local_input.extra_system_prompt == "metadata"
        assert forward.await_args.kwargs["text"] == "Normalized words"
        assert forward.await_args.kwargs["context"] is context

    _run(tmp_path, check)


def test_fallback_exception_returns_terminal_error_once(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log

        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="fallback-error")
        fallback = AsyncMock(side_effect=RuntimeError("private details"))
        local = AsyncMock(return_value=None)
        hass.states.async_set("conversation.test_agent", "unknown")
        with (
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(er, "async_get", return_value=_registry("conversation.test_agent")),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(conversation, "async_converse", new=fallback),
        ):
            result = await entity._async_handle_message(_input(), log)

        assert result.response.error_code == IntentResponseErrorCode.FAILED_TO_HANDLE
        local.assert_awaited_once()
        fallback.assert_awaited_once()

    _run(tmp_path, check)


def test_entity_advertises_only_control_without_streaming(tmp_path: Path) -> None:
    from homeassistant.components.conversation.const import ConversationEntityFeature

    entity = MorpholizerConversationEntity(_entry())
    assert entity.supported_languages == ["ru"]
    assert entity.supported_features == ConversationEntityFeature.CONTROL
    assert not entity.supports_streaming


def test_failed_fallback_closes_pending_tool_call_without_leaking_exception(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation import chat_log
        from homeassistant.helpers import llm

        entity = MorpholizerConversationEntity(_entry())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = chat_log.ChatLog(hass=hass, conversation_id="failed-tool-session")
        hass.states.async_set("conversation.test_agent", "unknown")
        tool = llm.ToolInput(tool_name="HassTurnOn", tool_args={}, id="pending-tool")

        async def partial_failure(**kwargs):
            # Simulate a provider that fails after emitting a tool call.
            log.content.append(
                conversation.AssistantContent(agent_id="conversation.test_agent", tool_calls=[tool])
            )
            raise RuntimeError("private provider details")

        with (
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(er, "async_get", return_value=_registry("conversation.test_agent")),
            patch.object(conversation, "async_get_agent", return_value=object()),
            patch.object(
                conversation, "async_converse", new=AsyncMock(side_effect=partial_failure)
            ),
        ):
            result = await entity._async_handle_message(_input(), log)

        assert result.response.error_code == IntentResponseErrorCode.FAILED_TO_HANDLE
        assert log.content[-2].role == "tool_result"
        assert log.content[-2].tool_call_id == tool.id
        assert log.content[-2].result.error
        assert log.content[-1].role == "assistant"
        assert "private provider details" not in str(log.as_dict())

    _run(tmp_path, check)


def test_session_cleanup_clears_continuation_for_reused_non_ulid_id(tmp_path: Path) -> None:
    from homeassistant.helpers import chat_session

    class Agent(conversation.ConversationEntity):
        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Continuing")
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content="Continuing")
            )
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=True,
            )

    async def check(hass: HomeAssistant) -> None:
        conversation_id = "a-human-readable-session"
        proxy, downstream = MorpholizerConversationEntity(_entry()), Agent()
        proxy.entity_id = "conversation.morpholizer"
        downstream.entity_id = "conversation.test_agent"
        entities = {proxy.entity_id: proxy, downstream.entity_id: downstream}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(downstream.entity_id, "unknown")
        local = AsyncMock(return_value=None)
        registry = _registry(downstream.entity_id)
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=downstream),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(Agent, "async_write_ha_state"),
        ):
            first = await conversation.async_converse(
                hass,
                "First",
                conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )
            assert local.await_count == 1
            with chat_session.async_get_chat_session(
                hass, conversation_id, context=Context()
            ) as session:
                session.async_cleanup()
            await conversation.async_converse(
                hass,
                "After cleanup",
                conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )

        assert first.conversation_id == conversation_id
        assert local.await_count == 2

    _run(tmp_path, check)


def test_real_dispatch_logs_return_only_fallback_speech_once(tmp_path: Path) -> None:
    class ReturnOnlyAgent(conversation.ConversationEntity):
        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Returned, not written")
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
            )

    async def check(hass: HomeAssistant) -> None:
        from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS

        proxy, fallback = MorpholizerConversationEntity(_entry()), ReturnOnlyAgent()
        proxy.entity_id = "conversation.morpholizer"
        fallback.entity_id = "conversation.test_agent"
        entities = {proxy.entity_id: proxy, fallback.entity_id: fallback}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(fallback.entity_id, "unknown")
        with (
            patch.object(er, "async_get", return_value=_registry(fallback.entity_id)),
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(ReturnOnlyAgent, "async_write_ha_state"),
        ):
            result = await conversation.async_converse(
                hass,
                "Please answer",
                None,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )

        assert result.conversation_id is not None
        log = hass.data[DATA_CHAT_LOGS][result.conversation_id]
        assistant_messages = [message for message in log.content if message.role == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == "Returned, not written"
        assert assistant_messages[0].agent_id == fallback.entity_id

    _run(tmp_path, check)


def test_cancelled_continuation_is_consumed_and_next_turn_uses_local(tmp_path: Path) -> None:
    class Agent(conversation.ConversationEntity):
        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            if user_input.text == "Block":
                started.set()
                await asyncio.Event().wait()
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Continue")
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content="Continue")
            )
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=True,
            )

    async def check(hass: HomeAssistant) -> None:
        nonlocal started
        started = asyncio.Event()
        conversation_id = "cancel-followup-session"
        proxy, downstream = MorpholizerConversationEntity(_entry()), Agent()
        proxy.entity_id = "conversation.morpholizer"
        downstream.entity_id = "conversation.test_agent"
        entities = {proxy.entity_id: proxy, downstream.entity_id: downstream}
        for entity in entities.values():
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(get_entity=entities.get)
        hass.states.async_set(downstream.entity_id, "unknown")
        local = AsyncMock(return_value=None)
        registry = _registry(downstream.entity_id)
        with (
            patch.object(er, "async_get", return_value=registry),
            patch.object(conversation, "async_get_agent", return_value=downstream),
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(Agent, "async_write_ha_state"),
        ):
            await conversation.async_converse(
                hass, "First", conversation_id, Context(), language="en", agent_id=proxy.entity_id
            )
            blocked = asyncio.create_task(
                conversation.async_converse(
                    hass,
                    "Block",
                    conversation_id,
                    Context(),
                    language="en",
                    agent_id=proxy.entity_id,
                )
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            before = local.await_count
            await conversation.async_converse(
                hass,
                "After cancellation",
                conversation_id,
                Context(),
                language="en",
                agent_id=proxy.entity_id,
            )

        assert before == 1
        assert local.await_count == 2

    started = asyncio.Event()
    _run(tmp_path, check)
