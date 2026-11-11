"""End-to-end coverage for morphology in the conversation runtime."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("homeassistant")
pytest.importorskip("pymorphy3")

from homeassistant.components import conversation
from homeassistant.components.conversation.agent_manager import get_agent_manager
from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS, ChatLog
from homeassistant.components.conversation.default_agent import DefaultAgent
from homeassistant.components.homeassistant.exposed_entities import (
    DATA_EXPOSED_ENTITIES,
    ExposedEntities,
    async_expose_entity,
)
from homeassistant.components.intent import OnOffIntentHandler
from homeassistant.config_entries import SOURCE_USER, ConfigEntries, ConfigEntry
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry, device_registry, floor_registry, intent
from homeassistant.helpers import entity_registry as er

from custom_components.morpholizer.catalog import EntityCatalog
from custom_components.morpholizer.const import CONF_FALLBACK_AGENT, DOMAIN
from custom_components.morpholizer.conversation import (
    MorpholizerConversationEntity,
)
from custom_components.morpholizer.conversation import (
    async_setup_entry as async_setup_conversation_entry,
)
from custom_components.morpholizer.morphology import Morphology
from custom_components.morpholizer.normalizer import Normalizer


def _entry() -> ConfigEntry:
    return ConfigEntry(
        domain=DOMAIN,
        title="Morpholizer",
        data={CONF_FALLBACK_AGENT: "conversation.test_agent"},
        options={},
        source=SOURCE_USER,
        version=1,
        minor_version=1,
        unique_id="normalization-runtime",
        discovery_keys={},
        subentries_data=(),
    )


def _run(tmp_path: Path, check) -> None:
    async def run() -> None:
        hass = HomeAssistant(str(tmp_path))
        hass.config_entries = ConfigEntries(hass, {})
        hass.config.language = "ru"
        try:
            await hass.config_entries.async_initialize()
            await check(hass)
        finally:
            await hass.async_stop(force=True)

    asyncio.run(run())


async def _catalog(hass: HomeAssistant, entity_id: str, name: str) -> EntityCatalog:
    device_registry.async_setup(hass)
    await device_registry.async_load(hass, load_empty=True)
    await area_registry.async_load(hass, load_empty=True)
    await floor_registry.async_load(hass, load_empty=True)
    await er.async_load(hass, load_empty=True)
    exposure = ExposedEntities(hass)
    hass.data[DATA_EXPOSED_ENTITIES] = exposure
    await exposure.async_initialize()
    registry = er.async_get(hass)
    registry.async_get_or_create(
        entity_id.split(".", 1)[0],
        "test",
        entity_id,
        suggested_object_id=entity_id.split(".", 1)[1],
        original_name=name,
    )
    hass.states.async_set(entity_id, "off", {"friendly_name": name})
    async_expose_entity(hass, "conversation", entity_id, True)
    catalog = EntityCatalog(hass)
    await catalog.async_start()
    return catalog


def test_real_local_intent_normalizes_name_and_dispatches_once(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        catalog = await _catalog(hass, "switch.washer", "Стиральная машинка")
        entry = _entry()
        entry.runtime_data = catalog
        _ = catalog.snapshot
        proxy = MorpholizerConversationEntity(entry, Normalizer(Morphology()))
        proxy.entity_id = "conversation.morpholizer"
        proxy.hass = hass
        await get_agent_manager(hass).async_setup_default_agent(DefaultAgent(hass))
        intent.async_register(
            hass, OnOffIntentHandler(intent.INTENT_TURN_ON, "homeassistant", "turn_on")
        )
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(
            get_entity=lambda entity_id: proxy if entity_id == proxy.entity_id else None
        )
        calls = []

        async def turn_on(call):
            calls.append(call)
            hass.states.async_set("switch.washer", "on")

        hass.services.async_register("switch", "turn_on", turn_on)
        hass.services.async_register("homeassistant", "turn_on", turn_on)
        context = Context()
        with (
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
        ):
            result = await conversation.async_converse(
                hass,
                "Включи стиральную машинку",
                None,
                context,
                language="ru",
                agent_id=proxy.entity_id,
            )
        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
        assert hass.states.get("switch.washer").state == "on"
        assert len(calls) == 1
        log = hass.data[DATA_CHAT_LOGS][result.conversation_id]
        assert [m.content for m in log.content if m.role == "user"] == ["Включи Стиральная машинка"]
        assert len([m for m in log.content if m.role == "assistant" and m.content is not None]) == 1
        assert calls[0].context is context
        catalog.async_stop()

    _run(tmp_path, check)


def test_stale_catalog_generation_retries_without_removed_candidate(tmp_path: Path) -> None:
    class BlockingNormalizer:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.candidates = []

        def normalize(self, text, candidates):
            self.candidates.append(tuple(candidates))
            if len(self.candidates) == 1:
                self.started.set()
                assert self.release.wait(5)
            return SimpleNamespace(text="normalized" if candidates else text)

    async def check(hass: HomeAssistant) -> None:
        catalog = await _catalog(hass, "switch.washer", "Стиральная машинка")
        entry = _entry()
        entry.runtime_data = catalog
        normalizer = BlockingNormalizer()
        entity = MorpholizerConversationEntity(entry, normalizer)
        entity.hass = hass
        worker = asyncio.create_task(entity._async_normalize("машинку", "ru"))
        try:
            assert await asyncio.to_thread(normalizer.started.wait, 3)
            async_expose_entity(hass, "conversation", "switch.washer", False)
            assert catalog.snapshot.entries == ()
            normalizer.release.set()
            assert await asyncio.wait_for(worker, 5) == "машинку"
            assert len(normalizer.candidates) == 2
            assert normalizer.candidates[0] and not normalizer.candidates[1]
        finally:
            normalizer.release.set()
            if not worker.done():
                await worker
            catalog.async_stop()

    _run(tmp_path, check)


def test_normalizer_initialization_runs_in_executor(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        entry = _entry()
        loop_thread = threading.get_ident()
        observed = []
        normalizer = object()
        with (
            patch(
                "custom_components.morpholizer.conversation._load_normalizer",
                side_effect=lambda: (observed.append(threading.get_ident()), normalizer)[1],
            ),
            patch(
                "custom_components.morpholizer.conversation.MorpholizerConversationEntity"
            ) as entity_cls,
        ):
            await async_setup_conversation_entry(hass, entry, lambda entities: None)
        assert observed and observed[0] != loop_thread
        entity_cls.assert_called_once_with(entry, normalizer)

    _run(tmp_path, check)


def test_real_dispatch_followup_bypasses_normalization_and_keeps_one_message_each_turn(
    tmp_path: Path,
) -> None:
    class EchoAgent(conversation.ConversationEntity):
        def __init__(self):
            self.inputs = []

        @property
        def supported_languages(self):
            return ["*"]

        async def _async_handle_message(self, user_input, chat_log):
            self.inputs.append(user_input)
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_speech(user_input.text)
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content=user_input.text)
            )
            return conversation.ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=user_input.text == "Стиральная машинка",
            )

    async def check(hass: HomeAssistant) -> None:
        catalog = await _catalog(hass, "switch.washer", "Стиральная машинка")
        entry = _entry()
        entry.runtime_data = catalog
        proxy, downstream = (
            MorpholizerConversationEntity(entry, Normalizer(Morphology())),
            EchoAgent(),
        )
        proxy.entity_id, downstream.entity_id = (
            "conversation.morpholizer",
            "conversation.test_agent",
        )
        for entity in (proxy, downstream):
            entity.hass = hass
        hass.data[conversation.DATA_COMPONENT] = SimpleNamespace(
            get_entity=lambda entity_id: {
                proxy.entity_id: proxy,
                downstream.entity_id: downstream,
            }.get(entity_id)
        )
        er.async_get(hass).async_get_or_create(
            "conversation",
            "test",
            downstream.entity_id,
            suggested_object_id="test_agent",
        )
        hass.states.async_set(downstream.entity_id, "unknown")
        context = Context()
        with (
            patch.object(conversation, "async_handle_intents", new=AsyncMock(return_value=None)),
            patch.object(MorpholizerConversationEntity, "async_write_ha_state"),
            patch.object(EchoAgent, "async_write_ha_state"),
        ):
            first = await conversation.async_converse(
                hass,
                "Стиральной машинкой",
                None,
                context,
                language="ru",
                agent_id=proxy.entity_id,
                device_id="test-device",
                satellite_id="assist_satellite.test",
                extra_system_prompt="metadata",
            )
            second = await conversation.async_converse(
                hass,
                "Стиральной машинкой",
                first.conversation_id,
                Context(),
                language="ru",
                agent_id=proxy.entity_id,
            )
            third = await conversation.async_converse(
                hass,
                "Стиральной машинкой",
                second.conversation_id,
                context,
                language="ru",
                agent_id=proxy.entity_id,
            )
        assert first.conversation_id == second.conversation_id
        assert third.conversation_id == first.conversation_id
        assert first.response.error_code is None
        assert first.continue_conversation and not second.continue_conversation
        assert third.continue_conversation
        assert downstream.inputs[0].context is context
        assert downstream.inputs[0].device_id == "test-device"
        assert downstream.inputs[0].satellite_id == "assist_satellite.test"
        assert downstream.inputs[0].extra_system_prompt == "metadata"
        log = hass.data[DATA_CHAT_LOGS][first.conversation_id]
        assert [m.role for m in log.content] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [m.content for m in log.content if m.role == "user"] == [
            "Стиральная машинка",
            "Стиральной машинкой",
            "Стиральная машинка",
        ]
        assert [m.content for m in log.content if m.role == "assistant"] == [
            "Стиральная машинка",
            "Стиральной машинкой",
            "Стиральная машинка",
        ]
        catalog.async_stop()

    _run(tmp_path, check)


def test_cancelled_normalization_never_dispatches_or_falls_back(tmp_path: Path) -> None:
    class BlockingNormalizer:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def normalize(self, text, candidates):
            self.started.set()
            assert self.release.wait(5)
            return SimpleNamespace(text=text)

    async def check(hass: HomeAssistant) -> None:
        catalog = await _catalog(hass, "switch.washer", "Стиральная машинка")
        entry = _entry()
        entry.runtime_data = catalog
        normalizer = BlockingNormalizer()
        entity = MorpholizerConversationEntity(entry, normalizer)
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        local, fallback = AsyncMock(), AsyncMock()
        with (
            patch.object(conversation, "async_handle_intents", new=local),
            patch.object(conversation, "async_converse", new=fallback),
        ):
            task = asyncio.create_task(
                entity._async_handle_message(
                    conversation.ConversationInput(
                        text="машинку",
                        context=Context(),
                        conversation_id="cancel-normalize",
                        device_id=None,
                        satellite_id=None,
                        language="ru",
                        agent_id=entity.entity_id,
                    ),
                    ChatLog(hass=hass, conversation_id="cancel-normalize"),
                )
            )
            try:
                assert await asyncio.to_thread(normalizer.started.wait, 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                local.assert_not_awaited()
                fallback.assert_not_awaited()
            finally:
                normalizer.release.set()
                catalog.async_stop()

    _run(tmp_path, check)


def test_user_text_replacement_preserves_original_message_metadata(tmp_path: Path) -> None:
    async def check(hass: HomeAssistant) -> None:
        original = conversation.UserContent(content="Стиральной машинкой", attachments=[])
        log = ChatLog(hass=hass, conversation_id="metadata-test")
        log.content.append(original)
        MorpholizerConversationEntity._update_user_text(log, original.content, "Стиральная машинка")
        assert original.content == "Стиральной машинкой"
        assert len(log.content) == 2
        assert log.content[-1].content == "Стиральная машинка"
        assert log.content[-1].created is original.created
        assert log.content[-1].attachments is original.attachments

    _run(tmp_path, check)


def test_normalization_error_is_safe_and_does_not_execute_or_fallback(tmp_path: Path) -> None:
    class BrokenNormalizer:
        def normalize(self, text, candidates):
            raise RuntimeError("private analysis details")

    async def check(hass: HomeAssistant) -> None:
        catalog = await _catalog(hass, "switch.washer", "Стиральная машинка")
        entry = _entry()
        entry.runtime_data = catalog
        entity = MorpholizerConversationEntity(entry, BrokenNormalizer())
        entity.hass = hass
        entity.entity_id = "conversation.morpholizer"
        log = ChatLog(hass=hass, conversation_id="broken-normalization")
        user_input = conversation.ConversationInput(
            text="Включи стиральную машинку",
            context=Context(),
            conversation_id=log.conversation_id,
            device_id=None,
            satellite_id=None,
            language="ru",
            agent_id=entity.entity_id,
        )
        with (
            patch.object(conversation, "async_handle_intents", new=AsyncMock()) as local,
            patch.object(conversation, "async_converse", new=AsyncMock()) as fallback,
        ):
            result = await entity._async_handle_message(user_input, log)
        assert result.response.error_code == intent.IntentResponseErrorCode.FAILED_TO_HANDLE
        assert "private analysis details" not in str(result.as_dict())
        assert "private analysis details" not in str(log.as_dict())
        assert user_input.text == "Включи стиральную машинку"
        local.assert_not_awaited()
        fallback.assert_not_awaited()
        catalog.async_stop()

    _run(tmp_path, check)
