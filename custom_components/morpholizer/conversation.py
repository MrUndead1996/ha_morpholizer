"""Morpholizer conversation agent runtime."""

import logging
import math
from dataclasses import dataclass, replace
from typing import Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity
from homeassistant.components.conversation.const import ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, intent
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .catalog import CatalogSnapshot, EntityCatalog
from .const import (
    CONF_FALLBACK_AGENT,
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_MARGIN,
    CONF_FUZZY_SCORE_CUTOFF,
    DOMAIN,
    ENTITY_NAME,
    FUZZY_DEFAULT_ENABLED,
    FUZZY_DEFAULT_MARGIN,
    FUZZY_DEFAULT_SCORE_CUTOFF,
)
from .matching import FuzzySettings
from .morphology import Morphology
from .normalizer import NameCandidate, Normalizer

_LOGGER = logging.getLogger(__name__)


@dataclass
class _ContinuationState:
    """Pending fallback route for a chat session, if any."""

    fallback_id: str | None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Morpholizer conversation agent entity."""
    settings = _entry_fuzzy_settings(entry.options)
    # Keep the legacy no-argument construction path for old entries, whose
    # effective settings are exactly the Normalizer defaults.
    if settings == FuzzySettings():
        normalizer = await hass.async_add_executor_job(_load_normalizer)
    else:
        normalizer = await hass.async_add_executor_job(_load_normalizer, settings)
    async_add_entities([MorpholizerConversationEntity(entry, normalizer)])


def _load_normalizer(settings: FuzzySettings | None = None) -> Normalizer:
    """Load the dictionaries away from the event loop."""
    return Normalizer(Morphology(), fuzzy=settings or FuzzySettings())


def _entry_fuzzy_settings(options: dict[str, object]) -> FuzzySettings:
    """Read settings defensively; corrupted options disable fuzzy matching."""
    enabled = options.get(CONF_FUZZY_ENABLED, FUZZY_DEFAULT_ENABLED)
    score = options.get(CONF_FUZZY_SCORE_CUTOFF, FUZZY_DEFAULT_SCORE_CUTOFF)
    margin = options.get(CONF_FUZZY_MARGIN, FUZZY_DEFAULT_MARGIN)
    if type(enabled) is not bool or isinstance(score, bool) or isinstance(margin, bool):
        return FuzzySettings()
    if not isinstance(score, (int, float)) or not isinstance(margin, (int, float)):
        return FuzzySettings()
    try:
        score = float(score)
        margin = float(margin)
        if not math.isfinite(score) or not math.isfinite(margin):
            return FuzzySettings()
        return FuzzySettings(enabled, score, margin)
    except TypeError, ValueError, OverflowError:
        return FuzzySettings()


class MorpholizerConversationEntity(ConversationEntity):
    """Handle HA intents locally before delegating unmatched input to an LLM."""

    _attr_has_entity_name = True
    _attr_name = ENTITY_NAME
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry, normalizer: Normalizer | None = None) -> None:
        """Initialize this agent from its config entry."""
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._normalizer = normalizer
        self._continuations: dict[str, _ContinuationState] = {}
        self.async_on_remove(self._continuations.clear)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return the integration's supported languages."""
        return ["ru"]

    @callback
    def _async_error_result(
        self, language: str, chat_log: conversation.ChatLog, message: str
    ) -> conversation.ConversationResult:
        """Return a safe error and record its sole assistant reply in the log."""
        response = intent.IntentResponse(language=language)
        response.async_set_error(
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            message,
        )
        # Close any unfinished tool calls before adding the safe error reply.
        # Never pass the original exception: it can contain private user data.
        chat_log.async_add_error(self.entity_id, HomeAssistantError(message))
        return conversation.ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Identity path for an entity constructed without a loaded normalizer."""
        return text

    def _normalize_snapshot(self, text: str, snapshot: CatalogSnapshot) -> str:
        """Build name candidates and run morphology entirely in the executor."""
        candidates = tuple(
            NameCandidate(entry.entity_id, name)
            for entry in snapshot.entries
            for name in entry.names
        ) + tuple(
            NameCandidate(location.id, name, kind=location.kind)
            for location in snapshot.locations
            for name in location.names
        )
        assert self._normalizer is not None
        return self._normalizer.normalize(text, candidates).text

    async def _async_normalize(self, text: str, language: str) -> str:
        if self._normalizer is None:
            return self._normalize_text(text)
        if language.split("-", 1)[0] != "ru":
            return text
        catalog: EntityCatalog = self._entry.runtime_data
        # Exposure or names can change while an executor job is running. Retry
        # once on a new generation; never apply a result from a stale snapshot.
        for _ in range(2):
            snapshot = catalog.snapshot
            normalized = await self.hass.async_add_executor_job(
                self._normalize_snapshot, text, snapshot
            )
            if catalog.snapshot is snapshot:
                return normalized
        return text

    @staticmethod
    def _update_user_text(chat_log: conversation.ChatLog, original: str, text: str) -> None:
        """Keep one user message when nested dispatch receives normalized text."""
        if original == text:
            return
        message = chat_log.content[-1]
        if isinstance(message, conversation.UserContent) and message.content == original:
            # UserContent is immutable. Preserve attachments and creation time;
            # the request/trace still retains the original text independently.
            normalized_message = replace(message, content=text)
            # `created` has init=False: dataclasses.replace runs its factory
            # again. Restore it on our new, unpublished record, not the original.
            object.__setattr__(normalized_message, "created", message.created)
            chat_log.content[-1] = normalized_message

    def _async_take_continuation(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> str | None:
        """Consume this conversation's pending fallback before any await."""
        session = chat_session.current_session.get()
        if session is not None and session.conversation_id == chat_log.conversation_id:
            state = self._continuations.get(chat_log.conversation_id)
            if state is None:
                return None
            fallback_id = state.fallback_id
            state.fallback_id = None
            return fallback_id
        with chat_session.async_get_chat_session(
            self.hass, chat_log.conversation_id, context=user_input.context
        ):
            state = self._continuations.get(chat_log.conversation_id)
            if state is None:
                return None
            fallback_id = state.fallback_id
            state.fallback_id = None
            return fallback_id

    def _async_store_continuation(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        fallback_id: str,
    ) -> None:
        """Store continuation state until the owning chat session is cleaned up."""
        conversation_id = chat_log.conversation_id
        if state := self._continuations.get(conversation_id):
            state.fallback_id = fallback_id
            return

        session = chat_session.current_session.get()
        if session is not None and session.conversation_id == conversation_id:
            session.async_on_cleanup(lambda: self._continuations.pop(conversation_id, None))
            self._continuations[conversation_id] = _ContinuationState(fallback_id)
            return
        with chat_session.async_get_chat_session(
            self.hass, conversation_id, context=user_input.context
        ) as session:
            session.async_on_cleanup(lambda: self._continuations.pop(conversation_id, None))
            self._continuations[conversation_id] = _ContinuationState(fallback_id)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Try local intents first, then delegate a genuine miss exactly once."""
        # An explicitly continued LLM turn belongs to that agent; do not let
        # local intent matching steal its response.
        # Consume before any asynchronous work: cancellation must not leave a
        # stale route that captures a later, unrelated turn.
        continuation_agent = self._async_take_continuation(user_input, chat_log)
        text = user_input.text
        if continuation_agent is None:
            try:
                text = await self._async_normalize(text, user_input.language)
            except Exception:  # noqa: BLE001
                _LOGGER.error("Entity name normalization failed")
                return self._async_error_result(
                    user_input.language, chat_log, "Не удалось обработать название устройства."
                )
            self._update_user_text(chat_log, user_input.text, text)
            normalized_input = replace(
                user_input, text=text, conversation_id=chat_log.conversation_id
            )
            try:
                # Intent filter intentionally omitted. HA's native
                # prefer_local_intents CONTROL-agent filter excludes
                # HassGetState and HassMediaSearchAndPlay; Morpholizer instead
                # handles all standard intents locally, including turn_on/off.
                local_response = await conversation.async_handle_intents(
                    self.hass, normalized_input, chat_log, intent_filter=None
                )
            except Exception:  # noqa: BLE001
                _LOGGER.error("Local intent handling failed")
                return self._async_error_result(
                    user_input.language,
                    chat_log,
                    "Не удалось выполнить локальную команду.",
                )
            if local_response is not None:
                speech = local_response.speech.get("plain", {}).get("speech", "")
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content=speech,
                    )
                )
                return conversation.ConversationResult(
                    response=local_response,
                    conversation_id=chat_log.conversation_id,
                    continue_conversation=False,
                )

        fallback_id = continuation_agent or self._entry.options.get(
            CONF_FALLBACK_AGENT,
            self._entry.data.get(CONF_FALLBACK_AGENT),
        )
        if not fallback_id or fallback_id == self.entity_id:
            return self._async_error_result(
                user_input.language,
                chat_log,
                "Резервный разговорный агент недоступен.",
            )

        # Verify both the registry owner and loaded/available agent. This blocks
        # stale options, self-routing, and registry entries from this platform.
        registry_entry = er.async_get(self.hass).async_get(fallback_id)
        state = self.hass.states.get(fallback_id)
        if (
            registry_entry is None
            or registry_entry.platform == DOMAIN
            or fallback_id.partition(".")[0] != conversation.DOMAIN
            or state is None
            or state.state == STATE_UNAVAILABLE
            or conversation.async_get_agent(self.hass, fallback_id) is None
        ):
            return self._async_error_result(
                user_input.language,
                chat_log,
                "Резервный разговорный агент недоступен.",
            )

        # ConversationEntity.async_process already opened the caller's chat session
        # and log. Reuse its ID so downstream writes to that same log; this wrapper
        # returns the downstream result unchanged and adds no second assistant reply.
        try:
            result = await conversation.async_converse(
                hass=self.hass,
                text=text,
                conversation_id=chat_log.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=fallback_id,
                device_id=user_input.device_id,
                satellite_id=user_input.satellite_id,
                extra_system_prompt=user_input.extra_system_prompt,
            )
            # Some agents return speech without writing to HA's chat log. Match
            # the pipeline's compatibility path without duplicating chat-log agents.
            if chat_log.content[-1].role == "user" and (
                speech := result.response.speech.get("plain", {}).get("speech", "")
            ):
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(agent_id=fallback_id, content=speech)
                )
            if result.continue_conversation:
                self._async_store_continuation(user_input, chat_log, fallback_id)
            return result
        except Exception:  # noqa: BLE001
            # Never retry locally-executed work, and do not leak exception/user
            # text into the response or logs.
            _LOGGER.error("Fallback conversation agent failed")
            return self._async_error_result(
                user_input.language,
                chat_log,
                "Не удалось получить ответ от резервного агента.",
            )
