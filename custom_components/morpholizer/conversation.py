"""Conversation agent scaffold for Morpholizer."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_FALLBACK_AGENT, DOMAIN, ENTITY_NAME


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Morpholizer conversation agent entity."""
    async_add_entities([MorpholizerConversationEntity(entry)])


class MorpholizerConversationEntity(ConversationEntity):
    """Transparent conversation-agent wrapper; no local processing is performed."""

    _attr_has_entity_name = True
    _attr_name = ENTITY_NAME

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize this agent from its config entry."""
        self._entry = entry
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return the language supported by this scaffold."""
        return ["ru"]

    @callback
    def _async_error_result(
        self, language: str, chat_log: conversation.ChatLog
    ) -> conversation.ConversationResult:
        """Return an error result and record its sole assistant reply in the log."""
        message = "Morpholizer fallback conversation agent is unavailable."
        response = intent.IntentResponse(language=language)
        response.async_set_error(
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            message,
        )
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=message,
            )
        )
        return conversation.ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
        )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Forward input unchanged to the selected fallback; do not normalize it."""
        fallback_id = self._entry.options.get(
            CONF_FALLBACK_AGENT,
            self._entry.data.get(CONF_FALLBACK_AGENT),
        )
        if not fallback_id or fallback_id == self.entity_id:
            return self._async_error_result(user_input.language, chat_log)

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
            return self._async_error_result(user_input.language, chat_log)

        # ConversationEntity.async_process already opened the caller's chat session
        # and log. Reuse its ID so downstream writes to that same log; this wrapper
        # returns the downstream result unchanged and adds no second assistant reply.
        try:
            return await conversation.async_converse(
                hass=self.hass,
                text=user_input.text,
                conversation_id=chat_log.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=fallback_id,
                device_id=user_input.device_id,
                satellite_id=user_input.satellite_id,
                extra_system_prompt=user_input.extra_system_prompt,
            )
        except ValueError:
            # The agent may disappear between registry validation and dispatch.
            return self._async_error_result(user_input.language, chat_log)
