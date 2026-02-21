"""Agent lifecycle callback functions for monitoring and memory.

This module provides callback functions that execute at various stages of the
agent lifecycle. These callbacks enable comprehensive logging and session
memory persistence.
"""

import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)
SELECTED_MODEL_STATE_KEY = "selectedModel"


async def add_session_to_memory(callback_context: CallbackContext) -> None:
    """Automatically save completed sessions to memory bank.

    This callback checks if the invocation context has a memory service.
    If so, it saves the session to memory for future retrieval.

    Args:
        callback_context: The callback context with access to invocation context
    """
    logger.info("*** Starting add_session_to_memory callback ***")
    try:
        await callback_context.add_session_to_memory()
    except ValueError as e:
        logger.warning(e)
    except Exception as e:
        logger.warning(f"Failed to add session to memory: {type(e).__name__}: {e}")

    return None


class LoggingCallbacks:
    """Provides comprehensive logging callbacks for ADK agent lifecycle events.

    This class groups all agent lifecycle callback methods together and supports
    logger injection following the strategy pattern. All callbacks are
    non-intrusive and return None.

    Attributes:
        logger: Logger instance for recording agent lifecycle events.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize logging callbacks with optional logger.

        Args:
            logger: Optional logger instance. If not provided, creates one
                   using the module name.
        """
        if logger is None:
            logger = logging.getLogger(self.__class__.__module__)
        self.logger = logger

    def _read_selected_model_from_state(
        self,
        callback_context: CallbackContext,
    ) -> str | None:
        state_data = callback_context.state.to_dict()
        selected_model = state_data.get(SELECTED_MODEL_STATE_KEY)

        if not isinstance(selected_model, str):
            return None

        normalized_selected_model = selected_model.strip()
        if not normalized_selected_model:
            return None

        return normalized_selected_model

    def _apply_selected_model_to_request(
        self,
        llm_request: LlmRequest,
        selected_model: str,
    ) -> None:
        # LiteLLM expects openrouter/provider/model-name to route to OpenRouter.
        # OpenRouter API returns ids like arcee-ai/trinity-large-preview:free
        # without the openrouter/ prefix, so we add it when missing.
        model_for_request = selected_model
        if not model_for_request.lower().startswith("openrouter/"):
            model_for_request = f"openrouter/{model_for_request}"

        llm_request.model = model_for_request  # type: ignore[attr-defined]
        return None

    def _format_model_for_log(self, model_value: Any) -> str:
        if isinstance(model_value, str):
            return model_value

        lite_llm_model_name = getattr(model_value, "model", None)
        if isinstance(lite_llm_model_name, str):
            return lite_llm_model_name

        return repr(model_value)

    def _get_allowed_model_ids(self) -> set[str]:
        from .openrouter import get_cached_model_ids

        return get_cached_model_ids()

    def before_agent(self, callback_context: CallbackContext) -> None:
        """Callback executed before agent processing begins.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
        """
        self.logger.info(
            f"*** Starting agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        return None

    def after_agent(self, callback_context: CallbackContext) -> None:
        """Callback executed after agent processing completes.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
        """
        self.logger.info(
            f"*** Leaving agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        return None

    def before_model(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        """Callback executed before LLM model invocation.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
            llm_request (LlmRequest): The request being sent to the LLM model
                containing message contents.
        """
        self.logger.info(
            f"*** Before LLM call for agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        selected_model = self._read_selected_model_from_state(callback_context)
        if selected_model is None:
            self.logger.info(
                "No selected model in state; using request model: %s",
                self._format_model_for_log(llm_request.model),
            )
        else:
            allowed_ids = self._get_allowed_model_ids()
            is_allowed = selected_model in allowed_ids
            if not is_allowed and not allowed_ids:
                is_allowed = "/" in selected_model
                if is_allowed:
                    self.logger.info(
                        "Model list unavailable; allowing provider-prefixed model: %s",
                        selected_model,
                    )
            if not is_allowed:
                self.logger.warning(
                    "Selected model '%s' not in allowed list; using default model",
                    selected_model,
                )
            else:
                self.logger.info("Requested model from state: %s", selected_model)
                try:
                    self._apply_selected_model_to_request(llm_request, selected_model)
                    self.logger.info(
                        "Applied model for this request: %s",
                        self._format_model_for_log(llm_request.model),
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to apply selected model '%s': %s: %s",
                        selected_model,
                        type(e).__name__,
                        e,
                    )

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        self.logger.debug(f"LLM request contains {len(llm_request.contents)} messages:")
        for i, content in enumerate(llm_request.contents, start=1):
            self.logger.debug(
                f"Content {i}: {content.model_dump(exclude_none=True, mode='json')}"
            )

        return None

    def after_model(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """Callback executed after LLM model responds.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
            llm_response (LlmResponse): The response received from the LLM model.
        """
        self.logger.info(
            f"*** After LLM call for agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        if llm_content := llm_response.content:
            response_data = llm_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"LLM response: {response_data}")

        return None

    def before_tool(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        """Callback executed before tool invocation.

        Args:
            tool (BaseTool): The tool being invoked.
            args (dict[str, Any]): Arguments being passed to the tool.
            tool_context (ToolContext): Context containing agent name, invocation ID,
                state, user content, and event actions.
        """
        self.logger.info(
            f"*** Before invoking tool '{tool.name}' in agent "
            f"'{tool_context.agent_name}' with invocation_id "
            f"'{tool_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {tool_context.state.to_dict().keys()}")

        if content := tool_context.user_content:
            self.logger.debug(
                f"User Content: {content.model_dump(exclude_none=True, mode='json')}"
            )

        actions_data = tool_context.actions.model_dump(exclude_none=True, mode="json")
        self.logger.debug(f"EventActions: {actions_data}")
        self.logger.debug(f"args: {args}")

        return None

    def after_tool(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[str, Any],
    ) -> None:
        """Callback executed after tool invocation completes.

        Args:
            tool (BaseTool): The tool that was invoked.
            args (dict[str, Any]): Arguments that were passed to the tool.
            tool_context (ToolContext): Context containing agent name, invocation ID,
                state, user content, and event actions.
            tool_response (dict[str, Any]): The response returned by the tool.
        """
        self.logger.info(
            f"*** After invoking tool '{tool.name}' in agent "
            f"'{tool_context.agent_name}' with invocation_id "
            f"'{tool_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {tool_context.state.to_dict().keys()}")

        if content := tool_context.user_content:
            self.logger.debug(
                f"User Content: {content.model_dump(exclude_none=True, mode='json')}"
            )

        actions_data = tool_context.actions.model_dump(exclude_none=True, mode="json")
        self.logger.debug(f"EventActions: {actions_data}")
        self.logger.debug(f"args: {args}")
        self.logger.debug(f"Tool response: {tool_response}")

        return None
