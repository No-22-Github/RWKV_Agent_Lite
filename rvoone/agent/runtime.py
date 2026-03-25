"""Agent runtime execution engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from rvoone.agent.loop import AgentLoop


class AgentRuntime:
    """Execute the model/tool loop independently from message transport."""

    _EMPTY_DIRECT_REPLY = "Sorry, I generated an empty response. Please try again."

    def __init__(self, owner: "AgentLoop"):
        self.owner = owner

    async def run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
        session_key: str | None = None,
        tool_status: dict[str, Any] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        owner = self.owner
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        async def inject_event() -> bool:
            """Inject queued user events into the prompt as a high-priority system message."""
            if not session_key:
                return False
            event = await owner.bus.check_events(session_key)
            if not event:
                return False
            messages.append(
                {
                    "role": "system",
                    "content": f'<SYS_EVENT type="user_interrupt">{event}</SYS_EVENT>',
                }
            )
            return True

        while iteration < owner.max_iterations:
            iteration += 1
            visible_tool_names = owner._get_visible_tool_names(session_key)

            interrupted = await inject_event()
            if session_key:
                owner._update_session_runtime(session_key, phase="waiting_for_model")

            saw_text_delta = False

            async def _handle_text_delta(delta: str) -> None:
                nonlocal saw_text_delta
                saw_text_delta = True
                if on_text_delta:
                    await on_text_delta(delta)

            response = await owner.model_gateway.chat(
                messages=messages,
                tools=owner.tools.get_definitions(visible_tool_names),
                temperature=owner.temperature,
                max_tokens=owner.max_tokens,
                reasoning_effort=owner.reasoning_effort,
                on_text_delta=_handle_text_delta if on_text_delta else None,
            )

            clean = owner._strip_think(response.content)

            if response.has_tool_calls:
                if on_progress:
                    if clean and not saw_text_delta:
                        await on_progress(clean)
                    await on_progress(owner._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = owner.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                if interrupted:
                    logger.info("Interrupt already queued for this turn, cancelling tool execution")
                    for tc in response.tool_calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "content": "CANCELLED: User interrupted",
                            }
                        )
                    continue

                terminal_called = False
                for i, tool_call in enumerate(response.tool_calls):
                    interrupted = await inject_event()
                    if interrupted:
                        logger.info(
                            "Event received during tool execution, cancelling remaining tools"
                        )
                        for tc in response.tool_calls[i:]:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": tc.name,
                                    "content": "CANCELLED: User interrupted",
                                }
                            )
                        break

                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    if session_key:
                        owner._update_session_runtime(
                            session_key,
                            phase="running_tool",
                            current_tool=tool_call.name,
                            current_tool_args=args_str[:120],
                        )
                    if tool_status and tool_status.get("enabled") and tool_status.get("key"):
                        text = owner.presenter.format_tool_status(
                            tool_call.name, tool_call.arguments
                        )
                        if not tool_status.get("created"):
                            await owner.presenter.publish_tool_status_create(
                                tool_status["channel"],
                                tool_status["chat_id"],
                                tool_status["key"],
                                text,
                                pin_in_chat=bool(tool_status.get("pin_enabled")),
                            )
                            tool_status["created"] = True
                        else:
                            await owner.presenter.publish_tool_status_update(
                                tool_status["channel"],
                                tool_status["chat_id"],
                                tool_status["key"],
                                text,
                            )
                    result = await owner.tools.execute(
                        tool_call.name,
                        tool_call.arguments,
                        allowed_names=visible_tool_names,
                    )
                    messages = owner.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = owner.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean or self._EMPTY_DIRECT_REPLY
                if clean is None:
                    logger.warning("Model returned an empty direct reply; using fallback reply")
                break
        if final_content is None and iteration >= owner.max_iterations:
            logger.warning("Max iterations ({}) reached", owner.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({owner.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages
