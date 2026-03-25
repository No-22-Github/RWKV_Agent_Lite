"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_SystemPromptCacheKey = tuple[tuple[str, ...], bool, bool, str, str]


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES: list[str] = []
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._system_prompt_cache: dict[_SystemPromptCacheKey, str] = {}

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        enable_event_handling: bool = False,
        include_skills: bool = True,
    ) -> str:
        agents_signature, agents_prompt = self._read_agents_prompt()
        cache_key: _SystemPromptCacheKey = (
            tuple(skill_names or ()),
            enable_event_handling,
            include_skills,
            "chat",
            agents_signature,
        )
        cached = self._system_prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        if agents_prompt:
            parts = [agents_prompt, self._get_identity()]
        else:
            parts = [self._get_identity()]

        if enable_event_handling:
            parts.append(self._get_event_handling_directive())

        prompt = "\n---\n".join(parts)
        self._system_prompt_cache[cache_key] = prompt
        return prompt

    @staticmethod
    def _append_section(base: str, title: str, content: str) -> str:
        if not content:
            return base
        return f"{base}\n---\n# {title}\n{content}"

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"Runtime: {runtime}\nWorkspace: {workspace_path}"

    def _read_agents_prompt(self) -> tuple[str, str]:
        path = self.workspace / "AGENTS.md"
        try:
            stat = path.stat()
            signature = f"{stat.st_mtime_ns}:{stat.st_size}"
        except FileNotFoundError:
            return "missing", ""
        except OSError:
            return "unreadable", ""

        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            return signature, ""
        return signature, content

    @staticmethod
    def _get_event_handling_directive() -> str:
        """Get the event handling directive for system prompt."""
        return (
            "## Event Handling\n"
            "If you receive a <SYS_EVENT> message during tool execution:\n"
            "1. IMMEDIATELY acknowledge the event\n"
            "2. The event content ALWAYS takes priority over your current task\n"
            "3. Respond naturally to the event\n"
            "4. Decide whether to continue your previous task or switch to the new request\n"
        )

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        enable_event_handling: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        system_prompt = self.build_system_prompt(
            skill_names,
            enable_event_handling=enable_event_handling,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        messages.extend(history)

        messages.extend(
            [
                {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
                {"role": "user", "content": self._build_user_content(current_message, media)},
            ]
        )
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
