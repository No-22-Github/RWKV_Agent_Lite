"""Tool registry for dynamic tool management."""

from dataclasses import dataclass
from typing import Any

from rvoone.agent.tools.base import Tool


@dataclass(frozen=True)
class ToolRegistration:
    tool: Tool
    category: str
    always_exposed: bool = False


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, ToolRegistration] = {}
        self._definitions_cache: dict[frozenset[str] | None, list[dict[str, Any]]] = {}

    def register(self, tool: Tool, *, category: str = "core", always_exposed: bool = False) -> None:
        """Register a tool."""
        self._tools[tool.name] = ToolRegistration(
            tool=tool,
            category=category,
            always_exposed=always_exposed,
        )
        self._definitions_cache.clear()

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._definitions_cache.clear()

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        registration = self._tools.get(name)
        return registration.tool if registration else None

    def _resolve_allowed_names(self, allowed_names: set[str] | None) -> set[str]:
        visible = {
            name for name, registration in self._tools.items() if registration.always_exposed
        }
        if allowed_names is not None:
            visible.update(allowed_names)
        else:
            visible.update(self._tools.keys())
        return visible

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self, allowed_names: set[str] | None = None) -> list[dict[str, Any]]:
        visible = self._resolve_allowed_names(allowed_names)
        cache_key = None if len(visible) == len(self._tools) else frozenset(visible)
        cached = self._definitions_cache.get(cache_key)
        if cached is not None:
            return cached

        definitions = [
            registration.tool.to_schema()
            for name, registration in self._tools.items()
            if name in visible
        ]
        self._definitions_cache[cache_key] = definitions
        return definitions

    def get_visible_tool_names(self, allowed_categories: set[str] | None = None) -> set[str]:
        if allowed_categories is None:
            return self._resolve_allowed_names(None)

        return {
            name
            for name, registration in self._tools.items()
            if registration.always_exposed or registration.category in allowed_categories
        }

    def list_tool_catalog(self, enabled_categories: set[str] | None = None) -> list[dict[str, Any]]:
        enabled_categories = enabled_categories or set()
        visible = self.get_visible_tool_names(enabled_categories)
        return [
            {
                "name": name,
                "category": registration.category,
                "always_exposed": registration.always_exposed,
                "enabled": name in visible,
                "description": registration.tool.description,
            }
            for name, registration in self._tools.items()
        ]

    def list_categories(self) -> list[str]:
        return sorted({registration.category for registration in self._tools.values()})

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        *,
        allowed_names: set[str] | None = None,
    ) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        registration = self._tools.get(name)
        if not registration:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        visible = self._resolve_allowed_names(allowed_names)
        if name not in visible:
            return (
                f"Error: Tool '{name}' is not currently enabled. "
                "Use list_tool_categories to inspect available categories and "
                "enable_tool_categories to unlock more tools." + _HINT
            )

        tool = registration.tool

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
