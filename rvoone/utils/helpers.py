"""Utility functions for rvoone."""

import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """~/.rvoone data directory."""
    return ensure_dir(Path.home() / ".rvoone")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure workspace path. Defaults to ~/.rvoone/workspace."""
    path = Path(workspace).expanduser() if workspace else Path.home() / ".rvoone" / "workspace"
    return ensure_dir(path)


def get_user_systemd_dir() -> Path:
    return ensure_dir(Path.home() / ".config" / "systemd" / "user")


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
    added: list[str] = []

    agents_path = workspace / "AGENTS.md"
    if not agents_path.exists():
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        agents_path.write_text(
            "# rvoone 🕊\n\n"
            "You are rvoone, a helpful AI assistant.\n\n"
            "Core rules:\n"
            "- Be concise and direct.\n"
            "- Do not claim tool results before receiving them.\n",
            encoding="utf-8",
        )
        added.append(str(agents_path.relative_to(workspace)))

    heartbeat_path = workspace / "HEARTBEAT.md"
    if not heartbeat_path.exists():
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(
            "# Heartbeat\n\n- [ ] No active tasks.\n",
            encoding="utf-8",
        )
        added.append(str(heartbeat_path.relative_to(workspace)))

    if added and not silent:
        from rich.console import Console

        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")
    return added
