import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rvoone.cli.commands import app
from rvoone.config.loader import load_config
from rvoone.config.schema import Config, ProviderConfig

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with (
        patch("rvoone.config.loader.get_config_path") as mock_cp,
        patch("rvoone.config.loader.save_config_templates") as mock_sc,
        patch("rvoone.config.loader.load_config") as mock_lc,
        patch("rvoone.cli.commands.get_workspace_path") as mock_ws,
    ):
        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_dir = base_dir / "config"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_dir
        mock_ws.return_value = workspace_dir

        def _write_templates(config, config_path=None):
            target = config_path or config_dir
            target.mkdir(parents=True, exist_ok=True)
            (target / "agent.toml").write_text(
                '[defaults]\nmodel = "gpt-4o-mini"\n', encoding="utf-8"
            )
            (target / "chat.toml").write_text("sendProgress = true\n", encoding="utf-8")
            (target / "llm.toml").write_text(
                '# LLM settings.\n\nupstreamTimeout = 60\n\n# [customSources.example]\n# apiBase = "http://localhost:8000/v1"\n# apiKey = "no-key"\n\n[groq]\napiKey = ""\n',
                encoding="utf-8",
            )
            (target / "server.toml").write_text("[heartbeat]\nenabled = true\n", encoding="utf-8")
            (target / "tools.toml").write_text("[exec]\ntimeout = 60\n", encoding="utf-8")

        mock_sc.side_effect = _write_templates
        mock_lc.return_value = Config()

        yield config_dir, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_dir, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "rvoone is ready" in result.stdout
    assert config_dir.exists()
    assert (config_dir / "agent.toml").exists()
    assert (config_dir / "llm.toml").exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_dir, workspace_dir = mock_paths
    config_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "supported values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_refresh_preserves_existing_values_and_writes_missing_fields(tmp_path, monkeypatch):
    """Refresh should keep user values while materializing new default fields into TOML."""
    from rvoone.cli import commands

    root_dir = tmp_path / ".rvoone"
    config_dir = root_dir / "config"
    workspace_dir = tmp_path / "workspace"

    config_dir.mkdir(parents=True)
    (config_dir / "agent.toml").write_text(
        """
[defaults]
model = "custom-model"
workspace = """
        + f'"{workspace_dir}"'
        + """
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "tools.toml").write_text(
        """
[exec]
timeout = 99
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("rvoone.config.loader.get_config_path", lambda: config_dir)
    monkeypatch.setattr(commands, "get_workspace_path", lambda workspace: Path(workspace))
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda workspace: None)

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "supported values preserved" in result.stdout

    data = load_config(config_dir)
    agents_text = (config_dir / "agent.toml").read_text(encoding="utf-8")
    providers_text = (config_dir / "llm.toml").read_text(encoding="utf-8")

    assert data.agents.defaults.model == "custom-model"
    assert data.tools.exec.timeout == 99
    assert data.agents.defaults.provider == "auto"
    assert data.channels.send_progress is True
    assert data.gateway.heartbeat.enabled is True
    assert f'workspace = "{workspace_dir}"' in agents_text
    assert 'provider = "auto"' in agents_text
    assert "# [customSources.example]" in providers_text
    assert "[custom]" in providers_text
    assert data.get_provider_name() is None


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_dir, workspace_dir = mock_paths
    config_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_dir, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_config_prefers_custom_when_api_base_is_set_without_api_key():
    config = Config()
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://localhost:8000/v1"

    assert config.get_provider_name() == "custom"


def test_config_prefers_custom_source_prefix_when_present():
    config = Config()
    config.agents.defaults.model = "siliconflow/deepseek-ai/DeepSeek-V3"
    config.providers.custom_sources["siliconflow"] = config.providers.custom.model_copy(
        update={"api_base": "https://example.com/v1", "api_key": "key"}
    )

    assert config.get_provider_name() == "custom"
    provider = config.get_provider()
    assert provider is not None
    assert provider.api_base == "https://example.com/v1"
    assert (
        config.strip_model_provider_prefix(config.agents.defaults.model)
        == "deepseek-ai/DeepSeek-V3"
    )


def test_config_auto_selects_first_configured_custom_source_without_prefix():
    config = Config()
    config.agents.defaults.model = "qwen/qwen3-coder"
    config.providers.custom_sources["first"] = config.providers.custom.model_copy(
        update={"api_base": "https://first.example/v1", "api_key": "k1"}
    )
    config.providers.custom_sources["second"] = config.providers.custom.model_copy(
        update={"api_base": "https://second.example/v1", "api_key": "k2"}
    )

    assert config.get_provider_name() is None
    provider = config.get_provider()
    assert provider is None
    assert config.strip_model_provider_prefix(config.agents.defaults.model) == "qwen/qwen3-coder"


def test_config_prefers_default_custom_for_unprefixed_model_even_with_named_sources():
    config = Config()
    config.agents.defaults.model = "qwen/qwen3-coder"
    config.providers.custom.api_base = "https://default.example/v1"
    config.providers.custom_sources["first"] = config.providers.custom.model_copy(
        update={"api_base": "https://first.example/v1", "api_key": "k1"}
    )

    assert config.get_provider_name() == "custom"
    provider = config.get_provider()
    assert provider is not None
    assert provider.api_base == "https://default.example/v1"


def test_subagent_provider_override_uses_requested_provider():
    config = Config()
    config.agents.defaults.provider = "custom"

    assert config.get_provider_name("gpt-4o-mini", "custom") == "custom"


def test_install_writes_user_systemd_unit(tmp_path, monkeypatch):
    from rvoone.cli import commands

    unit_dir = tmp_path / "systemd" / "user"
    monkeypatch.setattr(commands, "get_user_systemd_dir", lambda: unit_dir)

    result = runner.invoke(app, ["install"])

    assert result.exit_code == 0
    unit_path = unit_dir / "rvoone.service"
    assert unit_path.exists()
    unit_text = unit_path.read_text(encoding="utf-8")
    assert f"ExecStart={sys.executable} -m rvoone gateway" in unit_text
    assert "systemctl --user daemon-reload" in result.stdout
    assert "systemctl --user enable rvoone.service" in result.stdout
    assert "systemctl --user start rvoone.service" in result.stdout


def test_install_keeps_existing_user_systemd_unit(tmp_path, monkeypatch):
    from rvoone.cli import commands

    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    unit_path = unit_dir / "rvoone.service"
    unit_path.write_text("existing-unit\n", encoding="utf-8")
    monkeypatch.setattr(commands, "get_user_systemd_dir", lambda: unit_dir)

    result = runner.invoke(app, ["install"])

    assert result.exit_code == 0
    assert unit_path.read_text(encoding="utf-8") == "existing-unit\n"
    assert "already exists" in result.stdout
    assert "No changes were made." in result.stdout


def test_status_shows_only_configured_custom_sources(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    workspace_dir = tmp_path / "workspace"
    config_dir.mkdir()
    workspace_dir.mkdir()
    config = Config()
    config.agents.defaults.workspace = str(workspace_dir)
    config.providers.custom.api_base = "https://default.example/v1"
    config.providers.custom_sources["ready"] = config.providers.custom.model_copy(
        update={"api_base": "https://ready.example/v1"}
    )
    config.providers.custom_sources["empty"] = ProviderConfig()

    monkeypatch.setattr("rvoone.config.loader.get_config_path", lambda: config_dir)
    monkeypatch.setattr("rvoone.config.loader.load_config", lambda: config)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Custom Sources: ready" in result.stdout
    assert "empty" not in result.stdout
