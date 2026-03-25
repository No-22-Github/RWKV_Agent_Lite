from rvoone.config.loader import get_config_path, load_config, save_config, save_config_templates
from rvoone.config.schema import Config


def test_get_config_path_points_to_config_directory():
    path = get_config_path()

    assert path.name == "config"
    assert path.suffix == ""


def test_load_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[agents.defaults]",
                'workspace = "~/custom-workspace"',
                'model = "gpt-4.1-mini"',
                "",
                "[tools]",
                "restrictToWorkspace = true",
                "",
                "[providers.custom]",
                'apiKey = "test-key"',
                'availableModels = ["gpt-4.1-mini", "gpt-4o-mini"]',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agents.defaults.workspace == "~/custom-workspace"
    assert config.agents.defaults.model == "gpt-4.1-mini"
    assert config.agents.subagent.model is None
    assert config.tools.restrict_to_workspace is True
    assert config.providers.custom.api_key == "test-key"
    assert config.providers.custom.available_models == ["gpt-4.1-mini", "gpt-4o-mini"]


def test_load_config_reads_subagent_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[agents.subagent]",
                'model = "gpt-4o-mini"',
                'provider = "custom"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agents.subagent.model == "gpt-4o-mini"
    assert config.agents.subagent.provider == "custom"


def test_load_config_reads_split_config_directory(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent.toml").write_text(
        "\n".join(
            [
                "[defaults]",
                'model = "gpt-4.1-mini"',
                "",
                "[subagent]",
                'provider = "custom"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "llm.toml").write_text(
        "\n".join(
            [
                "upstreamTimeout = 45",
                "",
                "[custom]",
                'apiBase = "http://localhost:8000/v1"',
                'tokenEstimation = "auto"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_dir)

    assert config.agents.defaults.model == "gpt-4.1-mini"
    assert config.agents.subagent.provider == "custom"
    assert config.providers.upstream_timeout == 45
    assert config.providers.custom.token_estimation == "auto"


def test_load_config_reads_named_custom_sources(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "llm.toml").write_text(
        "\n".join(
            [
                "[custom_sources.siliconflow]",
                'apiBase = "https://example.com/v1"',
                'apiKey = "key"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_dir)

    assert config.providers.custom_sources["siliconflow"].api_base == "https://example.com/v1"


def test_load_config_reports_failing_split_config_file(tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent.toml").write_text(
        "\n".join(
            [
                "[defaults]",
                'model = "gpt-4.1-mini"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "chat.toml").write_text(
        "\n".join(
            [
                "[telegram]",
                "enabled = enable",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_dir)

    output = capsys.readouterr().out
    assert "Failed to load config from" in output
    assert str(config_dir / "chat.toml") in output
    assert "line 2" in output
    assert "column 11" in output
    assert "Hint: check TOML syntax near this location." in output
    assert "Using default configuration." in output
    assert config.agents.defaults.model == "gpt-4o-mini"


def test_save_config_writes_split_config_directory_by_default(tmp_path):
    config_dir = tmp_path / "config"
    config = Config()
    config.providers.custom.api_key = "saved-key"
    config.providers.custom.token_estimation = "auto"
    config.providers.custom_sources["siliconflow"] = config.providers.custom.model_copy(
        update={"api_key": "source-key", "api_base": "https://example.com/v1"}
    )

    save_config(config, config_dir)

    agents_text = (config_dir / "agent.toml").read_text(encoding="utf-8")
    providers_text = (config_dir / "llm.toml").read_text(encoding="utf-8")
    channels_text = (config_dir / "chat.toml").read_text(encoding="utf-8")

    assert "[defaults]" in agents_text
    assert 'apiKey = "saved-key"' in providers_text
    assert 'tokenEstimation = "auto"' in providers_text
    assert "\n[customSources.siliconflow]\n" in providers_text
    assert 'apiBase = "https://example.com/v1"' in providers_text
    assert "[telegram]" in channels_text
    assert "pinToolStatus = false" in channels_text


def test_save_config_templates_writes_user_facing_provider_template(tmp_path):
    config_dir = tmp_path / "config"
    config = Config()

    save_config_templates(config, config_dir)

    providers_text = (config_dir / "llm.toml").read_text(encoding="utf-8")

    assert "# LLM settings." in providers_text
    assert "# [customSources.example]" in providers_text
    assert '# tokenEstimation = "auto"  # off | auto | on' in providers_text
    assert "# Single OpenAI-compatible endpoint." in providers_text
    assert "[custom]" in providers_text
    assert "apiBase = null" in providers_text
    assert 'tokenEstimation = "auto"' in providers_text
    assert "availableModels = []" in providers_text
    assert "[groq]" in providers_text
    assert 'apiKey = ""' in providers_text


def test_save_config_templates_keeps_empty_but_configurable_fields_visible(tmp_path):
    config_dir = tmp_path / "config"

    save_config_templates(Config(), config_dir)

    agents_text = (config_dir / "agent.toml").read_text(encoding="utf-8")
    channels_text = (config_dir / "chat.toml").read_text(encoding="utf-8")
    server_text = (config_dir / "server.toml").read_text(encoding="utf-8")
    tools_text = (config_dir / "tools.toml").read_text(encoding="utf-8")

    assert "reasoningEffort = null" in agents_text
    assert "model = null" in agents_text
    assert 'token = ""' in channels_text
    assert "allowFrom = []" in channels_text
    assert "pinToolStatus = false" in channels_text
    assert "proxy = null" in channels_text
    assert "heartbeat" in server_text
    assert 'pathAppend = ""' in tools_text


def test_save_config_templates_does_not_activate_example_custom_source(tmp_path):
    config_dir = tmp_path / "config"

    save_config_templates(Config(), config_dir)

    config = load_config(config_dir)

    assert config.providers.custom_sources == {}
    assert config.get_provider_name() is None


def test_save_config_templates_renders_configured_custom_sources(tmp_path):
    config_dir = tmp_path / "config"
    config = Config()
    config.providers.custom_sources["siliconflow"] = config.providers.custom.model_copy(
        update={
            "api_key": "key",
            "api_base": "https://example.com/v1",
            "token_estimation": "auto",
        }
    )

    save_config_templates(config, config_dir)

    providers_text = (config_dir / "llm.toml").read_text(encoding="utf-8")

    assert "[customSources.siliconflow]" in providers_text
    assert 'apiBase = "https://example.com/v1"' in providers_text
    assert 'tokenEstimation = "auto"' in providers_text


def test_save_config_writes_toml_with_tools_settings(tmp_path):
    config = Config()
    config.providers.custom.api_key = "saved-key"
    config.providers.custom.token_estimation = "auto"
    config.providers.custom.available_models = ["gpt-4o-mini", "gpt-4.1-mini"]
    config.tools.restrict_to_workspace = True
    config_path = tmp_path / "config.toml"

    save_config(config, config_path)
    text = config_path.read_text(encoding="utf-8")
    loaded = load_config(config_path)

    assert 'apiKey = "saved-key"' in text
    assert 'tokenEstimation = "auto"' in text
    assert 'availableModels = ["gpt-4o-mini", "gpt-4.1-mini"]' in text
    assert "restrictToWorkspace = true" in text
    assert loaded.providers.custom.api_key == "saved-key"
    assert loaded.providers.custom.token_estimation == "auto"
    assert loaded.providers.custom.available_models == ["gpt-4o-mini", "gpt-4.1-mini"]
    assert loaded.tools.restrict_to_workspace is True


def test_save_config_writes_null_literals_and_load_config_restores_none(tmp_path):
    config = Config()
    config_path = tmp_path / "config.toml"

    save_config(config, config_path)

    text = config_path.read_text(encoding="utf-8")
    loaded = load_config(config_path)

    assert "reasoningEffort = null" in text
    assert "model = null" in text
    assert "proxy = null" in text
    assert "apiBase = null" in text
    assert loaded.agents.defaults.reasoning_effort is None
    assert loaded.agents.subagent.model is None
    assert loaded.channels.telegram.proxy is None
    assert loaded.providers.custom.api_base is None


def test_save_config_omits_empty_parent_tables(tmp_path):
    config_path = tmp_path / "config.toml"

    save_config(Config(), config_path)

    text = config_path.read_text(encoding="utf-8")

    assert "\n[agents]\n" not in text
    assert "\n[tools.web]\n" not in text
    assert "[agents.defaults]" in text
    assert "[providers]" in text
    assert "[providers.custom]" in text
    assert "[tools.exec]" in text
