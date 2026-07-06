import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/6_2_plan.py")


class FakeResponsesAPI:
    def __init__(self):
        self.create_calls = []
        self.response = None
        self.side_effects = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.side_effects:
            next_effect = self.side_effects.pop(0)
            if isinstance(next_effect, Exception):
                raise next_effect
            return next_effect
        return self.response


class FakeOpenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.responses = FakeResponsesAPI()


def load_module_with_fake_openai():
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAIClient
    sys.modules["openai"] = fake_openai

    spec = importlib.util.spec_from_file_location("deep_research_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deep_research_example = load_module_with_fake_openai()


def build_fake_response():
    report = "Semaglutide lowers complication costs and can shift spending toward prevention."
    citation = types.SimpleNamespace(
        title="WHO report",
        url="https://www.who.int/example",
        start_index=0,
        end_index=11,
    )
    message_item = types.SimpleNamespace(
        type="message",
        content=[
            types.SimpleNamespace(
                text=report,
                annotations=[citation],
            )
        ],
    )
    reasoning_item = types.SimpleNamespace(
        type="reasoning",
        summary=[
            types.SimpleNamespace(
                text="The model first scoped the topic, then compared healthcare cost categories."
            )
        ],
    )
    search_item = types.SimpleNamespace(
        type="web_search_call",
        action={
            "query": "semaglutide healthcare system economic impact",
            "sources": [
                {
                    "title": "WHO report",
                    "url": "https://www.who.int/example",
                },
                {
                    "title": "OECD brief",
                    "url": "https://www.oecd.org/example",
                },
            ],
        },
    )
    return types.SimpleNamespace(
        output_text=report,
        output=[reasoning_item, search_item, message_item],
    )


class FakeRateLimitError(Exception):
    def __init__(self, message, *, status_code=429, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def test_default_configuration_uses_small_model_and_low_effort():
    config = deep_research_example.DeepResearchConfig()

    assert config.model == "gpt-5.4-mini"
    assert config.reasoning_effort == "low"
    assert config.search_context_size == "medium"


def test_deep_research_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="query must not be empty"):
        deep_research_example.DeepResearchConfig(query="   ")

    with pytest.raises(ValueError, match="system_prompt must not be empty"):
        deep_research_example.DeepResearchConfig(system_prompt="   ")

    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        deep_research_example.DeepResearchConfig(reasoning_effort="extreme")

    with pytest.raises(ValueError, match="search_context_size must be one of"):
        deep_research_example.DeepResearchConfig(search_context_size="tiny")


def test_validate_runtime_environment_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        deep_research_example.validate_runtime_environment()


def test_validate_runtime_environment_allows_non_openai_models(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    deep_research_example.validate_runtime_environment(model="gemini-2.5-flash")


def test_build_research_input_uses_developer_and_user_roles():
    messages = deep_research_example.build_research_input(
        config=deep_research_example.DeepResearchConfig(
            query="Research topic",
            system_prompt="You are a researcher.",
        )
    )

    assert messages == [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": "You are a researcher."}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Research topic"}],
        },
    ]


def test_build_web_search_tool_applies_domain_filters():
    config = deep_research_example.DeepResearchConfig(
        allowed_domains=("www.who.int", "www.oecd.org"),
        blocked_domains=("reddit.com",),
    )

    tool = deep_research_example.build_web_search_tool(config)

    assert tool == {
        "type": "web_search",
        "search_context_size": "medium",
        "filters": {
            "allowed_domains": ["www.who.int", "www.oecd.org"],
            "blocked_domains": ["reddit.com"],
        },
    }


def test_build_web_search_tool_uses_low_context_in_fast_test_mode():
    config = deep_research_example.DeepResearchConfig(
        fast_test_mode=True,
        search_context_size="high",
    )

    tool = deep_research_example.build_web_search_tool(config)

    assert tool["search_context_size"] == "low"


def test_build_response_request_matches_openai_responses_pattern():
    config = deep_research_example.DeepResearchConfig(
        query="Research semaglutide",
        system_prompt="Use reliable sources.",
        model="gpt-5.5",
        reasoning_effort="medium",
        allowed_domains=("www.who.int",),
    )

    request = deep_research_example.build_response_request(config)

    assert request["model"] == "gpt-5.5"
    assert request["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert request["tool_choice"] == "auto"
    assert request["include"] == ["web_search_call.action.sources"]
    assert request["tools"][0]["type"] == "web_search"
    assert request["tools"][0]["search_context_size"] == "medium"
    assert request["tools"][0]["filters"]["allowed_domains"] == ["www.who.int"]


def test_build_response_request_in_fast_test_mode_constrains_query_and_search_context():
    config = deep_research_example.DeepResearchConfig(
        query="Research semaglutide",
        fast_test_mode=True,
    )

    request = deep_research_example.build_response_request(config)

    assert request["tools"][0]["search_context_size"] == "low"
    user_text = request["input"][1]["content"][0]["text"]
    assert "Fast test mode" in user_text
    assert "use only a few high-quality sources" in user_text


def test_parse_deep_research_response_extracts_report_reasoning_and_sources():
    response = build_fake_response()

    result = deep_research_example.parse_deep_research_response(response)

    assert "Semaglutide lowers complication costs" in result.report
    assert result.reasoning_summary == (
        "The model first scoped the topic, then compared healthcare cost categories.",
    )
    assert result.search_queries == (
        "semaglutide healthcare system economic impact",
    )
    assert result.citations[0].title == "WHO report"
    assert result.citations[0].cited_text == "Semaglutide"
    assert result.sources == (
        deep_research_example.SourceReference(
            title="WHO report",
            url="https://www.who.int/example",
        ),
        deep_research_example.SourceReference(
            title="OECD brief",
            url="https://www.oecd.org/example",
        ),
    )


def test_create_openai_client_uses_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    client = deep_research_example.create_openai_client()

    assert client.api_key == "env-key"


def test_build_runtime_config_from_environment_reads_fast_mode_and_retry_settings(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_DEEP_RESEARCH_FAST_TEST_MODE", "true")
    monkeypatch.setenv("OPENAI_DEEP_RESEARCH_MAX_RETRIES", "5")
    monkeypatch.setenv("OPENAI_DEEP_RESEARCH_RETRY_BASE_DELAY_SECONDS", "0.25")

    config = deep_research_example.build_runtime_config_from_environment(
        query="Research semaglutide",
    )

    assert config.fast_test_mode is True
    assert config.max_retries == 5
    assert config.retry_base_delay_seconds == 0.25


def test_build_runtime_config_from_environment_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("OPENAI_DEEP_RESEARCH_FAST_TEST_MODE", "sometimes")

    with pytest.raises(ValueError, match="OPENAI_DEEP_RESEARCH_FAST_TEST_MODE must be one of"):
        deep_research_example.build_runtime_config_from_environment()


def test_run_deep_research_retries_rate_limit_errors(monkeypatch):
    client = FakeOpenAIClient(api_key="test-key")
    client.responses.side_effects = [
        FakeRateLimitError("Please try again in 1.25s.", retry_after=1.25),
        build_fake_response(),
    ]
    config = deep_research_example.DeepResearchConfig(
        max_retries=2,
        retry_base_delay_seconds=0.01,
    )
    sleep_calls = []

    response = deep_research_example.run_deep_research(
        client=client,
        config=config,
        sleep_func=sleep_calls.append,
    )

    assert response.output_text.startswith("Semaglutide lowers")
    assert sleep_calls == [1.25]
    assert len(client.responses.create_calls) == 2


def test_run_deep_research_raises_after_retry_budget_is_exhausted():
    client = FakeOpenAIClient(api_key="test-key")
    client.responses.side_effects = [
        FakeRateLimitError("Please try again in 0.5s."),
        FakeRateLimitError("Please try again in 0.5s."),
    ]
    config = deep_research_example.DeepResearchConfig(
        max_retries=1,
        retry_base_delay_seconds=0.01,
    )

    with pytest.raises(FakeRateLimitError):
        deep_research_example.run_deep_research(
            client=client,
            config=config,
            sleep_func=lambda _: None,
        )


def test_openai_deep_research_app_runs_request_and_prints_reasoning_first(capsys):
    client = FakeOpenAIClient(api_key="test-key")
    client.responses.response = build_fake_response()
    config = deep_research_example.DeepResearchConfig(
        query="Research semaglutide",
        allowed_domains=("www.who.int",),
    )
    app = deep_research_example.OpenAIDeepResearchApp.build(
        client=client,
        config=config,
    )

    result = app.print_execution()
    output = capsys.readouterr().out

    assert "Semaglutide lowers complication costs" in result.report
    assert client.responses.create_calls == [
        deep_research_example.build_response_request(config)
    ]
    assert "## Research reasoning" in output
    assert "## Research outcome" in output
    assert output.index("## Research reasoning") < output.index("## Research outcome")
    assert "Search queries:" in output
    assert "## Sources" in output


def test_require_openai_package_raises_helpful_error_when_missing(monkeypatch):
    monkeypatch.setattr(deep_research_example, "OpenAI", None)

    with pytest.raises(ImportError, match="openai is not installed"):
        deep_research_example.require_openai_package()
