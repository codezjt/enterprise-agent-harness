from unittest.mock import ANY, patch

import pytest
from langchain_openai import ChatOpenAI

from enterprise_harness.agent import AgentConfig, DeepAgentRuntime


def create_config() -> AgentConfig:
    return AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
        system_prompt="You are a test agent.",
    )


def test_deep_agent_runtime_config():
    config = create_config()

    fake_agent = object()

    with patch(
        "enterprise_harness.agent.deepagent_runtime.create_deep_agent",
        return_value=fake_agent,
    ) as mock_create:

        runtime = DeepAgentRuntime(config)

        assert runtime.config is config
        assert callable(runtime.build_agent)

        built = runtime.build_agent(run_context=None)

    assert built is fake_agent

    mock_create.assert_called_once()

    kwargs = mock_create.call_args.kwargs

    assert isinstance(kwargs["model"], ChatOpenAI)
    assert kwargs["model"].model_name == "test-model"

    assert kwargs["system_prompt"] == "You are a test agent."
    assert kwargs["name"] == "Test Agent"
    assert kwargs["tools"] == []