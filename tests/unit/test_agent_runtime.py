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
    assert runtime.agent is fake_agent

    # create_deep_agent 只应被调用一次
    mock_create.assert_called_once()

    kwargs = mock_create.call_args.kwargs

    # model 参数已被包装成 ChatOpenAI 实例
    assert isinstance(kwargs["model"], ChatOpenAI)
    assert kwargs["model"].model_name == "test-model"

    # 其余参数保持原样
    assert kwargs["system_prompt"] == "You are a test agent."
    assert kwargs["name"] == "Test Agent"
    assert kwargs["tools"] == []