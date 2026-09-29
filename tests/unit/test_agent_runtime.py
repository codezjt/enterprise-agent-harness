from unittest.mock import patch

import pytest

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

    mock_create.assert_called_once_with(
        model="test-model",
        system_prompt="You are a test agent.",
        name="Test Agent",
    )


@pytest.mark.asyncio
async def test_deep_agent_runtime_run():
    config = create_config()

    class FakeAgent:
        async def ainvoke(self, input_data):
            return {
                "result": "test-result",
                "input": input_data,
            }

    fake_agent = FakeAgent()

    with patch(
        "enterprise_harness.agent.deepagent_runtime.create_deep_agent",
        return_value=fake_agent,
    ):
        runtime = DeepAgentRuntime(config)

        result = await runtime.run(
            task="test task",
        )

    assert result["result"] == "test-result"
    assert result["input"]["messages"] == [
        {
            "role": "user",
            "content": "test task",
        }
    ]