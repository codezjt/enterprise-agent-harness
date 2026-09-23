from enterprise_harness.agent import AgentConfig


def test_agent_config():
    config = AgentConfig(
        agent_id="europe-order-agent",
        name="European Order Agent",
        model="qwen3-max",
        system_prompt="你是一个订单分析 Agent",
        tools=[
            "query_order",
            "query_inventory",
        ],
        skills=[
            "order-analysis",
        ],
    )

    assert config.agent_id == "europe-order-agent"
    assert config.name == "European Order Agent"
    assert config.model == "qwen3-max"

    assert config.tools == [
        "query_order",
        "query_inventory",
    ]

    assert config.skills == [
        "order-analysis",
    ]