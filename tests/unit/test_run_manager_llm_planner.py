import os

import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.llm_planner import LLMPlanner
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus


class FakeAgentRuntime(AgentRuntime):

    async def run(self, task: str, context=None):
        return {
            "task": task,
            "status": "success",
            "context": context,
        }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_manager_with_real_llm_planner():
    """
    验证：

    Real LLM
        ↓
    LLMPlanner
        ↓
    PlanValidator
        ↓
    TaskGraph
        ↓
    Scheduler
        ↓
    AgentRuntime
        ↓
    RunManager
    """

    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is not configured")

    config = AgentConfig(
        agent_id="planner-test-agent",
        name="Planner Test Agent",
        model=os.getenv("MODEL_NAME", "qwen3.6-flash"),
    )

    runtime = FakeAgentRuntime(config)

    run_manager = RunManager()

    run = await run_manager.create_run(
        agent_id=config.agent_id,
        task=(
            "处理订单 1001。"
            "先查询订单信息和库存信息，这两个任务可以并行执行。"
            "然后根据订单和库存结果进行综合分析。"
        ),
    )

    planner = LLMPlanner(
        model=config.model,
    )

    result = await run_manager.start_planned_run(
        run_id=run.run_id,
        runtime=runtime,
        planner=planner,
    )

    assert result.status == RunStatus.COMPLETED

    assert result.result is not None
    assert isinstance(result.result, dict)
    assert len(result.result) > 0