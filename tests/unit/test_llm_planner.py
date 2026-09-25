import os

import pytest

from enterprise_harness.orchestration.llm_planner import LLMPlanner


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_planner():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip(
            "DASHSCOPE_API_KEY is not configured"
        )

    planner = LLMPlanner(
        model=os.getenv(
            "MODEL_NAME",
            "qwen3.6-flash",
        )
    )

    plan = await planner.plan(
        """
        处理订单 1001。
        先查询订单信息和库存信息，
        两者可以并行执行。
        然后根据订单和库存结果进行综合分析。
        """
    )

    assert plan.task

    assert len(plan.tasks) >= 2

    task_ids = {
        task.task_id
        for task in plan.tasks
    }

    assert len(task_ids) == len(plan.tasks)