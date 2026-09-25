import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus


class FakeAgentRuntime(AgentRuntime):
    def __init__(self, config):
        super().__init__(config)
        self.calls: dict[str, int] = {}

    async def run(self, task: str, context=None):
        task_id = context["task_id"]

        self.calls[task_id] = (
            self.calls.get(task_id, 0) + 1
        )

        if task_id == "T2" and self.calls[task_id] == 1:
            raise RuntimeError("temporary failure")

        return {
            "task": task_id,
            "status": "success",
        }


class RecoveryPlanner(Planner):
    async def plan(self, task: str) -> Plan:
        return Plan(
            task=task,
            tasks=[
                Task(
                    task_id="T1",
                    name="查询订单",
                    description="查询订单",
                ),
                Task(
                    task_id="T2",
                    name="查询库存",
                    description="查询库存",
                    dependencies=["T1"],
                ),
                Task(
                    task_id="T3",
                    name="更新订单",
                    description="更新订单",
                    dependencies=["T2"],
                ),
            ],
        )


@pytest.mark.asyncio
async def test_run_manager_can_recover_failed_task():
    config = AgentConfig(
        agent_id="recovery-agent",
        name="Recovery Agent",
        model="test-model",
    )

    runtime = FakeAgentRuntime(config)
    planner = RecoveryPlanner()

    run_manager = RunManager()

    run = await run_manager.create_run(
        agent_id=config.agent_id,
        task="查询订单、库存并更新订单",
    )

    result = await run_manager.start_planned_run(
        run_id=run.run_id,
        runtime=runtime,
        planner=planner,
    )

    assert result.status == RunStatus.COMPLETED

    assert runtime.calls["T1"] == 1
    assert runtime.calls["T2"] == 2
    assert runtime.calls["T3"] == 1

    assert result.result["T1"]["status"] == "success"
    assert result.result["T2"]["status"] == "success"
    assert result.result["T3"]["status"] == "success"