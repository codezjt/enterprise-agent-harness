import pytest
from langgraph.graph import StateGraph

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime


class FakeAgentRuntime(AgentRuntime):
    def __init__(self, config, calls: dict[str, int] | None = None):
        super().__init__(config)
        self.calls: dict[str, int] = calls if calls is not None else {}

    def build_agent(self, run_context=None):
        calls = self.calls

        async def execute(state):
            ctx = state.get("context", {})
            task_id = ctx.get("task_id", "")

            calls[task_id] = calls.get(task_id, 0) + 1

            if task_id == "T2" and calls[task_id] == 1:
                raise RuntimeError("temporary failure")

            return {
                "messages": [
                    type("msg", (), {
                        "content": {
                            "task": task_id,
                            "status": "success",
                        }
                    })()
                ],
                "context": ctx,
            }

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


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

    calls: dict[str, int] = {}
    agent_runtime = FakeAgentRuntime(config, calls=calls)
    planner = RecoveryPlanner()

    langgraph_rt = LangGraphRuntime(
        deepagent_runtime=agent_runtime,
        planner=planner,
    )

    run_manager = RunManager(
        runtime=langgraph_rt,
    )

    run = await run_manager.create_run(
        agent_id=config.agent_id,
        task="查询订单、库存并更新订单",
    )

    result = await run_manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
    )

    assert result.status == RunStatus.COMPLETED

    assert calls["T1"] == 1
    assert calls["T2"] == 2
    assert calls["T3"] == 1

    assert result.result["T1"]["status"] == "success"
    assert result.result["T2"]["status"] == "success"
    assert result.result["T3"]["status"] == "success"