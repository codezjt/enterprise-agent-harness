from __future__ import annotations

import pytest
from langgraph.graph import StateGraph

from enterprise_harness.agent import AgentConfig, AgentRuntime
from enterprise_harness.orchestration.llm_replanner import LLMReplanner
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.replanner import ReplanResult
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime import RunManager, RunStatus, LangGraphRuntime


class FakeAgentRuntime(AgentRuntime):
    def __init__(self, config: AgentConfig, calls: list[str] | None = None):
        super().__init__(config)
        self.calls: list[str] = calls if calls is not None else []

    def build_agent(self, run_context=None):
        calls = self.calls

        async def execute(state):
            ctx = state.get("context", {})
            task_id = ctx.get("task_id", "")
            calls.append(task_id)

            if task_id == "T2":
                raise RuntimeError("permission denied")

            return {
                "messages": [
                    type("msg", (), {
                        "content": {
                            "task_id": task_id,
                            "answer": f"completed: {state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''}",
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


class InitialPlanner(Planner):
    async def plan(self, task: str) -> Plan:
        return Plan(
            task=task,
            tasks=[
                Task(
                    task_id="T1",
                    name="查询订单",
                    description="查询订单信息",
                ),
                Task(
                    task_id="T2",
                    name="查询库存",
                    description="查询库存信息",
                    dependencies=["T1"],
                    max_retries=0,
                ),
                Task(
                    task_id="T3",
                    name="更新订单",
                    description="根据库存更新订单",
                    dependencies=["T2"],
                ),
            ],
        )


class FakeLLMReplanner(LLMReplanner):

    def __init__(self):
        pass

    async def replan(self, request):
        return ReplanResult(
            reason="原库存接口权限不足，切换备用库存接口",
            tasks=[
                Task(
                    task_id="T1",
                    name="查询订单",
                    description="查询订单信息",
                    status="SUCCESS",
                ),
                Task(
                    task_id="T2_ALT",
                    name="调用备用库存接口",
                    description="通过备用库存接口查询库存",
                    dependencies=["T1"],
                ),
                Task(
                    task_id="T3",
                    name="更新订单",
                    description="根据备用库存结果更新订单",
                    dependencies=["T2_ALT"],
                ),
            ],
        )


@pytest.mark.asyncio
async def test_run_manager_llm_replanner_recovery():
    config = AgentConfig(
        agent_id="order-agent",
        name="Order Agent",
        model="test-model",
    )

    calls: list[str] = []
    agent_runtime = FakeAgentRuntime(config, calls=calls)

    replanner = FakeLLMReplanner()

    langgraph_rt = LangGraphRuntime(
        deepagent_runtime=agent_runtime,
        planner=InitialPlanner(),
        replanner=replanner,
        max_replans=1,
    )

    manager = RunManager(
        runtime=langgraph_rt,
    )

    run = await manager.create_run(
        agent_id="order-agent",
        task="查询订单并更新库存",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
    )

    assert result.status == RunStatus.COMPLETED

    assert "T1" in calls
    assert "T2" in calls
    assert "T2_ALT" in calls
    assert "T3" in calls

    assert calls.count("T2") == 1

    assert calls.count("T2_ALT") == 1

    assert calls.count("T3") == 1

    assert result.result["T2_ALT"] is not None
    assert result.result["T3"] is not None