import pytest

from enterprise_harness.orchestration.executor import (
    AgentTaskExecutor,
)
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.replanner import SimpleReplanner
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import Task, TaskStatus
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.context import RunContext


class FakeLangGraphRuntime:
    def __init__(self):
        self.runs: list[dict] = []

    async def run(self, run_id, input_data):
        task_content = input_data["messages"][0]["content"]
        self.runs.append(
            {
                "run_id": run_id,
                "input_data": input_data,
            }
        )
        return {
            "messages": [{"role": "assistant", "content": f"Done: {task_content}"}],
        }


class FakeFailingLangGraphRuntime:
    async def run(self, run_id, input_data):
        raise ValueError("simulated tool failure")


class FakePlanner(Planner):
    def __init__(self, tasks: list[Task]):
        self._tasks = tasks

    async def plan(self, task: str) -> Plan:
        return Plan(task=task, tasks=self._tasks)


class TestAgentTaskExecutorLangGraph:

    def make_context(self):
        return RunContext(
            run_id="run-orch-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="orchestrate multi-step task",
        )

    @pytest.mark.asyncio
    async def test_executor_runs_through_langgraph(self):
        ctx = self.make_context()
        lg_runtime = FakeLangGraphRuntime()

        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        task = Task(
            task_id="step-1",
            name="Analyze",
            description="Analyze the data",
        )

        result = await executor.execute(task)
        assert len(lg_runtime.runs) == 1
        assert "Done: Analyze the data" in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_executor_includes_task_context(self):
        ctx = self.make_context()
        ctx.context["system_policy"] = "Be accurate"
        lg_runtime = FakeLangGraphRuntime()

        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        task = Task(
            task_id="step-1",
            name="Verify",
            description="Verify the result",
            input={"threshold": 0.9},
        )

        await executor.execute(task)

        run = lg_runtime.runs[0]
        assert run["run_id"] == "run-orch-001"
        assert run["input_data"]["context"]["task_id"] == "step-1"
        assert run["input_data"]["context"]["task_name"] == "Verify"
        assert run["input_data"]["context"]["task_input"]["threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_executor_passes_dependency_results(self):
        ctx = self.make_context()
        lg_runtime = FakeLangGraphRuntime()

        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        task = Task(
            task_id="step-3",
            name="Combine",
            description="Combine results",
            input={
                "dependency_results": {
                    "step-1": {"result": "A"},
                    "step-2": {"result": "B"},
                }
            },
        )

        await executor.execute(task)

        run = lg_runtime.runs[0]
        ctx_data = run["input_data"]["context"]
        assert ctx_data["dependency_results"]["step-1"]["result"] == "A"
        assert ctx_data["dependency_results"]["step-2"]["result"] == "B"

    @pytest.mark.asyncio
    async def test_executor_passes_principal(self):
        from enterprise_harness.policy.rbac import Principal

        ctx = RunContext(
            run_id="run-p-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="test",
            principal=Principal(principal_id="user-1", role="admin"),
        )
        lg_runtime = FakeLangGraphRuntime()

        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        task = Task(task_id="t1", name="Test")

        await executor.execute(task)

        run = lg_runtime.runs[0]
        assert run["input_data"]["context"]["_principal"] == ctx.principal

    @pytest.mark.asyncio
    async def test_executor_with_built_context(self):
        from enterprise_harness.context.models import ContextItem

        ctx = self.make_context()
        ctx.built_context = [
            ContextItem(priority=0, source="system", content="System prompt"),
            ContextItem(priority=1, source="task", content="Task description"),
        ]
        lg_runtime = FakeLangGraphRuntime()

        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        task = Task(task_id="t1", name="Test")

        await executor.execute(task)

        run = lg_runtime.runs[0]
        ctx_data = run["input_data"]["context"]
        assert "items" in ctx_data
        assert len(ctx_data["items"]) == 2
        assert ctx_data["items"][0]["source"] == "system"


class TestTaskGraphSchedulerIntegration:

    @pytest.mark.asyncio
    async def test_scheduler_executes_linear_tasks(self):
        lg_runtime = FakeLangGraphRuntime()
        ctx = RunContext(
            run_id="run-sched-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="test",
        )
        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        graph = TaskGraph()
        t1 = Task(task_id="t1", name="Step 1")
        t2 = Task(task_id="t2", name="Step 2", dependencies=["t1"])
        t3 = Task(task_id="t3", name="Step 3", dependencies=["t2"])

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)
        graph.add_dependency("t2", "t1")
        graph.add_dependency("t3", "t2")

        scheduler = Scheduler(graph=graph, executor=executor)
        result_graph = await scheduler.run()

        assert result_graph.is_completed()
        assert len(lg_runtime.runs) == 3

    @pytest.mark.asyncio
    async def test_scheduler_handles_failure(self):
        ctx = RunContext(
            run_id="run-fail-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="test",
        )
        executor = AgentTaskExecutor(
            langgraph_runtime=FakeFailingLangGraphRuntime(),
            run_context=ctx,
        )

        graph = TaskGraph()
        t1 = Task(task_id="t1", name="Bad Step")
        graph.add_task(t1)

        scheduler = Scheduler(graph=graph, executor=executor)
        result_graph = await scheduler.run()

        assert result_graph.has_failed()
        assert result_graph.get_task("t1").status == TaskStatus.FAILED


class TestPlannerPlanValidatorRecovery:

    @pytest.mark.asyncio
    async def test_planner_validator_scheduler_integration(self):
        lg_runtime = FakeLangGraphRuntime()
        ctx = RunContext(
            run_id="run-plan-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="Complete project",
        )
        executor = AgentTaskExecutor(
            langgraph_runtime=lg_runtime,
            run_context=ctx,
        )

        planner = FakePlanner(
            tasks=[
                Task(task_id="design", name="Design", description="Design solution"),
                Task(
                    task_id="implement",
                    name="Implement",
                    description="Write code",
                    dependencies=["design"],
                ),
                Task(
                    task_id="test",
                    name="Test",
                    description="Run tests",
                    dependencies=["implement"],
                ),
            ]
        )

        plan = await planner.plan("Complete project")
        assert len(plan.tasks) == 3

        graph = plan.to_task_graph()
        assert len(graph.tasks()) == 3

        scheduler = Scheduler(graph=graph, executor=executor)
        result_graph = await scheduler.run()

        assert result_graph.is_completed()
        assert len(lg_runtime.runs) == 3

    def test_simple_replanner_maintains_completed_tasks(self):
        from enterprise_harness.orchestration.replanner import ReplanRequest

        replanner = SimpleReplanner()

        completed = [
            Task(
                task_id="t1",
                name="Step 1",
                status=TaskStatus.SUCCESS,
                output="done",
            )
        ]
        failed = Task(
            task_id="t2",
            name="Step 2",
            status=TaskStatus.FAILED,
            error="timeout",
        )
        remaining = [
            Task(
                task_id="t3",
                name="Step 3",
                status=TaskStatus.PENDING,
            )
        ]

        request = ReplanRequest(
            original_task="test",
            failed_task=failed,
            completed_tasks=completed,
            remaining_tasks=remaining,
        )

        assert request.original_task == "test"
        assert request.failed_task.task_id == "t2"
        assert isinstance(replanner, SimpleReplanner)