import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.evaluation import (
    EvaluationDataset,
    EvaluationInput,
    EvaluationResult,
    EvaluationRunner,
    RecordedToolCall,
    ToolCallRecorder,
    ToolEvaluation,
)
from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.observability import (
    AuditLogger,
    MetricCollector,
    TraceManager,
)
from enterprise_harness.observability.trace import SpanStatus, SpanType
from enterprise_harness.policy import PolicyEngine
from enterprise_harness.runtime.manager import RunManager
from tests.helpers import FakeLangGraphRuntimeForAgent


class _ToolCallAgent(AgentRuntime):
    def __init__(self, config, gateway=None, trace_manager=None):
        super().__init__(config)
        self.gateway = gateway
        self.trace_manager = trace_manager

    async def run(self, task, context=None):
        if self.gateway is None:
            return {"answer": f"no gateway for: {task}"}

        tools_to_call = context.get("_tools", []) if context else []
        results = []
        run_id = context.get("run_id", "eval-default") if context else "eval-default"

        if self.trace_manager:
            self.trace_manager.start_run_span(
                run_id=run_id,
                agent_name=self.config.agent_id,
                input=task,
            )

        for tool_name in tools_to_call:
            try:
                result = await self.gateway.execute(
                    tool_name=tool_name,
                    arguments=context.get("tool_args", {}).get(tool_name, {}) if context else {},
                    run_id=run_id,
                )
                results.append({"tool": tool_name, "result": result, "success": True})
            except Exception as exc:
                results.append({"tool": tool_name, "error": str(exc), "success": False})

        return {"answer": f"done", "tools": results}


class TestToolCallRecorder:

    def test_recorder_extracts_tool_names_from_traces(self):
        trace_mgr = TraceManager()

        trace_mgr.start_span(
            run_id="run-1", span_type=SpanType.TOOL, name="query_order",
        )
        trace_mgr.start_span(
            run_id="run-1", span_type=SpanType.TOOL, name="update_order",
        )
        trace_mgr.start_span(
            run_id="run-1", span_type=SpanType.AGENT, name="agent",
        )

        recorder = ToolCallRecorder(trace_mgr)
        names = recorder.get_tool_names("run-1")

        assert sorted(names) == ["query_order", "update_order"]

    def test_recorder_returns_empty_for_no_tools(self):
        trace_mgr = TraceManager()
        trace_mgr.start_span(
            run_id="run-1", span_type=SpanType.AGENT, name="agent",
        )

        recorder = ToolCallRecorder(trace_mgr)
        names = recorder.get_tool_names("run-1")

        assert names == []

    def test_recorder_clear(self):
        trace_mgr = TraceManager()
        trace_mgr.start_span(
            run_id="run-1", span_type=SpanType.TOOL, name="query_order",
        )

        recorder = ToolCallRecorder(trace_mgr)
        assert len(recorder.get_tool_names("run-1")) == 1

        recorder.clear("run-1")
        assert recorder.get_recorded_calls("run-1") == []

    def test_recorded_tool_call_name_alias(self):
        call = RecordedToolCall(tool_name="query_order", success=True)
        assert call.name == "query_order"


class TestEvaluationResult:

    def test_success_rate(self):
        result = EvaluationResult(
            run_id="r1", dataset_name="d1", dataset_version="1.0",
            total_cases=10, success_cases=7, failed_cases=3,
        )
        assert result.success_rate == 0.7

    def test_success_rate_zero_cases(self):
        result = EvaluationResult(run_id="r1", dataset_name="d1", dataset_version="1.0")
        assert result.success_rate == 0.0

    def test_tool_precision(self):
        result = EvaluationResult(run_id="r1", dataset_name="d1", dataset_version="1.0")
        result.tool_evaluations = [
            ToolEvaluation(case_id="c1", expected=["a", "b"], actual=["a"], tp=1, fp=0, fn=1),
            ToolEvaluation(case_id="c2", expected=["a"], actual=["a", "b"], tp=1, fp=1, fn=0),
        ]
        assert result.average_tool_precision == 0.75

    def test_tool_recall(self):
        result = EvaluationResult(run_id="r1", dataset_name="d1", dataset_version="1.0")
        result.tool_evaluations = [
            ToolEvaluation(case_id="c1", expected=["a", "b"], actual=["a"], tp=1, fp=0, fn=1),
            ToolEvaluation(case_id="c2", expected=["a"], actual=["a"], tp=1, fp=0, fn=0),
        ]
        assert result.average_tool_recall == 0.75

    def test_token_usage_property(self):
        result = EvaluationResult(
            run_id="r1", dataset_name="d1", dataset_version="1.0",
            input_tokens=100, output_tokens=50,
        )
        assert result.token_usage == {"input": 100, "output": 50, "total": 150}

    def test_summary_includes_all_fields(self):
        result = EvaluationResult(
            run_id="r1", dataset_name="d1", dataset_version="1.0",
            total_cases=5, success_cases=4, failed_cases=1,
            estimated_cost=0.05, planning_score=0.8,
            replan_count=2, retry_count=1,
            policy_denials=1, approval_required=1, approval_approved=1,
            input_tokens=200, output_tokens=100,
            latencies_ms=[10.0, 20.0, 30.0],
        )
        summary = result.summary()

        assert summary["total_cases"] == 5
        assert summary["success_rate"] == 0.8
        assert summary["estimated_cost"] == 0.05
        assert summary["planning_score"] == 0.8
        assert summary["replan_count"] == 2
        assert summary["retry_count"] == 1
        assert summary["policy_denials"] == 1
        assert summary["approval_required"] == 1
        assert summary["approval_approved"] == 1
        assert summary["token_usage"]["input"] == 200
        assert summary["token_usage"]["output"] == 100
        assert summary["average_latency_ms"] == 20.0


class TestToolEvaluation:

    def test_precision_zero_when_no_predictions(self):
        te = ToolEvaluation(case_id="c1", expected=["a"], actual=[], tp=0, fp=0, fn=1)
        assert te.precision == 0.0

    def test_precision_perfect(self):
        te = ToolEvaluation(case_id="c1", expected=["a", "b"], actual=["a", "b"], tp=2, fp=0, fn=0)
        assert te.precision == 1.0

    def test_recall_zero_when_no_expected(self):
        te = ToolEvaluation(case_id="c1", expected=[], actual=["a"], tp=0, fp=1, fn=0)
        assert te.recall == 0.0

    def test_recall_perfect(self):
        te = ToolEvaluation(case_id="c1", expected=["a", "b"], actual=["a", "b"], tp=2, fp=0, fn=0)
        assert te.recall == 1.0


class TestEvaluationRunnerToolCapture:

    @pytest.mark.asyncio
    async def test_runner_via_runtime_captures_actual_tools(self):
        async def query_order(order_id: str):
            return {"order_id": order_id, "status": "PAID"}

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="query_order", description="Query order",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=query_order,
        ))

        trace_mgr = TraceManager()
        gateway = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
            trace_manager=trace_mgr,
        )

        agent = _ToolCallAgent(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
            gateway=gateway,
            trace_manager=trace_mgr,
        )

        runner = EvaluationRunner(
            runtime=agent,
            trace_manager=trace_mgr,
        )

        dataset = EvaluationDataset(
            name="test-dataset",
            cases=[
                EvaluationInput(
                    id="c1",
                    input="query order 1001",
                    expected_tools=["query_order"],
                    metadata={"_tools": ["query_order"], "tool_args": {"query_order": {"order_id": "1001"}}},
                ),
            ],
        )

        result = await runner.run(dataset)

        assert result.success_cases == 1
        assert result.total_cases == 1
        assert len(result.tool_evaluations) == 1

        te = result.tool_evaluations[0]
        assert "query_order" in te.actual
        assert te.precision == 1.0
        assert te.recall == 1.0

    @pytest.mark.asyncio
    async def test_runner_via_runtime_no_tools(self):
        agent = _ToolCallAgent(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
        )

        trace_mgr = TraceManager()
        runner = EvaluationRunner(runtime=agent, trace_manager=trace_mgr)

        dataset = EvaluationDataset(
            name="test-dataset",
            cases=[
                EvaluationInput(
                    id="c2",
                    input="hello",
                    expected_tools=[],
                ),
            ],
        )

        result = await runner.run(dataset)

        assert result.success_cases == 1
        te = result.tool_evaluations[0]
        assert te.actual == []

    @pytest.mark.asyncio
    async def test_runner_tool_precision(self):
        async def order_handler(order_id: str):
            return {"order_id": order_id, "ok": True}

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="query_order", description="Query order",
            input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            handler=order_handler,
        ))

        trace_mgr = TraceManager()
        gateway = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
            trace_manager=trace_mgr,
        )

        agent = _ToolCallAgent(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
            gateway=gateway,
            trace_manager=trace_mgr,
        )

        runner = EvaluationRunner(runtime=agent, trace_manager=trace_mgr)

        dataset = EvaluationDataset(
            name="precision-test",
            cases=[
                EvaluationInput(
                    id="c1",
                    input="check order",
                    expected_tools=["query_order"],
                    metadata={"_tools": ["query_order"], "tool_args": {"query_order": {"order_id": "1"}}},
                ),
            ],
        )

        result = await runner.run(dataset)
        assert result.average_tool_precision == 1.0

    @pytest.mark.asyncio
    async def test_runner_tool_recall(self):
        async def order_handler(order_id: str):
            return {"order_id": order_id, "ok": True}

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="query_order", description="Query order",
            input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            handler=order_handler,
        ))

        trace_mgr = TraceManager()
        gateway = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
            trace_manager=trace_mgr,
        )

        agent = _ToolCallAgent(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
            gateway=gateway,
            trace_manager=trace_mgr,
        )

        runner = EvaluationRunner(runtime=agent, trace_manager=trace_mgr)

        dataset = EvaluationDataset(
            name="recall-test",
            cases=[
                EvaluationInput(
                    id="c1",
                    input="check order",
                    expected_tools=["query_order"],
                    metadata={"_tools": ["query_order"], "tool_args": {"query_order": {"order_id": "1"}}},
                ),
            ],
        )

        result = await runner.run(dataset)
        assert result.average_tool_recall == 1.0

    @pytest.mark.asyncio
    async def test_runner_tool_partial_match(self):
        async def query_order(order_id: str):
            return {"order_id": order_id}

        async def update_order(order_id: str, status: str):
            return {"order_id": order_id, "status": status}

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="query_order", description="Query order",
            input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            handler=query_order,
        ))
        registry.register(ToolDefinition(
            name="update_order", description="Update order",
            input_schema={"type": "object", "properties": {"order_id": {"type": "string"}, "status": {"type": "string"}}, "required": ["order_id", "status"]},
            handler=update_order,
        ))

        trace_mgr = TraceManager()
        gateway = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
            trace_manager=trace_mgr,
        )

        agent = _ToolCallAgent(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
            gateway=gateway,
            trace_manager=trace_mgr,
        )

        runner = EvaluationRunner(runtime=agent, trace_manager=trace_mgr)

        dataset = EvaluationDataset(
            name="partial-test",
            cases=[
                EvaluationInput(
                    id="c1",
                    input="check and update order",
                    expected_tools=["query_order", "update_order", "verify_order"],
                    metadata={
                        "_tools": ["query_order", "update_order"],
                        "tool_args": {
                            "query_order": {"order_id": "1"},
                            "update_order": {"order_id": "1", "status": "done"},
                        },
                    },
                ),
            ],
        )

        result = await runner.run(dataset)

        te = result.tool_evaluations[0]
        assert te.tp == 2
        assert te.fn == 1
        assert te.fp == 0
        expected_precision = 1.0
        expected_recall = 2 / 3
        assert te.precision == expected_precision
        assert abs(te.recall - expected_recall) < 0.001


class TestEvaluationRunnerViaRunManager:

    @pytest.mark.asyncio
    async def test_runner_with_run_manager(self):
        class SimpleRuntime(AgentRuntime):
            async def run(self, task, context=None):
                return {"answer": f"completed: {task}"}

        trace_mgr = TraceManager()
        metric_collector = MetricCollector()
        audit_logger = AuditLogger()

        agent = SimpleRuntime(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
        )

        run_mgr = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        runner = EvaluationRunner(
            runtime=agent,
            run_manager=run_mgr,
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
            audit_logger=audit_logger,
        )

        dataset = EvaluationDataset(
            name="run-manager-test",
            cases=[
                EvaluationInput(
                    id="c1",
                    input="hello",
                    expected_tools=[],
                ),
            ],
        )

        result = await runner.run(dataset)

        assert result.success_cases == 1
        assert result.total_cases == 1
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_runner_run_manager_produces_metrics(self):
        class SimpleRuntime(AgentRuntime):
            async def run(self, task, context=None):
                return {"answer": f"completed: {task}"}

        trace_mgr = TraceManager()
        metric_collector = MetricCollector()
        audit_logger = AuditLogger()

        agent = SimpleRuntime(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
        )

        run_mgr = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        runner = EvaluationRunner(
            runtime=agent,
            run_manager=run_mgr,
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
            audit_logger=audit_logger,
        )

        dataset = EvaluationDataset(
            name="metrics-test",
            cases=[
                EvaluationInput(id="c1", input="task 1", expected_tools=[]),
                EvaluationInput(id="c2", input="task 2", expected_tools=[]),
            ],
        )

        result = await runner.run(dataset)

        assert result.success_cases == 2
        metrics_snapshot = metric_collector.snapshot()
        assert metrics_snapshot.agent_runs >= 2
        assert result.latencies_ms is not None
        assert len(result.latencies_ms) == 2

    @pytest.mark.asyncio
    async def test_runner_collects_latency(self):
        class SimpleRuntime(AgentRuntime):
            async def run(self, task, context=None):
                return {"answer": f"completed: {task}"}

        trace_mgr = TraceManager()
        metric_collector = MetricCollector()
        audit_logger = AuditLogger()

        agent = SimpleRuntime(
            config=AgentConfig(agent_id="eval-agent", name="Eval Agent", model="test"),
        )

        run_mgr = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        runner = EvaluationRunner(
            runtime=agent,
            run_manager=run_mgr,
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
            audit_logger=audit_logger,
        )

        dataset = EvaluationDataset(
            name="latency-test",
            cases=[
                EvaluationInput(id="c1", input="task", expected_tools=[]),
            ],
        )

        result = await runner.run(dataset)

        assert len(result.latencies_ms) == 1
        assert result.latencies_ms[0] > 0


class TestEvaluationDataset:

    def test_dataset_rejects_duplicate_case_ids(self):
        with pytest.raises(ValueError, match="Duplicate case id"):
            EvaluationDataset(
                name="d1",
                cases=[
                    EvaluationInput(id="c1", input="a"),
                    EvaluationInput(id="c1", input="b"),
                ],
            )

    def test_dataset_get_case(self):
        dataset = EvaluationDataset(
            name="d1",
            cases=[
                EvaluationInput(id="c1", input="hello"),
                EvaluationInput(id="c2", input="world"),
            ],
        )

        case = dataset.get_case("c2")
        assert case.input == "world"

    def test_dataset_get_case_not_found(self):
        dataset = EvaluationDataset(name="d1", cases=[EvaluationInput(id="c1", input="hello")])

        with pytest.raises(KeyError, match="Case not found"):
            dataset.get_case("c2")