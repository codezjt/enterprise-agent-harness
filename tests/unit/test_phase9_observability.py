import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.observability import (
    AuditLogger,
    MetricCollector,
    ModelPricing,
    ObservabilityManager,
    TraceManager,
)
from enterprise_harness.observability.cost import CostTracker
from enterprise_harness.observability.metrics import MetricSnapshot
from enterprise_harness.observability.trace import SpanStatus, SpanType
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.runtime.manager import RunManager
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeSimpleRuntime(AgentRuntime):
    async def run(self, task: str, context=None):
        return {"answer": f"completed: {task}"}


class TestObservabilityManager:

    def test_snapshot_exports_all_dimensions(self):
        trace_mgr = TraceManager()
        metrics = MetricCollector()
        audit = AuditLogger()
        cost = CostTracker(ModelPricing(input_per_1k=0.01, output_per_1k=0.03))

        obs = ObservabilityManager(
            trace_manager=trace_mgr,
            metric_collector=metrics,
            audit_logger=audit,
            cost_tracker=cost,
        )

        metrics.record_agent_run(success=True)
        metrics.record_tool(success=True)
        cost.record(input_tokens=100, output_tokens=50)
        audit.record(
            event="tool.call",
            user_id="user-1",
            agent_id="agent-1",
            run_id="run-1",
            tool="query_order",
        )

        snapshot = obs.snapshot()
        data = snapshot.to_dict()

        assert data["metrics"]["agent_runs"] == 1
        assert data["metrics"]["agent_successes"] == 1
        assert data["metrics"]["tool_calls"] == 1
        assert len(data["audits"]) == 1
        assert data["audits"][0]["event"] == "tool.call"
        assert data["cost"]["input_tokens"] == 100
        assert data["cost"]["output_tokens"] == 50

    def test_snapshot_scoped_by_run_id(self):
        trace_mgr = TraceManager()
        audit = AuditLogger()

        span1 = trace_mgr.start_run_span("run-1", "agent-1")
        trace_mgr.finish_span(span1.span_id)

        span2 = trace_mgr.start_run_span("run-2", "agent-2")
        trace_mgr.finish_span(span2.span_id)

        audit.record(
            event="tool.call", user_id="u1", agent_id="a1",
            run_id="run-1", tool="t1",
        )
        audit.record(
            event="tool.call", user_id="u2", agent_id="a2",
            run_id="run-2", tool="t2",
        )

        obs = ObservabilityManager(
            trace_manager=trace_mgr,
            audit_logger=audit,
        )

        snapshot = obs.snapshot(run_id="run-1")
        data = snapshot.to_dict()

        assert len(data["traces"]) == 1
        assert len(data["audits"]) == 1
        assert data["audits"][0]["run_id"] == "run-1"

    def test_snapshot_tenant_scoped(self):
        audit = AuditLogger()
        audit.record(
            event="tool.call", user_id="u1", agent_id="a1",
            run_id="r1", tool="t1", tenant_id="tenant-a",
        )
        audit.record(
            event="tool.call", user_id="u2", agent_id="a2",
            run_id="r2", tool="t2", tenant_id="tenant-b",
        )

        obs = ObservabilityManager(audit_logger=audit)
        snapshot = obs.snapshot(tenant_id="tenant-a")
        data = snapshot.to_dict()

        assert len(data["audits"]) == 1
        assert data["audits"][0]["tenant_id"] == "tenant-a"

    def test_clear_resets_all(self):
        metrics = MetricCollector()
        audit = AuditLogger()
        cost = CostTracker(ModelPricing(input_per_1k=0.01, output_per_1k=0.03))

        obs = ObservabilityManager(
            metric_collector=metrics,
            audit_logger=audit,
            cost_tracker=cost,
        )

        metrics.record_agent_run(success=True)
        metrics.record_tool(success=True)
        cost.record(input_tokens=100, output_tokens=50)
        audit.record(
            event="tool.call", user_id="u1", agent_id="a1",
            run_id="r1", tool="t1",
        )

        obs.clear()

        snapshot = obs.snapshot()
        data = snapshot.to_dict()
        assert data["metrics"]["agent_runs"] == 0
        assert data["metrics"]["tool_calls"] == 0
        assert len(data["audits"]) == 0
        assert data["cost"]["total_tokens"] == 0


class TestMetricsProduceRealData:

    @pytest.mark.asyncio
    async def test_run_manager_produces_agent_metrics(self):
        runtime = FakeSimpleRuntime(
            config=AgentConfig(
                agent_id="test-agent",
                name="Test Agent",
                model="test-model",
            )
        )

        metric_collector = MetricCollector()

        manager = RunManager(
            runtime=FakeLangGraphRuntimeForAgent(runtime),
            metric_collector=metric_collector,
        )

        run = await manager.create_run(
            agent_id="test-agent",
            task="hello",
        )

        await manager.start_run(
            run_id=run.run_id,
            runtime=FakeLangGraphRuntimeForAgent(runtime),
        )

        snapshot = metric_collector.snapshot()
        assert snapshot.agent_runs == 1
        assert snapshot.agent_successes == 1

    @pytest.mark.asyncio
    async def test_run_manager_produces_failure_metrics(self):
        class FailingRuntime(AgentRuntime):
            async def run(self, task: str, context=None):
                raise RuntimeError("agent failure")

        runtime = FailingRuntime(
            config=AgentConfig(
                agent_id="fail-agent",
                name="Fail Agent",
                model="test-model",
            )
        )

        metric_collector = MetricCollector()

        manager = RunManager(
            runtime=FakeLangGraphRuntimeForAgent(runtime),
            metric_collector=metric_collector,
        )

        run = await manager.create_run(
            agent_id="fail-agent",
            task="will fail",
        )

        await manager.start_run(
            run_id=run.run_id,
            runtime=FakeLangGraphRuntimeForAgent(runtime),
        )

        snapshot = metric_collector.snapshot()
        assert snapshot.agent_runs == 1
        assert snapshot.agent_successes == 0


class TestApprovalMetrics:

    @pytest.mark.asyncio
    async def test_approval_produces_human_approval_metric(self):
        metric_collector = MetricCollector()
        mgr = ApprovalManager(metric_collector=metric_collector)

        req = await mgr.create_request(
            run_id="run-1",
            tool_name="delete_order",
            arguments={"order_id": "999"},
        )

        await mgr.approve(req.approval_id)

        snapshot = metric_collector.snapshot()
        assert snapshot.human_approvals == 1


class TestTracingProducesRealData:

    @pytest.mark.asyncio
    async def test_run_produces_trace_spans(self):
        trace_mgr = TraceManager()
        runtime = FakeSimpleRuntime(
            config=AgentConfig(
                agent_id="tracing-agent",
                name="Tracing Agent",
                model="test-model",
            )
        )

        manager = RunManager(
            runtime=FakeLangGraphRuntimeForAgent(runtime),
            trace_manager=trace_mgr,
        )

        run = await manager.create_run(
            agent_id="tracing-agent",
            task="trace me",
        )

        await manager.start_run(
            run_id=run.run_id,
            runtime=FakeLangGraphRuntimeForAgent(runtime),
        )

        spans = trace_mgr.get_run_spans(run.run_id)
        assert len(spans) >= 1

        root_span = [s for s in spans if s.component == SpanType.AGENT][0]
        assert root_span.status == SpanStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_failed_run_produces_failed_trace(self):
        trace_mgr = TraceManager()

        class FailingRuntime(AgentRuntime):
            async def run(self, task: str, context=None):
                raise RuntimeError("test failure")

        runtime = FailingRuntime(
            config=AgentConfig(
                agent_id="fail-trace",
                name="Fail Trace",
                model="test-model",
            )
        )

        manager = RunManager(
            runtime=FakeLangGraphRuntimeForAgent(runtime),
            trace_manager=trace_mgr,
        )

        run = await manager.create_run(
            agent_id="fail-trace",
            task="will fail",
        )

        await manager.start_run(
            run_id=run.run_id,
            runtime=FakeLangGraphRuntimeForAgent(runtime),
        )

        spans = trace_mgr.get_run_spans(run.run_id)
        root_span = [s for s in spans if s.component == SpanType.AGENT][0]
        assert root_span.status == SpanStatus.FAILED
        assert "test failure" in root_span.metadata["error"]

    @pytest.mark.asyncio
    async def test_trace_tenant_isolation(self):
        trace_mgr = TraceManager()
        runtime = FakeSimpleRuntime(
            config=AgentConfig(
                agent_id="tenant-agent",
                name="Tenant Agent",
                model="test-model",
            )
        )

        manager = RunManager(
            runtime=FakeLangGraphRuntimeForAgent(runtime),
            trace_manager=trace_mgr,
        )

        run_a = await manager.create_run(
            agent_id="tenant-agent",
            task="task a",
            tenant_id="tenant-x",
        )
        run_b = await manager.create_run(
            agent_id="tenant-agent",
            task="task b",
            tenant_id="tenant-y",
        )

        await manager.start_run(run_id=run_a.run_id, runtime=FakeLangGraphRuntimeForAgent(runtime))
        await manager.start_run(run_id=run_b.run_id, runtime=FakeLangGraphRuntimeForAgent(runtime))

        x_spans = trace_mgr.get_tenant_spans("tenant-x")
        y_spans = trace_mgr.get_tenant_spans("tenant-y")

        assert len(x_spans) >= 1
        assert len(y_spans) >= 1
        assert all(s.tenant_id == "tenant-x" for s in x_spans)
        assert all(s.tenant_id == "tenant-y" for s in y_spans)


class TestCostTracker:

    def test_cost_accumulation(self):
        cost = CostTracker(
            ModelPricing(input_per_1k=0.01, output_per_1k=0.03)
        )

        cost.record(input_tokens=1000, output_tokens=500)

        snap = cost.snapshot()
        assert snap["input_tokens"] == 1000
        assert snap["output_tokens"] == 500