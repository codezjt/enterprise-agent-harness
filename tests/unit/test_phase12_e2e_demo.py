import os
import re
import tempfile
import pytest
from fastapi.testclient import TestClient

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.api.server import app, init_api
from enterprise_harness.gateway import (
    ApprovalRequiredError,
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.gateway.exceptions import ToolNotFoundError
from enterprise_harness.observability import (
    AuditLogger,
    MetricCollector,
    TraceManager,
)
from enterprise_harness.policy import PolicyEngine, Principal, RBAC, Role
from enterprise_harness.repositories import SqliteRunRepository
from enterprise_harness.runtime import RunManager, RunStatus


class MockAgentRuntime(AgentRuntime):

    def __init__(self, config, tool_gateway, metric_collector=None):
        super().__init__(config, metric_collector=metric_collector)
        self.tool_gateway = tool_gateway

    async def run(self, task, context=None):
        return {"status": "completed"}

    async def run_with_context(self, run_context):
        results = {}
        for tool_name in self.config.tools:
            args = self._extract_args(tool_name, run_context.task)
            try:
                result = await self.tool_gateway.execute(
                    tool_name=tool_name,
                    arguments=args,
                    principal=Principal(principal_id="mock-u1", role="viewer"),
                    run_id=run_context.run_id,
                )
                results[tool_name] = result
            except Exception:
                pass

        if self.metric_collector is not None:
            self.metric_collector.record_agent_run(success=True)

        return {"status": "completed", "tool_results": results}

    @staticmethod
    def _extract_args(tool_name, task):
        args = {}
        match = re.search(r'(?:ORD|order)[-_\s]*(\d+)', task, re.IGNORECASE)
        if match:
            args["order_id"] = f"ORD-{match.group(1)}"
        if tool_name == "update_order":
            if "shipped" in task.lower():
                args["status"] = "SHIPPED"
            elif "cancel" in task.lower():
                args["status"] = "CANCELLED"
        return args


def _build_full_gateway(
    trace_manager=None,
    audit_logger=None,
    metric_collector=None,
):
    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="query_order",
        description="Query an order by order_id",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        risk_level="LOW",
        permissions=["order:read"],
        handler=lambda order_id: {"order_id": order_id, "status": "PAID", "amount": 99.9},
    ))

    registry.register(ToolDefinition(
        name="update_order",
        description="Update order status",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["order_id", "status"],
        },
        risk_level="HIGH",
        permissions=["order:write"],
        handler=lambda order_id, status: {"order_id": order_id, "status": status},
    ))

    registry.register(ToolDefinition(
        name="cancel_order",
        description="Cancel an order",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        risk_level="CRITICAL",
        permissions=["order:cancel"],
        handler=lambda order_id: {"order_id": order_id, "cancelled": True},
    ))

    rbac = RBAC(roles=[
        Role(name="viewer", permissions=frozenset({"order:read"})),
        Role(name="editor", permissions=frozenset({"order:read", "order:write"})),
        Role(name="manager", permissions=frozenset({"order:read", "order:write", "order:cancel"})),
    ])

    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=rbac),
        trace_manager=trace_manager or TraceManager(),
        audit_logger=audit_logger or AuditLogger(),
        metric_collector=metric_collector or MetricCollector(),
    )


def _build_agent_runtime(config, gateway):
    return MockAgentRuntime(config=config, tool_gateway=gateway)


class TestFullClosedLoop:
    """Harness 完整闭环：Agent → RunManager → ToolGateway → Policy/RBAC → HITL → Trace/Audit/Metrics"""

    @pytest.mark.asyncio
    async def test_low_risk_tool_full_chain(self):
        trace_mgr = TraceManager()
        audit_logger = AuditLogger()
        metric_collector = MetricCollector()

        gw = _build_full_gateway(
            trace_manager=trace_mgr,
            audit_logger=audit_logger,
            metric_collector=metric_collector,
        )

        config = AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="openai/gpt-4o-mini",
            tools=["query_order"],
        )
        runtime = _build_agent_runtime(config, gw)

        manager = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        run = await manager.create_run(
            agent_id="order-agent",
            task="query order ORD-001",
        )

        principal = Principal(principal_id="u1", role="viewer")
        run = await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=principal,
        )

        assert run.status == RunStatus.COMPLETED

        spans = trace_mgr.get_run_spans(run.run_id)
        assert len(spans) >= 1

        events = audit_logger.get_events(tenant_id="default")
        assert len(events) >= 1

        snapshot = metric_collector.snapshot()
        assert snapshot.agent_runs >= 1
        assert snapshot.agent_successes >= 1

    @pytest.mark.asyncio
    async def test_policy_denial_due_to_rbac(self):
        gw = _build_full_gateway()
        principal = Principal(principal_id="u1", role="viewer")

        with pytest.raises(PermissionError):
            await gw.execute(
                tool_name="update_order",
                arguments={"order_id": "ORD-001", "status": "SHIPPED"},
                principal=principal,
            )

    @pytest.mark.asyncio
    async def test_high_risk_triggers_approval_and_resume(self):
        gw = _build_full_gateway()
        principal = Principal(principal_id="u2", role="editor")

        with pytest.raises(ApprovalRequiredError) as excinfo:
            await gw.execute(
                tool_name="update_order",
                arguments={"order_id": "ORD-001", "status": "SHIPPED"},
                principal=principal,
            )

        approval_id = excinfo.value.approval_id
        assert approval_id is not None

        request = await gw.approval_manager.get_request(approval_id)
        assert request.tool_name == "update_order"
        assert request.status.value == "PENDING"

        await gw.approval_manager.approve(approval_id, comment="approved")

        result = await gw.execute(
            tool_name="update_order",
            arguments={"order_id": "ORD-001", "status": "SHIPPED"},
            principal=principal,
            approval_id=approval_id,
        )
        assert result["status"] == "SHIPPED"

    @pytest.mark.asyncio
    async def test_critical_tool_requires_manager_role(self):
        gw = _build_full_gateway()
        principal_viewer = Principal(principal_id="u1", role="viewer")

        with pytest.raises(PermissionError):
            await gw.execute(
                tool_name="cancel_order",
                arguments={"order_id": "ORD-001"},
                principal=principal_viewer,
            )

        principal_manager = Principal(principal_id="u3", role="manager")

        with pytest.raises(ApprovalRequiredError) as excinfo:
            await gw.execute(
                tool_name="cancel_order",
                arguments={"order_id": "ORD-001"},
                principal=principal_manager,
            )

        approval_id = excinfo.value.approval_id
        await gw.approval_manager.approve(approval_id)

        result = await gw.execute(
            tool_name="cancel_order",
            arguments={"order_id": "ORD-001"},
            principal=principal_manager,
            approval_id=approval_id,
        )
        assert result["cancelled"] is True


class TestChatApiClosedLoop:

    def test_chat_routes_low_risk_tool_directly(self, _api):
        client = _api["client"]
        resp = client.post("/v1/chat", json={
            "message": "query order ORD-001",
            "principal_id": "u1",
            "principal_role": "viewer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["routed_to"].startswith("tool:")
        assert data["reply"]["status"] == "PAID"

    def test_chat_routes_high_risk_to_approval(self, _api):
        client = _api["client"]
        resp = client.post("/v1/chat", json={
            "message": "update order ORD-001 status to SHIPPED",
            "principal_id": "u2",
            "principal_role": "editor",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "WAITING_APPROVAL"
        assert data["approval_id"] is not None

    def test_chat_approval_workflow(self, _api):
        client = _api["client"]

        resp = client.post("/v1/chat", json={
            "message": "update order ORD-001 status to SHIPPED",
            "principal_id": "u2",
            "principal_role": "editor",
        })
        approval_id = resp.json()["approval_id"]

        resp = client.post(f"/v1/approvals/{approval_id}", json={
            "action": "approve",
            "comment": "looks good",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] in ("approved", "approve")

    @pytest.mark.skip(reason="Requires LLM backend for DeepAgentRuntime")
    def test_chat_unknown_goes_to_agent(self, _api):
        client = _api["client"]
        resp = client.post("/v1/chat", json={
            "message": "analyze quarterly sales performance",
            "principal_id": "u1",
            "principal_role": "viewer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["routed_to"] == "deepagent"


class TestObservabilityClosedLoop:

    @pytest.mark.asyncio
    async def test_trace_spans_record_full_tool_execution(self):
        trace_mgr = TraceManager()
        gw = _build_full_gateway(trace_manager=trace_mgr)
        principal = Principal(principal_id="u1", role="viewer")

        await gw.execute(
            tool_name="query_order",
            arguments={"order_id": "ORD-001"},
            principal=principal,
            run_id="e2e-trace-test",
        )

        spans = trace_mgr.get_run_spans("e2e-trace-test")
        tool_spans = [s for s in spans if s.name == "query_order"]
        assert len(tool_spans) >= 1
        tool_span = tool_spans[0]
        assert tool_span.output["status"] == "PAID"

    @pytest.mark.asyncio
    async def test_metrics_record_agent_runs(self):
        metric_collector = MetricCollector()
        trace_mgr = TraceManager()
        gw = _build_full_gateway(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        config = AgentConfig(
            agent_id="metrics-agent",
            name="Metrics Agent",
            model="openai/gpt-4o-mini",
            tools=["query_order"],
        )
        runtime = _build_agent_runtime(config, gw)

        manager = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        run = await manager.create_run(
            agent_id="metrics-agent",
            task="query ORD-001",
        )
        await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=Principal(principal_id="u1", role="viewer"),
        )

        snapshot = metric_collector.snapshot()
        assert snapshot.agent_runs >= 1
        assert snapshot.agent_successes >= 1

    @pytest.mark.asyncio
    async def test_observability_snapshot_exports_all_data(self):
        trace_mgr = TraceManager()
        audit_logger = AuditLogger()
        metric_collector = MetricCollector()

        gw = _build_full_gateway(
            trace_manager=trace_mgr,
            audit_logger=audit_logger,
            metric_collector=metric_collector,
        )

        config = AgentConfig(
            agent_id="snap-agent",
            name="Snapshot Agent",
            model="openai/gpt-4o-mini",
            tools=["query_order"],
        )
        runtime = _build_agent_runtime(config, gw)

        manager = RunManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
        )

        run = await manager.create_run(
            agent_id="snap-agent",
            task="query ORD-001",
        )
        run = await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=Principal(principal_id="u1", role="viewer"),
        )

        from enterprise_harness.observability.manager import ObservabilityManager

        obs = ObservabilityManager(
            trace_manager=trace_mgr,
            metric_collector=metric_collector,
            audit_logger=audit_logger,
        )
        snapshot = obs.snapshot(run_id=run.run_id)

        assert snapshot.run_id == run.run_id
        assert len(snapshot.traces) >= 1
        assert snapshot.metrics is not None
        assert snapshot.metrics["agent_runs"] >= 1
        assert len(snapshot.audits) >= 1


class TestPersistenceClosedLoop:

    @pytest.mark.asyncio
    async def test_run_survives_restart(self):
        db_dir = tempfile.mkdtemp(prefix="harness_e2e_")
        db_path = os.path.join(db_dir, "e2e.db")
        try:
            repo_a = SqliteRunRepository(db_path)
            repo_b = SqliteRunRepository(db_path)

            manager_a = RunManager(run_repository=repo_a)
            run_a = await manager_a.create_run(agent_id="test", task="persist me")
            run_a.status = RunStatus.COMPLETED
            await repo_a.save(run_a)

            saved = await repo_b.get(run_a.run_id)
            assert saved is not None
            assert saved.run_id == run_a.run_id
            assert saved.status == RunStatus.COMPLETED
            assert saved.task == "persist me"
        finally:
            await repo_a.close()
            await repo_b.close()
            try:
                os.unlink(db_path)
                os.rmdir(db_dir)
            except OSError:
                pass


class TestStateMachineInE2E:

    def test_invalid_transition_in_api(self, _api):
        client = _api["client"]

        resp = client.post("/v1/runs", json={
            "agent_id": "sm-agent", "task": "test",
        })
        run_id = resp.json()["run_id"]

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 409


class TestErrorScenarios:

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self):
        registry = ToolRegistry()
        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
        )

        with pytest.raises(ToolNotFoundError, match="not found"):
            await gw.execute(
                tool_name="nonexistent",
                arguments={},
                principal=Principal(principal_id="u1", role="admin"),
            )

    def test_agent_not_found_returns_404(self, _api):
        client = _api["client"]
        resp = client.get("/v1/agents/nonexistent-agent")
        assert resp.status_code == 404

    def test_get_nonexistent_run_returns_404(self, _api):
        client = _api["client"]
        resp = client.get("/v1/runs/nonexistent-run-id")
        assert resp.status_code == 404

    def test_create_agent_duplicate_returns_409(self, _api):
        client = _api["client"]
        client.post("/v1/agents", json={
            "agent_id": "dup-agent", "name": "Dup", "model": "test",
        })
        resp = client.post("/v1/agents", json={
            "agent_id": "dup-agent", "name": "Dup Again", "model": "test",
        })
        assert resp.status_code == 409


class TestEndToEndApiWorkflow:

    def test_full_api_workflow(self, _api):
        client = _api["client"]

        resp = client.post("/v1/agents", json={
            "agent_id": "workflow-agent", "name": "Workflow Agent",
            "model": "gpt-4", "tools": ["query_order", "update_order"],
        })
        assert resp.status_code == 200
        agent_data = resp.json()
        assert agent_data["agent_id"] == "workflow-agent"
        assert agent_data["status"] == "ACTIVE"

        resp = client.get("/v1/agents")
        agents = resp.json()
        assert any(a["agent_id"] == "workflow-agent" for a in agents)

        resp = client.get("/v1/agents/workflow-agent")
        assert resp.json()["agent_id"] == "workflow-agent"

        resp = client.post("/v1/agents/workflow-agent/disable")
        assert resp.json()["agent_status"] == "INACTIVE"

        resp = client.post("/v1/agents/workflow-agent/enable")
        assert resp.json()["agent_status"] == "ACTIVE"

        resp = client.put("/v1/agents/workflow-agent", json={"description": "pipeline demo"})
        assert resp.status_code == 200

        resp = client.post("/v1/runs", json={
            "agent_id": "workflow-agent", "task": "query order ORD-100",
        })
        assert resp.status_code == 200
        run_data = resp.json()
        run_id = run_data["run_id"]
        assert run_data["status"] == "CREATED"

        resp = client.get(f"/v1/runs/{run_id}")
        assert resp.json()["run_id"] == run_id

        resp = client.get("/v1/runs")
        runs = resp.json()
        assert any(r["run_id"] == run_id for r in runs)

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

        resp = client.get(f"/v1/runs/{run_id}/observability")
        assert resp.status_code == 200
        obs = resp.json()
        assert obs["run_id"] == run_id
        assert "traces" in obs
        assert "metrics" in obs

        resp = client.get("/v1/tools")
        assert resp.status_code == 200
        tools_data = resp.json()
        tool_names = [t["name"] for t in tools_data["tools"]]
        assert "query_order" in tool_names


@pytest.fixture
def _api():
    trace_mgr = TraceManager()
    audit_logger = AuditLogger()
    metric_collector = MetricCollector()

    gw = _build_full_gateway(
        trace_manager=trace_mgr,
        audit_logger=audit_logger,
        metric_collector=metric_collector,
    )

    registry = AgentRegistry()
    registry.register(AgentConfig(
        agent_id="sm-agent",
        name="State Machine Agent",
        model="openai/gpt-4o-mini",
        tools=["query_order", "update_order", "cancel_order"],
    ))

    run_manager = RunManager(
        trace_manager=trace_mgr,
        metric_collector=metric_collector,
    )

    init_api(
        registry=registry,
        run_manager=run_manager,
        trace_manager=trace_mgr,
        approval_manager=gw.approval_manager,
        tool_gateway=gw,
    )

    return {
        "client": TestClient(app),
        "gateway": gw,
        "registry": registry,
        "run_manager": run_manager,
        "trace_manager": trace_mgr,
        "audit_logger": audit_logger,
        "metric_collector": metric_collector,
    }