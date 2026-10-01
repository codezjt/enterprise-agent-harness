import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.models import ToolDefinition
from enterprise_harness.gateway.registry import ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.observability.audit import AuditLogger, AuditEvent
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.trace import SpanStatus, SpanType, TraceSpan
from enterprise_harness.policy.approval import ApprovalRequest, ApprovalStatus
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.policy.rbac import Principal
from enterprise_harness.runtime import RunManager


class TestTenantAuditIsolation:

    def test_audit_events_scoped_by_tenant(self):
        al = AuditLogger()
        al.record(
            event="tool_execution", user_id="u1", agent_id="a1", run_id="r1",
            tool="read", tenant_id="tenant-a",
            result={"success": True},
        )
        al.record(
            event="tool_execution", user_id="u2", agent_id="a2", run_id="r2",
            tool="write", tenant_id="tenant-b",
            result={"success": False},
        )

        all_events = al.get_events()
        assert len(all_events) == 2

        tenant_a = al.get_events(tenant_id="tenant-a")
        assert len(tenant_a) == 1
        assert tenant_a[0].tool == "read"

        tenant_b = al.get_events(tenant_id="tenant-b")
        assert len(tenant_b) == 1
        assert tenant_b[0].tool == "write"

        tenant_c = al.get_events(tenant_id="tenant-c")
        assert len(tenant_c) == 0

    def test_audit_event_has_tenant_id(self):
        al = AuditLogger()
        event = al.record(
            event="tool_execution", user_id="u1", agent_id="a1", run_id="r1",
            tool="read", tenant_id="t-123",
            result={"success": True},
        )

        assert event.tenant_id == "t-123"
        d = event.to_dict()
        assert d["tenant_id"] == "t-123"


class TestTenantTraceIsolation:

    def test_trace_span_has_tenant_id(self):
        span = TraceSpan(
            trace_id="t1", span_id="s1", parent_span_id=None,
            run_id="r1", component=SpanType.AGENT,
            tenant_id="tenant-x",
        )
        assert span.tenant_id == "tenant-x"
        assert span.to_dict()["tenant_id"] == "tenant-x"

    def test_trace_manager_start_span_with_tenant(self):
        tm = TraceManager()
        span = tm.start_span(
            run_id="r1",
            span_type=SpanType.TOOL,
            name="test",
            tenant_id="tenant-z",
        )
        assert span.tenant_id == "tenant-z"

    def test_trace_manager_tenant_inherits_from_parent(self):
        tm = TraceManager()
        root = tm.start_span(
            run_id="r1",
            span_type=SpanType.AGENT,
            tenant_id="tenant-parent",
        )
        tm._span_stack.setdefault("r1", []).append(root)

        child = tm.start_span(
            run_id="r1",
            span_type=SpanType.TOOL,
            name="child",
            parent_span_id=root.span_id,
        )
        assert child.tenant_id == "tenant-parent"

    def test_trace_manager_get_tenant_spans(self):
        tm = TraceManager()
        tm.start_span(run_id="r1", span_type=SpanType.TOOL, tenant_id="t-a")
        tm.start_span(run_id="r2", span_type=SpanType.TOOL, tenant_id="t-b")
        tm.start_span(run_id="r3", span_type=SpanType.TOOL, tenant_id="t-a")

        t_a_spans = tm.get_tenant_spans("t-a")
        assert len(t_a_spans) == 2

        t_b_spans = tm.get_tenant_spans("t-b")
        assert len(t_b_spans) == 1

    def test_trace_manager_get_run_spans_with_tenant_filter(self):
        tm = TraceManager()
        s1 = tm.start_span(run_id="same-run", span_type=SpanType.TOOL, tenant_id="t-a")
        s2 = tm.start_span(run_id="same-run", span_type=SpanType.TOOL, tenant_id="t-b")

        all_for_run = tm.get_run_spans("same-run")
        assert len(all_for_run) == 2

        only_ta = tm.get_run_spans("same-run", tenant_id="t-a")
        assert len(only_ta) == 1
        assert only_ta[0].span_id == s1.span_id


class TestTenantRunManagerIsolation:

    @pytest.mark.asyncio
    async def test_get_runs_scoped_by_tenant(self):
        manager = RunManager()
        run_a = await manager.create_run(agent_id="a1", task="t1", tenant_id="tenant-a")
        run_b = await manager.create_run(agent_id="a2", task="t2", tenant_id="tenant-b")

        a_runs = manager.get_runs(tenant_id="tenant-a")
        assert len(a_runs) == 1
        assert a_runs[0].run_id == run_a.run_id

        b_runs = manager.get_runs(tenant_id="tenant-b")
        assert len(b_runs) == 1
        assert b_runs[0].run_id == run_b.run_id

        c_runs = manager.get_runs(tenant_id="tenant-c")
        assert len(c_runs) == 0

    @pytest.mark.asyncio
    async def test_run_has_tenant_id_and_agent_version(self):
        manager = RunManager()
        run = await manager.create_run(
            agent_id="a1", task="t1",
            tenant_id="my-tenant",
            agent_version="2.0.0",
        )
        assert run.tenant_id == "my-tenant"
        assert run.agent_version == "2.0.0"


class TestTenantAgentRegistryIsolation:

    def test_list_agents_scoped_by_tenant(self):
        registry = AgentRegistry()
        registry.register(
            AgentConfig(agent_id="agent-a", name="Agent A", model="m1", tools=[]),
            tenant_id="tenant-1",
        )
        registry.register(
            AgentConfig(agent_id="agent-b", name="Agent B", model="m2", tools=[]),
            tenant_id="tenant-2",
        )

        all_agents = registry.list_agents()
        assert len(all_agents) == 2

        t1 = registry.list_agents(tenant_id="tenant-1")
        assert len(t1) == 1
        assert t1[0].agent_id == "agent-a"

        t2 = registry.list_agents(tenant_id="tenant-2")
        assert len(t2) == 1
        assert t2[0].agent_id == "agent-b"

        t3 = registry.list_agents(tenant_id="tenant-3")
        assert len(t3) == 0


class TestTenantApprovalManagerIsolation:

    @pytest.mark.asyncio
    async def test_approval_request_has_tenant_id(self):
        am = ApprovalManager()
        req = await am.create_request(
            run_id="r1", tool_name="write",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
            tenant_id="tnnt-abc",
        )
        assert req.tenant_id == "tnnt-abc"

    @pytest.mark.asyncio
    async def test_list_pending_scoped_by_tenant(self):
        am = ApprovalManager()
        await am.create_request(run_id="r1", tool_name="w1", arguments={}, tenant_id="t-a")
        await am.create_request(run_id="r2", tool_name="w2", arguments={}, tenant_id="t-b")
        await am.create_request(run_id="r3", tool_name="w3", arguments={}, tenant_id="t-a")

        pending_a = await am.list_pending(tenant_id="t-a")
        assert len(pending_a) == 2

        pending_b = await am.list_pending(tenant_id="t-b")
        assert len(pending_b) == 1

        pending_c = await am.list_pending(tenant_id="t-c")
        assert len(pending_c) == 0


class TestTenantToolGatewayPassesTenantId:

    @pytest.mark.asyncio
    async def test_gateway_execute_passes_tenant_id_to_audit(self):
        audit_logger = AuditLogger()
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="safe", handler=lambda: "ok")
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            audit_logger=audit_logger,
        )

        principal = Principal(principal_id="u1", role="admin")
        await gw.execute(
            tool_name="safe", arguments={}, principal=principal,
            tenant_id="corp-xyz",
        )

        events = audit_logger.get_events()
        assert len(events) == 1
        assert events[0].tenant_id == "corp-xyz"


class TestTenantToolGatewayPassesTenantToTrace:

    @pytest.mark.asyncio
    async def test_gateway_span_has_tenant_id(self):
        tm = TraceManager()
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="safe", handler=lambda: "ok")
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            trace_manager=tm,
        )

        principal = Principal(principal_id="u1", role="admin")

        await gw.execute(
            tool_name="safe", arguments={}, principal=principal,
            tenant_id="org-alpha",
        )

        spans = tm.snapshot()
        assert len(spans) == 1
        assert spans[0]["tenant_id"] == "org-alpha"


class TestAgentVersion:

    def test_agent_config_has_version(self):
        config = AgentConfig(
            agent_id="a1", name="Test", model="m1", version="3.2.1",
        )
        assert config.version == "3.2.1"

    def test_agent_config_default_version(self):
        config = AgentConfig(agent_id="a1", name="Test", model="m1")
        assert config.version == "1.0.0"

    def test_agent_registry_tracks_version(self):
        registry = AgentRegistry()
        registry.register(
            AgentConfig(agent_id="a1", name="Test", model="m1", version="1.0.0"),
        )
        registry.update_config(
            AgentConfig(agent_id="a1", name="Test", model="m1", version="2.0.0"),
        )

        profile = registry.get_profile("a1")
        assert profile.current_version == "2.0.0"
        assert "1.0.0" in profile.versions
        assert "2.0.0" in profile.versions

        config_v1 = registry.get_config("a1", version="1.0.0")
        assert config_v1.version == "1.0.0"

        config_v2 = registry.get_config("a1", version="2.0.0")
        assert config_v2.version == "2.0.0"