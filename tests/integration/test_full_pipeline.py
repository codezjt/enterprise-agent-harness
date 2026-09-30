"""
Enterprise Agent Harness - 集成测试

覆盖完整链路:
  Tool Gateway → Policy Engine (RBAC) → RunManager
  HITL Approval → Trace → Audit → Metrics
"""

from __future__ import annotations

import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.gateway.exceptions import ApprovalRequiredError
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.registry import ToolDefinition, ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.observability.audit import AuditLogger
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.policy.engine import PolicyDecision, PolicyEngine
from enterprise_harness.policy.rbac import Principal, RBAC, Role
from enterprise_harness.runtime.manager import RunManager


class InMemorySystem:
    def __init__(self) -> None:
        self._orders: dict[str, dict] = {"1001": {"quantity": 50}}
        self._inventory: dict[str, int] = {"widget": 100}

    def query_order(self, order_id: str) -> dict:
        return {"order_id": order_id, **self._orders.get(order_id, {})}

    def query_inventory(self, item: str) -> dict:
        return {"item": item, "stock": self._inventory.get(item, 0)}

    def update_order(self, order_id: str, quantity: int) -> dict:
        self._orders[order_id] = {"quantity": quantity}
        return {"updated": True, "order_id": order_id, "quantity": quantity}


def build_tools(system: InMemorySystem) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="query_order",
            description="查询订单",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            risk_level="LOW",
            permissions=["order:read"],
            handler=system.query_order,
        ),
        ToolDefinition(
            name="query_inventory",
            description="查询库存",
            input_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
            risk_level="LOW",
            permissions=["order:read"],
            handler=system.query_inventory,
        ),
        ToolDefinition(
            name="update_order",
            description="修改订单（高风险）",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["order_id"],
            },
            risk_level="HIGH",
            permissions=["order:update"],
            handler=system.update_order,
        ),
    ]


@pytest.fixture
def tool_registry() -> ToolRegistry:
    system = InMemorySystem()
    reg = ToolRegistry()
    for t in build_tools(system):
        reg.register(t)
    return reg


@pytest.fixture
def rbac() -> RBAC:
    return RBAC(
        roles=[
            Role(name="viewer", permissions=frozenset({"order:read"})),
            Role(
                name="manager",
                permissions=frozenset({"order:read", "order:update"}),
            ),
        ]
    )


@pytest.fixture
def trace_manager() -> TraceManager:
    return TraceManager()


@pytest.fixture
def audit_logger() -> AuditLogger:
    return AuditLogger()


@pytest.fixture
def metric_collector() -> MetricCollector:
    return MetricCollector()


@pytest.fixture
def gateway(
    tool_registry: ToolRegistry,
    rbac: RBAC,
    trace_manager: TraceManager,
    audit_logger: AuditLogger,
    metric_collector: MetricCollector,
) -> ToolGateway:
    return ToolGateway(
        registry=tool_registry,
        router=ToolRouter(tool_registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=rbac),
        trace_manager=trace_manager,
        audit_logger=audit_logger,
        metric_collector=metric_collector,
    )


@pytest.mark.asyncio
async def test_tool_gateway_allows_low_risk(
    gateway: ToolGateway,
    trace_manager: TraceManager,
    audit_logger: AuditLogger,
):
    principal = Principal(principal_id="u1", role="viewer")
    run_mgr = RunManager(trace_manager=trace_manager)
    run = await run_mgr.create_run(
        agent_id="test",
        task="test task",
        context={},
    )

    result = await gateway.execute(
        "query_order",
        {"order_id": "1001"},
        run_id=run.run_id,
        principal=principal,
    )

    assert result["order_id"] == "1001"
    assert result["quantity"] == 50

    events = audit_logger.get_run_events(run.run_id)
    assert len(events) == 1
    assert events[0].decision == PolicyDecision.ALLOW

    spans = trace_manager.get_run_spans(run.run_id)
    tool_spans = [s for s in spans if s.component.value == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].status.value == "success"


@pytest.mark.asyncio
async def test_tool_gateway_denies_without_permission(
    gateway: ToolGateway,
    trace_manager: TraceManager,
):
    principal = Principal(principal_id="u1", role="viewer")
    run_mgr = RunManager(trace_manager=trace_manager)
    run = await run_mgr.create_run(
        agent_id="test",
        task="test task",
        context={},
    )

    with pytest.raises(PermissionError):
        await gateway.execute(
            "update_order",
            {"order_id": "1001", "quantity": 30},
            run_id=run.run_id,
            principal=principal,
        )


@pytest.mark.asyncio
async def test_tool_gateway_requires_approval_for_high_risk(
    gateway: ToolGateway,
    trace_manager: TraceManager,
    audit_logger: AuditLogger,
):
    principal = Principal(principal_id="u1", role="manager")
    run_mgr = RunManager(trace_manager=trace_manager)
    run = await run_mgr.create_run(
        agent_id="test",
        task="test task",
        context={},
    )

    with pytest.raises(ApprovalRequiredError) as exc_info:
        await gateway.execute(
            "update_order",
            {"order_id": "1001", "quantity": 30},
            run_id=run.run_id,
            principal=principal,
        )

    approval_id = exc_info.value.approval_id
    assert approval_id

    approval = await gateway.approval_manager.get_request(approval_id)
    assert approval.status.value == "PENDING"

    events = audit_logger.get_run_events(run.run_id)
    assert len(events) == 1
    assert events[0].decision == PolicyDecision.REQUIRE_APPROVAL


@pytest.mark.asyncio
async def test_hitl_approval_workflow(
    gateway: ToolGateway,
    trace_manager: TraceManager,
):
    principal = Principal(principal_id="u1", role="manager")
    run_mgr = RunManager(trace_manager=trace_manager)
    run = await run_mgr.create_run(
        agent_id="test",
        task="test task",
        context={},
    )

    with pytest.raises(ApprovalRequiredError) as exc_info:
        await gateway.execute(
            "update_order",
            {"order_id": "1001", "quantity": 30},
            run_id=run.run_id,
            principal=principal,
        )

    approval_id = exc_info.value.approval_id
    await gateway.approval_manager.approve(approval_id, comment="ok")

    result = await gateway.execute(
        "update_order",
        {"order_id": "1001", "quantity": 30},
        run_id=run.run_id,
        principal=principal,
        approval_id=approval_id,
    )

    assert result["updated"] is True
    assert result["quantity"] == 30

    approval = await gateway.approval_manager.get_request(approval_id)
    assert approval.status.value == "APPROVED"


def test_agent_registry():
    registry = AgentRegistry()

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="mock",
        system_prompt="test",
    )

    registry.register(config, description="test agent", owner="test-team")

    profile = registry.get_profile("test-agent")
    assert profile is not None
    assert profile.name == "Test Agent"
    assert profile.owner == "test-team"

    configs = registry.list_agents()
    assert any(p.agent_id == "test-agent" for p in configs)

    registry.deactivate("test-agent")
    inactive = registry.get_profile("test-agent")
    assert inactive is not None
    assert inactive.status.value == "INACTIVE"


@pytest.mark.asyncio
async def test_run_manager_lifecycle(trace_manager: TraceManager):
    run_mgr = RunManager(trace_manager=trace_manager)

    run = await run_mgr.create_run(
        agent_id="test",
        task="test task",
        context={"key": "value"},
    )

    assert run.run_id
    assert run.agent_id == "test"

    retrieved = await run_mgr.get_run(run.run_id)
    assert retrieved is not None
    assert retrieved.run_id == run.run_id


def test_policy_engine_risk_level_escalation(rbac: RBAC):
    engine = PolicyEngine(rbac=rbac)
    manager = Principal(principal_id="u1", role="manager")

    low_tool = ToolDefinition(
        name="t",
        description="t",
        input_schema={},
        risk_level="LOW",
        permissions=["order:read"],
        handler=lambda: None,
    )

    high_tool = ToolDefinition(
        name="t",
        description="t",
        input_schema={},
        risk_level="HIGH",
        permissions=["order:update"],
        handler=lambda: None,
    )

    decision_low = engine.check(low_tool, {}, principal=manager)
    assert decision_low == PolicyDecision.ALLOW

    decision_high = engine.check(high_tool, {}, principal=manager)
    assert decision_high == PolicyDecision.REQUIRE_APPROVAL


@pytest.mark.asyncio
async def test_metrics_collection(
    gateway: ToolGateway,
    trace_manager: TraceManager,
    metric_collector: MetricCollector,
):
    principal = Principal(principal_id="u1", role="manager")
    run_mgr = RunManager(trace_manager=trace_manager)
    run = await run_mgr.create_run(
        agent_id="test",
        task="test",
        context={},
    )

    await gateway.execute(
        "query_order",
        {"order_id": "1001"},
        run_id=run.run_id,
        principal=principal,
    )
    await gateway.execute(
        "query_inventory",
        {"item": "widget"},
        run_id=run.run_id,
        principal=principal,
    )

    snap = metric_collector.snapshot()
    assert snap.tool_calls == 2
    assert snap.tool_successes == 2
