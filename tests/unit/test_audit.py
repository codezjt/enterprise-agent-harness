from datetime import datetime, timezone

from enterprise_harness.observability import (
    AuditEvent,
    AuditLogger,
)


def test_audit_event_contains_required_fields():
    timestamp = datetime.now(timezone.utc)

    event = AuditEvent(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        arguments={
            "order_id": "1001",
            "quantity": 80,
        },
        decision="APPROVED",
        approver="manager-001",
        result={
            "success": True,
        },
        timestamp=timestamp,
    )

    assert event.event == "tool_execution"
    assert event.user_id == "u1001"
    assert event.agent_id == "order-agent"
    assert event.run_id == "run-001"
    assert event.tool == "update_order"
    assert event.arguments["order_id"] == "1001"
    assert event.decision == "APPROVED"
    assert event.approver == "manager-001"
    assert event.result["success"] is True
    assert event.timestamp == timestamp


def test_audit_event_to_dict():
    event = AuditEvent(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        arguments={"order_id": "1001"},
        decision="APPROVED",
        approver="manager-001",
        result={"success": True},
    )

    data = event.to_dict()

    assert data["event"] == "tool_execution"
    assert data["user_id"] == "u1001"
    assert data["agent_id"] == "order-agent"
    assert data["run_id"] == "run-001"
    assert data["tool"] == "update_order"
    assert data["arguments"] == {"order_id": "1001"}
    assert data["decision"] == "APPROVED"
    assert data["approver"] == "manager-001"
    assert data["result"] == {"success": True}
    assert data["timestamp"]


def test_audit_logger_records_event():
    logger = AuditLogger()

    event = logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        arguments={"order_id": "1001"},
        decision="APPROVED",
        approver="manager-001",
        result={"success": True},
    )

    events = logger.get_events()

    assert len(events) == 1
    assert events[0] is event


def test_audit_logger_get_run_events():
    logger = AuditLogger()

    logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        decision="APPROVED",
    )

    logger.record(
        event="tool_execution",
        user_id="u1002",
        agent_id="order-agent",
        run_id="run-002",
        tool="update_order",
        decision="DENIED",
    )

    events = logger.get_run_events("run-001")

    assert len(events) == 1
    assert events[0].run_id == "run-001"


def test_audit_logger_get_tool_events():
    logger = AuditLogger()

    logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        decision="APPROVED",
    )

    logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="query_order",
        decision="ALLOW",
    )

    events = logger.get_tool_events("update_order")

    assert len(events) == 1
    assert events[0].tool == "update_order"


def test_audit_logger_snapshot():
    logger = AuditLogger()

    logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
        arguments={"order_id": "1001"},
        decision="APPROVED",
        approver="manager-001",
    )

    snapshot = logger.snapshot()

    assert len(snapshot) == 1
    assert snapshot[0]["event"] == "tool_execution"
    assert snapshot[0]["user_id"] == "u1001"
    assert snapshot[0]["tool"] == "update_order"
    assert snapshot[0]["decision"] == "APPROVED"
    assert snapshot[0]["approver"] == "manager-001"


def test_audit_logger_clear():
    logger = AuditLogger()

    logger.record(
        event="tool_execution",
        user_id="u1001",
        agent_id="order-agent",
        run_id="run-001",
        tool="update_order",
    )

    assert len(logger.get_events()) == 1

    logger.clear()

    assert logger.get_events() == []
    assert logger.snapshot() == []