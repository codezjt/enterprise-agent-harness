from enterprise_harness.gateway import ToolDefinition
from enterprise_harness.policy import (
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
)


def create_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
    )


def test_policy_allow():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.ALLOW,
            )
        ]
    )

    decision = engine.check(
        tool=create_tool("query_order"),
        arguments={},
    )

    assert decision == PolicyDecision.ALLOW


def test_policy_deny():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="delete_order",
                decision=PolicyDecision.DENY,
            )
        ]
    )

    decision = engine.check(
        tool=create_tool("delete_order"),
        arguments={},
    )

    assert decision == PolicyDecision.DENY


def test_policy_require_approval():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="cancel_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    decision = engine.check(
        tool=create_tool("cancel_order"),
        arguments={},
    )

    assert decision == PolicyDecision.REQUIRE_APPROVAL


def test_policy_default_allow():
    engine = PolicyEngine()

    decision = engine.check(
        tool=create_tool("query_order"),
        arguments={},
    )

    assert decision == PolicyDecision.ALLOW


def test_policy_multiple_rules():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.ALLOW,
            ),
            PolicyRule(
                tool_name="cancel_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            ),
            PolicyRule(
                tool_name="delete_order",
                decision=PolicyDecision.DENY,
            ),
        ]
    )

    assert (
        engine.check(
            create_tool("query_order"),
            {},
        )
        == PolicyDecision.ALLOW
    )

    assert (
        engine.check(
            create_tool("cancel_order"),
            {},
        )
        == PolicyDecision.REQUIRE_APPROVAL
    )

    assert (
        engine.check(
            create_tool("delete_order"),
            {},
        )
        == PolicyDecision.DENY
    )