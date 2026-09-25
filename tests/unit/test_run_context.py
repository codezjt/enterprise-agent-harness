from enterprise_harness.policy.rbac import Principal
from enterprise_harness.runtime import (
    Run,
    RunContext,
    RunContextFactory,
)


def test_create_run_context():

    run = Run(
        run_id="run-001",
        agent_id="order-agent",
        task="查询订单 ORD001",
        context={
            "tenant_id": "tenant-001",
        },
    )

    principal = Principal(
        principal_id="user-001",
        role="operator",
    )

    context = RunContextFactory.create(
        run,
        principal=principal,
    )

    assert context.run_id == "run-001"
    assert context.agent_id == "order-agent"
    assert context.task == "查询订单 ORD001"

    assert context.context["tenant_id"] == "tenant-001"

    assert context.principal is not None
    assert context.principal.principal_id == "user-001"

    assert context.trace_root_span_id is None
    assert context.current_span_id is None