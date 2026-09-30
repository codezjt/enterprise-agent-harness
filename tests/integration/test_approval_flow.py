from __future__ import annotations

import pytest

from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.orchestration.task import TaskStatus
from enterprise_harness.policy import (
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
)
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus


def update_order(order_id: str, quantity: int):
    return {
        "order_id": order_id,
        "quantity": quantity,
        "updated": True,
    }


def create_gateway(
    approval_manager: ApprovalManager,
) -> ToolGateway:
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="update_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="update_order",
            description="修改订单",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                    "quantity": {
                        "type": "integer",
                    },
                },
                "required": [
                    "order_id",
                    "quantity",
                ],
            },
            handler=update_order,
        )
    )

    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=policy_engine,
        approval_manager=approval_manager,
    )


@pytest.mark.asyncio
async def test_run_manager_approval_flow():
    approval_manager = ApprovalManager()
    gateway = create_gateway(approval_manager)

    execution_count = 0

    async def execute(state):
        nonlocal execution_count

        execution_count += 1

        context = state["context"]
        arguments = context["arguments"]

        return await gateway.execute(
            tool_name="update_order",
            arguments=arguments,
            run_id=run.run_id,
            approval_id=state.get("approval_id"),
        )

    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="european-order-agent",
        task="修改欧洲订单数量",
        context={
            "arguments": {
                "order_id": "1001",
                "quantity": 80,
            }
        },
    )

    # ---------------------------------------------------------
    # 1. 第一次启动
    # ---------------------------------------------------------

    run = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert run.status == RunStatus.WAITING_APPROVAL

    assert run.approval_id is not None

    approval_id = run.approval_id

    # 此时应该只有一个待审批请求
    pending = await approval_manager.list_pending()

    assert len(pending) == 1
    assert pending[0].approval_id == approval_id

    # 第一次执行只走到了 ApprovalRequired，
    # 真正的 update_order handler 还没有执行。
    assert execution_count == 1

    # ---------------------------------------------------------
    # 2. 人工审批
    # ---------------------------------------------------------

    approval = await approval_manager.approve(
        approval_id=approval_id,
        comment="approved by integration test",
    )

    assert approval.approval_id == approval_id
    assert approval.status.value == "APPROVED"

    # 审批已经处理，不应该还有 pending approval
    pending = await approval_manager.list_pending()

    assert pending == []

    # ---------------------------------------------------------
    # 3. Resume
    # ---------------------------------------------------------

    run = await manager.resume_run(
        run_id=run.run_id,
        value={
            "approval_id": approval_id,
            "approved": True,
        },
    )

    # ---------------------------------------------------------
    # 4. 最终 Run 完成
    # ---------------------------------------------------------

    assert run.status == RunStatus.COMPLETED

    assert run.completed_at is not None

    # Approval 完成后，Run 不再保留 approval_id
    assert run.approval_id is None

    # ---------------------------------------------------------
    # 5. 验证真实 Tool 已执行
    # ---------------------------------------------------------

    assert execution_count == 2

    assert run.result == {
        "order_id": "1001",
        "quantity": 80,
        "updated": True,
    }

    # ---------------------------------------------------------
    # 6. 验证审批记录最终状态
    # ---------------------------------------------------------

    final_approval = await approval_manager.get_request(
        approval_id,
    )

    assert final_approval.approval_id == approval_id
    assert final_approval.status.value == "APPROVED"
    assert final_approval.comment == "approved by integration test"
    assert final_approval.resolved_at is not None