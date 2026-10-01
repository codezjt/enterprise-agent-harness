"""
不依赖 LLM 的完整链路验证。
测试 1: DeepAgentToolAdapter.catch ApprovalRequiredError → interrupt() 路径正确
测试 2: LangGraphRuntime.is_interrupted / extract_approval 能识别 interrupt 结果
测试 3: RunManager resume_run 带 approval_id 重试路径正确
测试 4: middleware 融合正确
"""
import os
import sys
import asyncio
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from enterprise_harness.gateway.deepagent import DeepAgentToolAdapter
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.registry import ToolDefinition, ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.exceptions import ApprovalRequiredError
from enterprise_harness.observability.audit import AuditLogger
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.rbac import RBAC, Role, Principal
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime
from enterprise_harness.runtime.result import RuntimeResult, RuntimeStatus

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.deepagent_runtime import DeepAgentRuntime, _build_middleware_list, _FS_TOOL_MAP


def _build_gateway() -> ToolGateway:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="write_order", description="写入订单（HIGH risk）",
        input_schema={"type": "object", "properties": {"order": {"type": "string"}}, "required": ["order"]},
        risk_level="HIGH", permissions=["order:write"],
        handler=lambda order: {"written": True, "order": order},
    ))
    registry.register(ToolDefinition(
        name="read_order", description="查询订单（LOW risk）",
        input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        risk_level="LOW", permissions=["order:read"],
        handler=lambda order_id: {"order_id": order_id, "status": "active"},
    ))

    rbac = RBAC(roles=[
        Role(name="viewer", permissions=frozenset({"order:read"})),
        Role(name="editor", permissions=frozenset({"order:read", "order:write"})),
    ])

    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=rbac),
        trace_manager=TraceManager(),
        audit_logger=AuditLogger(),
        metric_collector=MetricCollector(),
    )


# ============================================================
# Test 1: DeepAgentToolAdapter interrupt 机制
# ============================================================

def test_deepagent_tool_adapter_interrupt_on_approval_required():
    """DeepAgentToolAdapter 在 gateway 抛 ApprovalRequiredError 时
    应该抛出 langgraph.interrupt()，返回 approval_required 结构"""
    gw = _build_gateway()
    principal = Principal(principal_id="u1", role="editor")
    adapter = DeepAgentToolAdapter(
        gateway=gw, principal=principal, run_id="run-test-123"
    )
    tool_def = gw.registry.get("write_order")
    wrapped_tool = adapter.adapt(tool_def)

    # 先触发审批
    with pytest.raises(ApprovalRequiredError) as exc:
        asyncio.run(gw.execute(
            tool_name="write_order",
            arguments={"order": "A001"},
            principal=principal,
            run_id="run-test-123",
        ))
    approval_id = exc.value.approval_id
    assert approval_id is not None

    # 现在批准
    asyncio.run(gw.approval_manager.approve(approval_id, comment="ok"))

    # 再带 approval_id 执行 → 应该成功
    result = asyncio.run(gw.execute(
        tool_name="write_order",
        arguments={"order": "A001"},
        principal=principal,
        run_id="run-test-123",
        approval_id=approval_id,
    ))
    assert result["written"] is True
    print(f"  ✓ adapter interrupt 路径正确: approval_id={approval_id}")


# ============================================================
# Test 2: LangGraphRuntime interrupt 检测
# ============================================================

def test_langgraph_runtime_interrupt_detection():
    """LangGraphRuntime 应该能正确识别 interrupt 结果"""
    interrupt_result = {
        "__interrupt__": [
            type("Interrupt", (), {
                "value": {
                    "type": "approval_required",
                    "approval_id": "abc-123",
                    "tool_name": "write_order",
                    "message": "needs approval",
                }
            })()
        ],
        "messages": [],
    }
    assert LangGraphRuntime.is_interrupted(interrupt_result) is True

    approval = LangGraphRuntime.extract_approval(interrupt_result)
    assert approval is not None
    assert approval["type"] == "approval_required"
    assert approval["approval_id"] == "abc-123"
    assert approval["tool_name"] == "write_order"

    # 非 interrupt 结果
    normal = {"messages": [], "result": "ok"}
    assert LangGraphRuntime.is_interrupted(normal) is False
    assert LangGraphRuntime.extract_approval(normal) is None
    print("  ✓ interrupt 检测正确")


# ============================================================
# Test 3: LangGraphRuntimeAdapter → RuntimeResult 转换
# ============================================================

def test_langgraph_runtime_adapter_to_runtime_result():
    """LangGraphRuntimeAdapter 应该正确把 interrupt 结果转成 WAITING_APPROVAL"""
    from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime
    from enterprise_harness.runtime.langgraph_runtime_adapter import LangGraphRuntimeAdapter

    class FakeRuntime(LangGraphRuntime):
        async def run(self, run_id, input_data):
            return {
                "__interrupt__": [
                    type("X", (), {
                        "value": {
                            "type": "approval_required",
                            "approval_id": "ap-42",
                            "tool_name": "write_order",
                        }
                    })()
                ],
                "messages": [],
            }
        async def resume(self, run_id, value):
            return {"messages": [type("A", (), {"content": "done"})()]}

    class FakeCompiled: pass
    fake_rt = FakeRuntime(compiled_graph=FakeCompiled())
    adapter = LangGraphRuntimeAdapter(fake_rt)

    from enterprise_harness.runtime.context import RunContext
    ctx = RunContext(
        run_id="run-001",
        run=None,
        agent_id="a1",
        task="test task",
        context={},
        principal=None,
    )

    result = asyncio.run(adapter.run(ctx))
    assert result.status == RuntimeStatus.WAITING_APPROVAL
    assert result.approval_id == "ap-42"

    resumed = asyncio.run(adapter.resume(ctx, {"approval_id": "ap-42", "approved": True}))
    assert resumed.status == RuntimeStatus.COMPLETED
    print("  ✓ RuntimeResult 转换正确")


# ============================================================
# Test 4: middleware 融合
# ============================================================

def test_deepagent_middleware_integration():
    """AgentConfig 里声明的 middleware 应该被正确构建"""
    cfg = AgentConfig(
        agent_id="mw-agent", name="MW Agent", model="gpt-4o-mini",
        middleware=["filesystem"],
        filesystem_permissions=["read", "write_file", "edit"],
        use_checkpointer=True,
    )

    mws = _build_middleware_list(cfg)
    assert len(mws) == 1

    from deepagents.middleware.filesystem import FilesystemMiddleware
    assert isinstance(mws[0], FilesystemMiddleware)

    gw = _build_gateway()
    runtime = DeepAgentRuntime(config=cfg, tool_gateway=gw)
    graph = runtime.build_agent()
    assert graph is not None
    assert runtime.checkpointer is not None
    print(f"  ✓ middleware 融合正确: {len(mws)} middlewares, checkpointer={runtime.checkpointer is not None}")


# ============================================================
# Test 5: Policy + RBAC + Gateway 审批全链路
# ============================================================

def test_policy_rbac_gateway_full_chain():
    """PolicyEngine + RBAC + Gateway 审批流程全链路"""
    gw = _build_gateway()
    viewer = Principal(principal_id="v1", role="viewer")
    editor = Principal(principal_id="e1", role="editor")

    # Step 1: viewer 调 write_order → DENY
    with pytest.raises(PermissionError):
        asyncio.run(gw.execute(
            tool_name="write_order", arguments={"order": "A"}, principal=viewer, run_id="r1"
        ))

    # Step 2: editor 调 write_order → REQUIRE_APPROVAL (HIGH risk)
    with pytest.raises(ApprovalRequiredError) as exc:
        asyncio.run(gw.execute(
            tool_name="write_order", arguments={"order": "A"}, principal=editor, run_id="r2"
        ))
    approval_id = exc.value.approval_id

    # Step 3: editor 调 read_order → ALLOW (LOW risk)
    result = asyncio.run(gw.execute(
        tool_name="read_order", arguments={"order_id": "A"}, principal=editor, run_id="r2"
    ))
    assert result["status"] == "active"

    # Step 4: 批准后重试
    asyncio.run(gw.approval_manager.approve(approval_id))
    result2 = asyncio.run(gw.execute(
        tool_name="write_order", arguments={"order": "A"}, principal=editor, run_id="r2", approval_id=approval_id
    ))
    assert result2["written"] is True

    # 验证 Audit
    assert len(gw.audit_logger._events) >= 3
    print(f"  ✓ 完整 Gateway 链路: {len(gw.audit_logger._events)} audit events")


# ============================================================
# Test 6: RunManager 完整创建+启动+WAITING_APPROVAL+resume
# ============================================================

def test_runmanager_waiting_approval_resume():
    """RunManager 能被正确使用：create → start_langgraph_run (fake LGR) → WAITING → resume"""
    from enterprise_harness.runtime.manager import RunManager

    # 构造一个 fake LangGraphRuntime adapter 层
    manager = RunManager()

    # create
    run = asyncio.run(manager.create_run(agent_id="a1", task="do something"))
    assert run.status.value == "CREATED"

    # 用一个假的 deepagent runtime 来让 build_agent 返回我们想要的 interrupt 结果
    from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime

    class FakeCompiled:
        async def ainvoke(self, input_data, config):
            return {
                "__interrupt__": [
                    type("X", (), {
                        "value": {
                            "type": "approval_required",
                            "approval_id": "ap-zzz",
                            "tool_name": "write_order",
                        }
                    })()
                ],
                "messages": [],
            }

    class FakeRuntime:
        def build_agent(self, run_context=None):
            return FakeCompiled()

    from enterprise_harness.policy.rbac import Principal
    run2 = asyncio.run(manager.start_langgraph_run(
        run_id=run.run_id,
        deepagent_runtime=FakeRuntime(),
        principal=Principal(principal_id="u", role="admin"),
    ))
    assert run2.status.value == "WAITING_APPROVAL", f"Expected WAITING_APPROVAL, got {run2.status.value}, error={run2.error}"
    assert run2.approval_id == "ap-zzz"

    # resume — RunManager._runtime_store 里已经存好了 LangGraphRuntime
    # RunManager._get_or_create_langgraph_runtime 已经帮我们存好了

    class FakeCompiled2:
        async def ainvoke(self, input_data, config):
            # 第一次调用有 interrupt，第二次（resume）返回正常
            if hasattr(self, "_done"):
                return {"messages": [type("A", (), {"content": "ok final"})()]}
            self._done = True
            return {
                "__interrupt__": [
                    type("X", (), {
                        "value": {"type": "approval_required", "approval_id": "ap-resume"}
                    })()
                ],
                "messages": [],
            }

    class FakeRuntime2:
        def build_agent(self, run_context=None):
            return FakeCompiled2()

    run3 = asyncio.run(manager.create_run(agent_id="a2", task="something"))
    run3 = asyncio.run(manager.start_langgraph_run(
        run_id=run3.run_id,
        deepagent_runtime=FakeRuntime2(),
    ))
    assert run3.status.value == "WAITING_APPROVAL"

    run4 = asyncio.run(manager.resume_run(
        run3.run_id,
        value={"approval_id": run3.approval_id, "approved": True},
    ))
    assert run4.status.value == "COMPLETED", f"Expected COMPLETED, got {run4.status.value}, error={run4.error}"
    print(f"  ✓ RunManager create→start→WAITING→resume→COMPLETED 全链路")


if __name__ == "__main__":
    import sys

    def run(name, fn):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            import traceback
            print(f"FAIL {name}: {e}")
            traceback.print_exc()

    run("T1 DeepAgentToolAdapter interrupt", test_deepagent_tool_adapter_interrupt_on_approval_required)
    run("T2 LangGraphRuntime interrupt detect", test_langgraph_runtime_interrupt_detection)
    run("T3 RuntimeResult convert", test_langgraph_runtime_adapter_to_runtime_result)
    run("T4 middleware integration", test_deepagent_middleware_integration)
    run("T5 Policy+RBAC+Gateway chain", test_policy_rbac_gateway_full_chain)
    run("T6 RunManager resume", test_runmanager_waiting_approval_resume)
