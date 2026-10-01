"""验证 Chat 路由 + Chat API 端到端（不需要 LLM）"""
import os, sys, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from enterprise_harness.api.server import _route_task
from enterprise_harness.gateway.registry import ToolDefinition, ToolRegistry
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.observability.audit import AuditLogger
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.rbac import RBAC, Role


def build_gateway():
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="secure_read_file", description="读取文件（低风险）",
        input_schema={"type":"object","properties":{"path":{"type":"string","description":"文件路径"}},"required":["path"]},
        risk_level="LOW", permissions=["file:read"],
        handler=lambda path: {"content": open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "not_found"},
    ))
    registry.register(ToolDefinition(
        name="secure_write_file", description="写入文件（高风险，需审批）",
        input_schema={"type":"object","properties":{"path":{"type":"string","description":"文件路径"},"content":{"type":"string","description":"文件内容"}},"required":["path","content"]},
        risk_level="HIGH", permissions=["file:write"],
        handler=lambda path, content: open(path, "w", encoding="utf-8").write(content) or {"bytes": len(content)},
    ))
    return ToolGateway(
        registry=registry, router=ToolRouter(registry),
        validator=ToolValidator(), executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=RBAC(roles=[
            Role(name="viewer", permissions=frozenset({"file:read"})),
            Role(name="editor", permissions=frozenset({"file:read","file:write"})),
        ])),
        trace_manager=TraceManager(), audit_logger=AuditLogger(),
        metric_collector=MetricCollector(),
    )


gw = build_gateway()
agent_tools = ["secure_read_file", "secure_write_file"]

print("=== 路由决策测试 ===")
tests = [
    ("把 D:/tmp/hello.txt 改成 0000", "tool:secure_write_file"),
    ("读取 D:/tmp/hello.txt 的内容", "tool:secure_read_file"),
    ("帮我做一个季度销售分析报告", "agent"),
    ("今天天气怎么样", "agent"),
]
for msg, expected in tests:
    route, payload = _route_task(msg, gw, agent_tools)
    label = f"{route}:{payload['tool_name']}" if payload else route
    ok = "✅" if (f"{route}:{payload['tool_name']}" if payload else route) == expected else "❌"
    print(f"  {ok} '{msg[:30]}...' → {label} (expected {expected})")
    if payload:
        print(f"       args={payload['arguments']}")

print()
print("=== Chat 端到端（无 LLM 版，直接调 gateway）===")
print("POST /v1/chat  {'message': '把 D:/tmp/hello.txt 改成 0000'}")

# 真实调 gateway 验证
async def demo():
    from enterprise_harness.policy.rbac import Principal
    principal = Principal(principal_id="u1", role="editor")

    # 尝试调用 → 应该触发审批
    try:
        result = await gw.execute(
            tool_name="secure_write_file",
            arguments={"path": "D:/tmp/hello.txt", "content": "0000"},
            principal=principal,
        )
        print(f"  直接成功: {result}")
    except Exception as exc:
        print(f"  触发: {exc}")
        aid = getattr(exc, 'approval_id', None)
        if aid:
            print(f"  approval_id={aid}")
            await gw.approval_manager.approve(aid, comment="ok")
            result2 = await gw.execute(
                tool_name="secure_write_file",
                arguments={"path": "D:/tmp/hello.txt", "content": "0000"},
                principal=principal,
                approval_id=aid,
            )
            print(f"  批准后重试成功: {result2}")

asyncio.run(demo())
