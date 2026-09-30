"""
Enterprise Agent Harness - API Server 启动脚本

注册 write_file (HIGH risk, 需审批) 和其他工具，
启动 FastAPI server，Postman 直接调。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import uvicorn

from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.api.server import app, init_api
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.registry import ToolDefinition, ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.observability.audit import AuditLogger
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.orchestration.replanner import SimpleReplanner
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.rbac import RBAC, Role
from enterprise_harness.runtime.manager import RunManager


def _write_file(path: str, content: str) -> dict:
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "written", "path": abs_path, "bytes": len(content.encode("utf-8"))}


def _read_file(path: str) -> dict:
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = f.read()
        return {"status": "ok", "path": abs_path, "content": data}
    except FileNotFoundError:
        return {"status": "not_found", "path": abs_path}


def build_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="read_file",
            description="读取文件内容",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件绝对或相对路径"}},
                "required": ["path"],
            },
            risk_level="LOW",
            permissions=["file:read"],
            handler=_read_file,
        ),
        ToolDefinition(
            name="write_file",
            description="写入/覆盖文件（高风险，需审批）",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对或相对路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
            risk_level="HIGH",
            permissions=["file:write"],
            handler=_write_file,
        ),
    ]


def build_gateway() -> ToolGateway:
    tool_registry = ToolRegistry()
    for t in build_tools():
        tool_registry.register(t)

    rbac = RBAC(
        roles=[
            Role(name="viewer", permissions=frozenset({"file:read"})),
            Role(name="editor", permissions=frozenset({"file:read", "file:write"})),
            Role(name="admin", permissions=frozenset({"file:read", "file:write", "file:delete"})),
        ]
    )

    return ToolGateway(
        registry=tool_registry,
        router=ToolRouter(tool_registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=rbac),
        trace_manager=TraceManager(),
        audit_logger=AuditLogger(),
        metric_collector=MetricCollector(),
    )


def main():
    gateway = build_gateway()
    registry = AgentRegistry()
    run_mgr = RunManager(
        replanner=SimpleReplanner(),
        trace_manager=TraceManager(),
    )

    init_api(
        registry=registry,
        run_manager=run_mgr,
        trace_manager=gateway.trace_manager,
        tool_gateway=gateway,
    )

    print("=" * 56)
    print("  Enterprise Agent Harness API Server")
    print("=" * 56)
    print(f"  Tools registered: {len(gateway.registry.list_tools())}")
    for t in gateway.registry.list_tools():
        print(f"    • {t.name}  (risk={t.risk_level}, perms={list(t.permissions)})")
    print()
    print("  RBAC roles: viewer / editor / admin")
    print()
    print("  POST /v1/tools/execute          直接执行 Tool")
    print("  POST /v1/tools/approval/approve 审批通过")
    print("  POST /v1/tools/approval/reject  审批拒绝")
    print("  GET  /v1/tools                  查看所有注册的 Tool")
    print()
    print("  Swagger UI: http://127.0.0.1:8000/docs")
    print("=" * 56)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
