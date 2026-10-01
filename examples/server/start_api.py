"""
Enterprise Agent Harness - API Server 完整启动脚本

三层融合：
  1. Tool Gateway: write_file (HIGH risk → HITL 审批) + read_file (LOW)
  2. DeepAgents FilesystemMiddleware: Agent 内部辅助 (read_file, glob, ls)
  3. Agent Runtime: create_deep_agent(middleware=..., tools=[gateway tools...])

启动后可以：
  直接调 Tool Gateway:  POST /v1/tools/execute
  调完整 Agent Runtime:  POST /v1/runs → POST /v1/runs/{id}/start
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import uvicorn

from enterprise_harness.agent.config import AgentConfig
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


def _secure_write_file(path: str, content: str) -> dict:
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "written", "path": abs_path, "bytes": len(content.encode("utf-8"))}


def _secure_read_file(path: str) -> dict:
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
            name="secure_read_file",
            description="读取文件（受 Harness 治理，LOW risk）",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件绝对或相对路径"}},
                "required": ["path"],
            },
            risk_level="LOW",
            permissions=["file:read"],
            handler=_secure_read_file,
        ),
        ToolDefinition(
            name="secure_write_file",
            description="写入/覆盖文件（受 Harness 治理，HIGH risk → 触发人工审批）",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对或相对路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
            risk_level="HIGH",
            permissions=["file:write"],
            handler=_secure_write_file,
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


def build_agents() -> AgentRegistry:
    registry = AgentRegistry()

    cfg = AgentConfig(
        agent_id="file-agent",
        name="File Agent",
        model="gpt-4o-mini",
        system_prompt=(
            "You are a file management assistant.\n"
            "Use secure_read_file to read files.\n"
            "Use secure_write_file to write files (this requires approval).\n"
            "You also have filesystem tools (read_file, glob, ls) for auxiliary operations."
        ),
        tools=["secure_read_file", "secure_write_file"],
        middleware=["filesystem"],
        filesystem_permissions=["read_file", "glob", "ls"],
        interrupt_on={"tool_call": True},
        use_checkpointer=True,
    )

    registry.register(cfg)
    return registry


def main():
    gateway = build_gateway()
    registry = build_agents()
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

    print("=" * 60)
    print("  Enterprise Agent Harness - Complete API Server")
    print("=" * 60)
    print()
    print("  [Tool Gateway] Tools registered:")
    for t in gateway.registry.list_tools():
        print(f"    • {t.name:<25} risk={t.risk_level:<5} perms={list(t.permissions)}")
    print()
    print("  [DeepAgents] Agent registered:")
    for aid in registry.list_agents():
        cfg = registry.get_config(aid)
        print(f"    • {cfg.agent_id}  model={cfg.model}")
        print(f"       tools={cfg.tools}")
        print(f"       middleware={cfg.middleware}  fs_perms={cfg.filesystem_permissions}")
        print(f"       interrupt_on={cfg.interrupt_on}  checkpointer={cfg.use_checkpointer}")
    print()
    print("  [RBAC] roles: viewer(read) / editor(read+write) / admin(all)")
    print()
    print("  ==== 3 种调用方式 ====")
    print()
    print("  A) 直接调 Tool Gateway (跳过 Agent)")
    print("     POST /v1/tools/execute")
    print("     POST /v1/tools/approval/approve")
    print()
    print("  B) 完整 Agent Runtime (LLM + DeepAgents Loop + HITL)")
    print("     POST /v1/runs")
    print("     POST /v1/runs/{id}/start")
    print("     POST /v1/runs/{id}/approve")
    print("     POST /v1/runs/{id}/resume")
    print()
    print("  C) Swagger UI (最方便)")
    print("     http://127.0.0.1:8000/docs")
    print()
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1:8000", log_level="info")


if __name__ == "__main__":
    main()
