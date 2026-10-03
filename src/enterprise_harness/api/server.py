from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry, AgentStatus
from enterprise_harness.gateway.exceptions import ApprovalRequiredError
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.observability import TraceManager
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import Run, RunStatus


app = FastAPI(
    title="Enterprise Agent Harness",
    version="1.0.0",
    description=(
        "Enterprise Agent Harness API - Agent Runtime Platform "
        "with Tool Gateway, Policy Engine, HITL and Trace"
    ),
)


_registry: AgentRegistry | None = None
_run_manager: RunManager | None = None
_trace_manager: TraceManager | None = None
_approval_manager: ApprovalManager | None = None
_tool_gateway: ToolGateway | None = None


def init_api(
    registry: AgentRegistry,
    run_manager: RunManager,
    trace_manager: TraceManager | None = None,
    approval_manager: ApprovalManager | None = None,
    tool_gateway: ToolGateway | None = None,
) -> None:
    global _registry, _run_manager, _trace_manager, _approval_manager, _tool_gateway
    _registry = registry
    _run_manager = run_manager
    _trace_manager = trace_manager or run_manager.trace_manager
    _approval_manager = approval_manager or ApprovalManager()
    _tool_gateway = tool_gateway


def get_tool_gateway() -> ToolGateway:
    if _tool_gateway is None:
        raise RuntimeError("ToolGateway not initialized. Call init_api(tool_gateway=...) first.")
    return _tool_gateway


def get_registry() -> AgentRegistry:
    if _registry is None:
        raise RuntimeError("API not initialized. Call init_api() first.")
    return _registry


def get_run_manager() -> RunManager:
    if _run_manager is None:
        raise RuntimeError("API not initialized. Call init_api() first.")
    return _run_manager


def get_trace_manager() -> TraceManager:
    if _trace_manager is None:
        raise RuntimeError("API not initialized. Call init_api() first.")
    return _trace_manager


def get_approval_manager() -> ApprovalManager:
    if _approval_manager is None:
        raise RuntimeError("API not initialized. Call init_api() first.")
    return _approval_manager


def _route_task(
    message: str,
    gateway: ToolGateway,
    agent_tools: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """
    基于 ToolGateway.registry 判断消息能不能直接映射到一个 Tool。

    返回:
        ("tool", {"tool_name": ..., "arguments": ...})
        ("agent", None)
    """
    available_tools = [t for t in gateway.registry.list_tools() if t.name in agent_tools]

    msg_lower = message.lower()
    matches: list[tuple[int, str, dict[str, Any]]] = []

    for tool in available_tools:
        name_parts = tool.name.replace("secure_", "").replace("_", " ")
        keywords = {name_parts, tool.name}
        for word in re.split(r"[\s,、，。.!！?？;；:：\"']+", tool.description.lower()):
            if len(word) >= 2:
                keywords.add(word)
        keywords.add(tool.name)

        matched = any(kw in msg_lower for kw in keywords if kw.strip())
        if not matched:
            continue

        args: dict[str, Any] = {}
        schema = tool.input_schema or {}
        properties = schema.get("properties", {})

        for prop_name, prop_schema in properties.items():
            desc = prop_schema.get("description", "").lower()
            prop_type = prop_schema.get("type", "string")

            if prop_type == "string":
                if "path" in desc or "file" in desc or "文件" in desc or "路径" in desc:
                    for pat in [
                        r"([A-Za-z]:[\\/][^\s,，。;；]+(?:\\[^\s,，。;；]*)*\.[A-Za-z0-9]+)",
                        r"(D:/[^\s,，。;；]+)",
                        r"(C:/[^\s,，。;；]+)",
                        r"(/[^\s,，。;；]+(?:/[^\s,，。;；]*)*)",
                    ]:
                        m = re.search(pat, message)
                        if m:
                            args[prop_name] = m.group(1)
                            break
                elif "content" in desc or "内容" in desc:
                    m = re.search(r"写[入作成]?[：:「\"']?([^\"'」，。，；;\s]+)[\"'」]?", message)
                    if not m:
                        m = re.search(r"['\"]([^'\"]+)['\"]", message)
                    if m:
                        args[prop_name] = m.group(1)
                elif prop_name not in args:
                    if "id" in prop_name.lower() or "code" in prop_name.lower():
                        m = re.search(r'([A-Z]{2,}[-\s]*\d+)', message)
                        if m:
                            args[prop_name] = m.group(1).strip().replace(" ", "-")
                    elif "status" in prop_name.lower() or "state" in prop_name.lower():
                        m = re.search(r'(?:status|state)\s+(?:to\s+)?[\"\']?(\w+)[\"\']?', message, re.IGNORECASE)
                        if m:
                            args[prop_name] = m.group(1).upper()

        required = schema.get("required", [])
        if all(r in args for r in required):
            score = 0
            if name_parts in msg_lower:
                score += 100
            matching_kw = sum(1 for kw in keywords if kw.strip() and kw in msg_lower)
            score += matching_kw
            matches.append((score, tool.name, args))

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        _, best_name, best_args = matches[0]
        return ("tool", {"tool_name": best_name, "arguments": best_args})

    return ("agent", None)


class ChatRequest(BaseModel):
    agent_id: str | None = None
    message: str
    principal_id: str = "api-user"
    principal_role: str = "editor"
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    status: str
    reply: Any | None = None
    run_id: str | None = None
    approval_id: str | None = None
    run_status: str | None = None
    routed_to: str | None = None


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    registry = get_registry()
    gateway = get_tool_gateway()
    manager = get_run_manager()

    from enterprise_harness.policy.rbac import Principal

    principal = Principal(
        principal_id=req.principal_id,
        role=req.principal_role,
    )

    agent_id = req.agent_id
    if agent_id is None:
        try:
            agent_id = registry.list_agents()[0].agent_id
        except (IndexError, KeyError):
            raise HTTPException(status_code=400, detail="No agent_id provided and no agent registered")

    agent_tools = registry.get_config(agent_id).tools

    route, payload = _route_task(req.message, gateway, agent_tools)

    if route == "tool" and payload is not None:
        try:
            result = await gateway.execute(
                tool_name=payload["tool_name"],
                arguments=payload["arguments"],
                principal=principal,
            )
            return ChatResponse(
                status="ok",
                reply=result,
                routed_to=f"tool:{payload['tool_name']}",
            )
        except ApprovalRequiredError as exc:
            return ChatResponse(
                status="WAITING_APPROVAL",
                approval_id=exc.approval_id,
                reply={
                    "tool_name": exc.tool_name,
                    "reason": "Tool execution requires HITL approval",
                    "message": str(exc),
                },
                routed_to=f"tool:{payload['tool_name']}",
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    from enterprise_harness.application import RuntimeFactory

    factory = RuntimeFactory(registry=registry, gateway=gateway)
    runtime = await factory.create(agent_id=agent_id)

    run = await manager.create_run(
        agent_id=agent_id,
        task=req.message,
        context=req.context,
    )

    run = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
        principal=principal,
    )

    if run.status.value == "WAITING_APPROVAL":
        return ChatResponse(
            status="WAITING_APPROVAL",
            run_id=run.run_id,
            approval_id=run.approval_id,
            run_status=run.status.value,
            routed_to="deepagent",
        )

    return ChatResponse(
        status=run.status.value.lower(),
        reply=run.result,
        run_id=run.run_id,
        run_status=run.status.value,
        routed_to="deepagent",
    )


class ApprovalHandleRequest(BaseModel):
    action: str = "approve"
    comment: str | None = None


@app.post("/v1/approvals/{approval_id}")
async def handle_approval(approval_id: str, req: ApprovalHandleRequest | None = None):
    manager = get_run_manager()
    approval_manager = get_approval_manager()

    action = (req.action.lower().strip() if req else "approve")

    run = None
    for r in manager.runs.values():
        if r.approval_id == approval_id:
            run = r
            break

    if action in ("approve", "approve_it", "pass", "ok", "y", "yes", "1"):
        await approval_manager.approve(approval_id, comment=(req.comment if req else None))
        status = "APPROVED"
    elif action in ("reject", "deny", "refuse", "n", "no", "0"):
        await approval_manager.reject(approval_id, comment=(req.comment if req else None))
        status = "REJECTED"
        if run is not None:
            return {
                "status": "ok",
                "action": "rejected",
                "approval_id": approval_id,
                "run_id": run.run_id,
                "run_status": run.status.value,
                "message": "Run rejected by human",
            }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action '{action}'. Use 'approve' or 'reject'.")

    if run is not None and run.status.value == "WAITING_APPROVAL":
        try:
            run = await manager.resume_run(
                run.run_id,
                value={"approval_id": approval_id, "approved": True},
            )
            return {
                "status": "ok",
                "action": "approved",
                "approval_id": approval_id,
                "run_id": run.run_id,
                "run_status": run.status.value,
                "result": run.result,
            }
        except Exception as exc:
            return {
                "status": "ok",
                "action": "approved",
                "approval_id": approval_id,
                "run_id": run.run_id,
                "run_status": "RESUME_FAILED",
                "error": str(exc),
            }

    return {
        "status": "ok",
        "action": "approved",
        "approval_id": approval_id,
        "run_status": status,
    }


class CreateAgentRequest(BaseModel):
    agent_id: str
    name: str
    model: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    middleware: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    owner: str | None = None


class CreateRunRequest(BaseModel):
    agent_id: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    run_id: str
    agent_id: str
    agent_version: str = "1.0.0"
    task: str
    trace_id: str = ""
    status: str
    result: Any | None = None
    error: str | None = None
    checkpoint_id: str | None = None
    approval_id: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class ApprovalDecisionRequest(BaseModel):
    comment: str | None = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "enterprise-agent-harness",
        "version": "1.0.0",
    }


@app.post("/v1/agents")
async def create_agent(req: CreateAgentRequest):
    registry = get_registry()
    config = AgentConfig(
        agent_id=req.agent_id,
        name=req.name,
        version=req.version,
        model=req.model,
        system_prompt=req.system_prompt,
        tools=req.tools,
        skills=req.skills,
        middleware=req.middleware,
    )
    try:
        profile = registry.register(
            config,
            description=req.description,
            owner=req.owner,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return profile.to_dict()


@app.get("/v1/agents")
async def list_agents():
    registry = get_registry()
    return [p.to_dict() for p in registry.list_agents()]


@app.get("/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    registry = get_registry()
    try:
        profile = registry.get_profile(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return profile.to_dict()


@app.get("/v1/agents/{agent_id}/config")
async def get_agent_config(agent_id: str, version: str | None = None):
    registry = get_registry()
    try:
        config = registry.get_config(agent_id, version=version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return config.model_dump()


@app.post("/v1/runs")
async def create_run(req: CreateRunRequest):
    manager = get_run_manager()
    registry = get_registry()

    try:
        registry.get_profile(req.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run = await manager.create_run(
        agent_id=req.agent_id,
        task=req.task,
        context=req.context,
    )
    return run_response(run)


class StartRunRequest(BaseModel):
    principal_id: str = "api-user"
    principal_role: str = "admin"


@app.post("/v1/runs/{run_id}/start")
async def start_run(run_id: str, req: StartRunRequest | None = None):
    manager = get_run_manager()
    registry = get_registry()
    gateway = _tool_gateway

    try:
        run = await manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from enterprise_harness.application import RuntimeFactory
    from enterprise_harness.policy.rbac import Principal

    factory = RuntimeFactory(registry=registry, gateway=gateway)
    runtime = await factory.create(agent_id=run.agent_id)

    principal = Principal(
        principal_id=(req.principal_id if req else "api-user"),
        role=(req.principal_role if req else "admin"),
    )

    run = await manager.start_run(
        run_id=run_id,
        runtime=runtime,
        principal=principal,
    )
    return run_response(run)


@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str):
    manager = get_run_manager()
    try:
        run = await manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run_response(run)


@app.post("/v1/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    manager = get_run_manager()
    try:
        run = await manager.cancel_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return run_response(run)


@app.post("/v1/runs/{run_id}/resume")
async def resume_run(run_id: str, value: dict[str, Any] | None = None):
    manager = get_run_manager()
    try:
        run = await manager.resume_run(run_id, value=value or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return run_response(run)


@app.post("/v1/runs/{run_id}/approve")
async def approve_run(run_id: str, req: ApprovalDecisionRequest):
    manager = get_run_manager()
    approval_manager = get_approval_manager()
    try:
        run = await manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if run.approval_id is not None:
        await approval_manager.approve(
            approval_id=run.approval_id,
            comment=req.comment,
        )

    run = await manager.resume_run(
        run_id,
        value={
            "approval_id": run.approval_id,
            "approved": True,
            "comment": req.comment,
        },
    )
    return run_response(run)


@app.post("/v1/runs/{run_id}/reject")
async def reject_run(run_id: str, req: ApprovalDecisionRequest):
    manager = get_run_manager()
    approval_manager = get_approval_manager()
    try:
        run = await manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if run.approval_id is not None:
        await approval_manager.reject(
            approval_id=run.approval_id,
            comment=req.comment,
        )

    run.status = RunStatus.FAILED
    run.error = f"Run rejected by human: {req.comment or 'no comment'}"
    run.completed_at = datetime.now(timezone.utc)

    return run_response(run)


@app.get("/v1/runs/{run_id}/trace")
async def get_run_trace(run_id: str):
    trace_manager = get_trace_manager()
    try:
        spans = trace_manager.get_run_spans(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "run_id": run_id,
        "spans": [s.to_dict() for s in spans],
    }


def run_response(run: Run) -> RunResponse:
    return RunResponse(
        run_id=run.run_id,
        agent_id=run.agent_id,
        agent_version=run.agent_version,
        task=run.task,
        trace_id=run.trace_id,
        status=run.status.value,
        result=run.result,
        error=run.error,
        checkpoint_id=run.checkpoint_id,
        approval_id=run.approval_id,
        created_at=run.created_at.isoformat(),
        started_at=(
            run.started_at.isoformat()
            if run.started_at is not None
            else None
        ),
        completed_at=(
            run.completed_at.isoformat()
            if run.completed_at is not None
            else None
        ),
    )


@app.get("/v1/runs")
async def list_runs(tenant_id: str | None = None):
    manager = get_run_manager()
    runs = await manager.list_runs(tenant_id=tenant_id)
    return [run_response(r) for r in runs]


@app.get("/v1/runs/{run_id}/observability")
async def get_run_observability(run_id: str):
    manager = get_run_manager()

    try:
        await manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from enterprise_harness.observability.manager import ObservabilityManager

    obs = ObservabilityManager(
        trace_manager=get_trace_manager(),
    )
    snapshot = obs.snapshot(run_id=run_id)
    return {
        "run_id": run_id,
        "traces": snapshot.traces,
        "metrics": snapshot.metrics,
    }


@app.post("/v1/agents/{agent_id}/enable")
async def enable_agent(agent_id: str):
    registry = get_registry()
    try:
        registry.activate(agent_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "agent_id": agent_id, "agent_status": "ACTIVE"}


@app.post("/v1/agents/{agent_id}/disable")
async def disable_agent(agent_id: str):
    registry = get_registry()
    try:
        registry.deactivate(agent_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "agent_id": agent_id, "agent_status": "INACTIVE"}


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    skills: list[str] | None = None
    middleware: list[str] | None = None
    description: str | None = None
    owner: str | None = None


@app.put("/v1/agents/{agent_id}")
async def update_agent(agent_id: str, req: UpdateAgentRequest):
    registry = get_registry()
    try:
        config = registry.get_config(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    updates = req.model_dump(exclude_none=True)

    description = updates.pop("description", None)
    owner = updates.pop("owner", None)

    if updates:
        for field, value in updates.items():
            if hasattr(config, field):
                setattr(config, field, value)

        try:
            registry.update_config(config)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = registry.get_profile(agent_id)
    return profile.to_dict()


class ExecuteToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    principal_id: str = "api-user"
    principal_role: str = "admin"
    run_id: str | None = None
    approval_id: str | None = None


@app.post("/v1/tools/execute")
async def execute_tool(req: ExecuteToolRequest):
    gateway = get_tool_gateway()

    from enterprise_harness.policy.rbac import Principal

    principal = Principal(
        principal_id=req.principal_id,
        role=req.principal_role,
    )

    try:
        result = await gateway.execute(
            tool_name=req.tool_name,
            arguments=req.arguments,
            principal=principal,
            run_id=req.run_id,
            approval_id=req.approval_id,
        )
        return {"status": "ok", "result": result}
    except ApprovalRequiredError as exc:
        return {
            "status": "REQUIRE_APPROVAL",
            "approval_id": exc.approval_id,
            "tool_name": exc.tool_name,
            "run_id": exc.run_id,
            "message": str(exc),
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/tools")
async def list_tools():
    gateway = get_tool_gateway()
    tools = gateway.registry.list_tools()
    return {
        "total": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "risk_level": t.risk_level,
                "permissions": list(t.permissions),
                "input_schema": t.input_schema,
            }
            for t in tools
        ],
    }


class ApprovalActionRequest(BaseModel):
    approval_id: str
    comment: str | None = None


@app.post("/v1/tools/approval/approve")
async def approve_tool(req: ApprovalActionRequest):
    gateway = get_tool_gateway()
    await gateway.approval_manager.approve(req.approval_id, comment=req.comment)
    approval = await gateway.approval_manager.get_request(req.approval_id)
    return {
        "status": "ok",
        "approval_id": req.approval_id,
        "approval_status": approval.status.value,
    }


@app.post("/v1/tools/approval/reject")
async def reject_tool(req: ApprovalActionRequest):
    gateway = get_tool_gateway()
    await gateway.approval_manager.reject(req.approval_id, comment=req.comment)
    approval = await gateway.approval_manager.get_request(req.approval_id)
    return {
        "status": "ok",
        "approval_id": req.approval_id,
        "approval_status": approval.status.value,
    }
