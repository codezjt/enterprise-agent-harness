from __future__ import annotations

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
    _approval_manager = approval_manager or run_manager.recovery_manager.replanner is not None
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
    task: str
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
        task=run.task,
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
