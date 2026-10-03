from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from enterprise_harness.policy.rbac import Principal
from enterprise_harness.context import ContextItem

class RunContext(BaseModel):
    """
    一次 Agent Run 的执行上下文。

    RunContext 是 Harness 内部各模块共享的运行时上下文，
    用于统一传递 Run、Agent、用户身份、业务上下文以及 Trace 信息。
    """

    run_id: str

    agent_id: str

    agent_version: str = "1.0.0"

    tenant_id: str = "default"

    task: str

    trace_id: str = ""

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    principal: Principal | None = None

    trace_root_span_id: str | None = None

    current_span_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    built_context: list[ContextItem] = Field(default_factory=list)