from __future__ import annotations

from typing import Any
from .exceptions import (
    ApprovalRequiredError,
    ToolApprovalRejectedError,
    ToolExecutionError,
    ToolPolicyError,
    classify_error,
)

from enterprise_harness.observability import (
    AuditLogger,
    MetricCollector,
    SpanStatus,
    SpanType,
    TraceManager,
)
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.models import PolicyDecision
from enterprise_harness.policy.rbac import Principal

from .executor import ToolExecutor
from .registry import ToolRegistry
from .result_validator import DefaultResultValidator, ResultValidator
from .router import ToolRouter
from .validator import ToolValidator
import time
from enterprise_harness.policy.approval_manager import (
    ApprovalManager,
)


class ToolGateway:

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        router: ToolRouter,
        validator: ToolValidator,
        executor: ToolExecutor,
        result_validator: ResultValidator | None = None,
        policy_engine: PolicyEngine | None = None,
        approval_manager: ApprovalManager | None = None,
        trace_manager: TraceManager | None = None,
        audit_logger: AuditLogger | None = None,
        metric_collector: MetricCollector | None = None,
    ):
        self.registry = registry
        self.router = router
        self.validator = validator
        self.executor = executor

        if metric_collector is not None and hasattr(self.executor, "_metric_collector"):
            self.executor._metric_collector = metric_collector

        self.result_validator = result_validator or DefaultResultValidator()

        self.policy_engine = (
            policy_engine
            or PolicyEngine()
        )
        self.approval_manager = (
            approval_manager
            or ApprovalManager()
        )

        self.trace_manager = (
            trace_manager
            or TraceManager()
        )
        self.audit_logger = audit_logger
        self.metric_collector = metric_collector

    def _record_audit(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        decision: PolicyDecision,
        result: Any = None,
        principal: Principal | None = None,
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
        approver: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if self.audit_logger is None:
            return

        context = context or {}

        self.audit_logger.record(
            event="tool_execution",
            user_id=(
                principal.principal_id
                if principal is not None
                else "anonymous"
            ),
            agent_id=str(
                context.get("agent_id", "unknown")
            ),
            run_id=run_id or "unknown",
            tool=tool_name,
            tenant_id=tenant_id or "default",
            arguments=arguments,
            decision=decision.value,
            approver=approver,
            result=result,
        )

    async def execute(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
            context: dict[str, Any] | None = None,
            principal: Principal | None = None,
            *,
            run_id: str | None = None,
            parent_span_id: str | None = None,
            approval_id: str | None = None,
            tenant_id: str | None = None,
    ) -> Any:
        tenant_id = tenant_id or "default"

        arguments = arguments or {}
        start_time = time.perf_counter()
        span = None

        if run_id is None and approval_id is not None:
            try:
                existing = await self.approval_manager.get_request(approval_id)
                run_id = existing.run_id
            except (KeyError, AttributeError):
                pass

        if run_id is None:
            import uuid as _uuid
            run_id = f"api-{_uuid.uuid4().hex[:12]}"

        if self.trace_manager is not None:
            span = self.trace_manager.start_span(
                run_id=run_id,
                span_type=SpanType.TOOL,
                name=tool_name,
                parent_span_id=parent_span_id,
                tenant_id=tenant_id,
                input=arguments,
                metadata={
                    "principal_id": (
                        principal.principal_id
                        if principal
                        else None
                    )
                },
            )

        try:

            # 0. Tenant Isolation Check
            if principal is not None and tenant_id is not None:
                if principal.tenant_id != "default" and principal.tenant_id != tenant_id:
                    raise ToolPolicyError(
                        f"Tenant mismatch: principal tenant {principal.tenant_id} "
                        f"!= run tenant {tenant_id}"
                    )

            # 1. Tool Routing
            tool = self.router.route(tool_name)

            # 2. Input Validation
            validated_arguments = self.validator.validate(
                tool,
                arguments,
            )

            # 3. Policy Check
            decision = self.policy_engine.check(
                tool=tool,
                arguments=validated_arguments,
                context=context,
                principal=principal,
            )

            # 4. Policy Deny
            if decision == PolicyDecision.DENY:
                self._record_audit(
                    tool_name=tool.name,
                    arguments=validated_arguments,
                    decision=decision,
                    result={
                        "success": False,
                        "error": "policy_denied",
                    },
                    principal=principal,
                    context=context,
                    run_id=run_id,
                    tenant_id=tenant_id,
                )

                if self.metric_collector is not None:
                    self.metric_collector.record_policy_denied()

                raise ToolPolicyError(tool.name)

            # 5. Approval
            if decision == PolicyDecision.REQUIRE_APPROVAL:

                if approval_id is not None:

                    if run_id is None:
                        raise ValueError(
                            "run_id is required when validating approval"
                        )

                    await self.approval_manager.validate_approval(
                        approval_id=approval_id,
                        run_id=run_id,
                        tool_name=tool.name,
                        arguments=validated_arguments,
                    )

                else:

                    requester_id = None

                    if principal is not None:
                        requester_id = principal.principal_id

                    if run_id is None:
                        raise ValueError(
                            "run_id is required when tool execution "
                            "requires approval"
                        )

                    approval_request = (
                        await self.approval_manager.create_request(
                            run_id=run_id,
                            tool_name=tool.name,
                            arguments=validated_arguments,
                            requester_id=requester_id,
                            tenant_id=tenant_id,
                        )
                    )

                    self._record_audit(
                        tool_name=tool.name,
                        arguments=validated_arguments,
                        decision=decision,
                        result={
                            "success": False,
                            "error": "approval_required",
                            "approval_id": (
                                approval_request.approval_id
                            ),
                        },
                        principal=principal,
                        context=context,
                        run_id=run_id,
                        tenant_id=tenant_id,
                    )

                    raise ApprovalRequiredError(
                        approval_id=approval_request.approval_id,
                        run_id=run_id,
                        tool_name=tool.name,
                        arguments=validated_arguments,
                    )

            # 6. Execute
            result = await self.executor.execute(
                tool,
                validated_arguments,
            )

            try:
                result = self.result_validator.validate(
                    tool_name=tool.name,
                    result=result,
                    output_schema=tool.output_schema or None,
                )
            except ValueError as val_exc:
                self._record_audit(
                    tool_name=tool.name,
                    arguments=validated_arguments,
                    decision=decision,
                    result={
                        "success": False,
                        "error": str(val_exc),
                        "error_type": "result_validation_failed",
                    },
                    principal=principal,
                    context=context,
                    run_id=run_id,
                    tenant_id=tenant_id,
                )

                if self.metric_collector is not None:
                    self.metric_collector.record_tool(success=False)

                if span is not None:
                    self.trace_manager.fail_span(
                        span.span_id,
                        val_exc,
                    )

                raise

            if self.metric_collector is not None:
                self.metric_collector.record_tool(success=True)

            self._record_audit(
                tool_name=tool.name,
                arguments=validated_arguments,
                decision=decision,
                result={
                    "success": True,
                    "output": result,
                },
                principal=principal,
                context=context,
                run_id=run_id,
                tenant_id=tenant_id,
            )

            # 7. Trace Success
            if span is not None:
                self.trace_manager.finish_span(
                    span.span_id,
                    output=result,
                    status=SpanStatus.SUCCESS,
                )

            return result

        except Exception as exc:
            if (
                    self.audit_logger is not None
                    and "tool" in locals()
                    and "validated_arguments" in locals()
                    and "decision" in locals()
                    and decision == PolicyDecision.ALLOW
            ):
                self._record_audit(
                    tool_name=tool.name,
                    arguments=validated_arguments,
                    decision=decision,
                    result={
                        "success": False,
                        "error": str(exc),
                        "error_type": classify_error(exc),
                    },
                    principal=principal,
                    context=context,
                    run_id=run_id,
                    tenant_id=tenant_id,
                )

            if (
                    self.metric_collector is not None
                    and "decision" in locals()
                    and decision == PolicyDecision.ALLOW
            ):
                self.metric_collector.record_tool(success=False)

            if span is not None:
                span.metadata["error_type"] = classify_error(exc)
                self.trace_manager.fail_span(
                    span.span_id,
                    exc,
                )

            raise
        finally:
            if self.metric_collector is not None:
                latency_ms = (
                                     time.perf_counter() - start_time
                             ) * 1000
                self.metric_collector.record_latency(
                    latency_ms
                )