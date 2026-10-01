import pytest

from enterprise_harness.gateway.executor import ToolExecutor, _is_retryable
from enterprise_harness.gateway.exceptions import ToolExecutionError, ToolTimeoutError
from enterprise_harness.gateway.models import ToolDefinition
from enterprise_harness.gateway.result_validator import (
    DefaultResultValidator,
    NoOpResultValidator,
    ResultValidator,
)


class TestResultValidator:

    def test_default_validator_raises_on_none(self):
        validator = DefaultResultValidator()
        with pytest.raises(ValueError, match="returned None"):
            validator.validate("my_tool", None)

    def test_default_validator_passes_non_none(self):
        validator = DefaultResultValidator()
        result = validator.validate("my_tool", {"key": "value"})
        assert result == {"key": "value"}

    def test_default_validator_with_output_schema_valid(self):
        validator = DefaultResultValidator()
        result = validator.validate(
            "echo",
            {"message": "hello", "count": 5},
            output_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["message", "count"],
            },
        )
        assert result["message"] == "hello"

    def test_default_validator_with_output_schema_invalid(self):
        validator = DefaultResultValidator()
        with pytest.raises(ValueError, match="result validation failed"):
            validator.validate(
                "echo",
                {"message": "hello"},
                output_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["message", "count"],
                },
            )

    def test_default_validator_passes_without_schema(self):
        validator = DefaultResultValidator()
        result = validator.validate("my_tool", 42)
        assert result == 42

    def test_noop_validator_passes_everything(self):
        validator = NoOpResultValidator()
        assert validator.validate("t", None) is None
        assert validator.validate("t", {}) == {}
        assert validator.validate("t", "bad", output_schema={"type": "integer"}) == "bad"

    def test_result_validator_is_abstract(self):
        assert issubclass(DefaultResultValidator, ResultValidator)
        assert issubclass(NoOpResultValidator, ResultValidator)

    @pytest.mark.asyncio
    async def test_gateway_pipeline_includes_result_validation(self):
        from enterprise_harness.gateway.gateway import ToolGateway
        from enterprise_harness.gateway.registry import ToolRegistry
        from enterprise_harness.gateway.router import ToolRouter
        from enterprise_harness.gateway.validator import ToolValidator
        from enterprise_harness.policy.rbac import Principal

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="greet",
                description="Greet someone",
                handler=lambda name: {"greeting": f"Hello, {name}"},
                output_schema={
                    "type": "object",
                    "properties": {"greeting": {"type": "string"}},
                    "required": ["greeting"],
                },
            )
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            result_validator=DefaultResultValidator(),
        )

        principal = Principal(principal_id="u1", role="admin")
        result = await gw.execute(
            tool_name="greet",
            arguments={"name": "World"},
            principal=principal,
        )
        assert result == {"greeting": "Hello, World"}

    @pytest.mark.asyncio
    async def test_result_validation_with_bad_output_rejects(self):
        from enterprise_harness.gateway.gateway import ToolGateway
        from enterprise_harness.gateway.registry import ToolRegistry
        from enterprise_harness.gateway.router import ToolRouter
        from enterprise_harness.gateway.validator import ToolValidator
        from enterprise_harness.policy.rbac import Principal

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="broken",
                description="Returns wrong type",
                handler=lambda: "not_a_dict",
                output_schema={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            )
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            result_validator=DefaultResultValidator(),
        )

        principal = Principal(principal_id="u1", role="admin")

        with pytest.raises(ValueError, match="result validation failed"):
            await gw.execute(
                tool_name="broken",
                arguments={},
                principal=principal,
            )


class TestToolRetry:

    def test_timeout_error_is_retryable(self):
        assert _is_retryable(ToolTimeoutError("t", 1.0)) is True

    def test_connection_error_is_retryable(self):
        assert _is_retryable(ConnectionError("refused")) is True
        assert _is_retryable(ConnectionResetError("reset")) is True
        assert _is_retryable(ConnectionRefusedError("refused")) is True
        assert _is_retryable(TimeoutError("timeout")) is True

    def test_tool_execution_error_not_retryable_by_default(self):
        exc = ToolExecutionError("t", "validation failed")
        assert _is_retryable(exc) is False

    def test_tool_execution_error_caused_by_timeout_is_retryable(self):
        cause = TimeoutError("timed out")
        exc = ToolExecutionError("t", "wrapped")
        exc.__cause__ = cause
        assert _is_retryable(exc) is True

    def test_tool_execution_error_caused_by_value_error_not_retryable(self):
        cause = ValueError("bad value")
        exc = ToolExecutionError("t", "wrapped")
        exc.__cause__ = cause
        assert _is_retryable(exc) is False

    @pytest.mark.asyncio
    async def test_executor_retries_on_timeout(self):
        call_count = [0]

        def flaky_handler():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("simulated timeout")
            return "success"

        executor = ToolExecutor()
        tool = ToolDefinition(
            name="flaky",
            handler=flaky_handler,
            timeout=1,
            retry_policy={"max_retries": 3, "retry_delay": 0.001},
        )

        result = await executor.execute(tool, {})
        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_executor_raises_after_max_retries(self):
        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise TimeoutError("always timeout")

        executor = ToolExecutor()
        tool = ToolDefinition(
            name="always_fail",
            handler=always_fails,
            timeout=1,
            retry_policy={"max_retries": 2, "retry_delay": 0.001},
        )

        with pytest.raises(ToolExecutionError, match="failed after 2 retries"):
            await executor.execute(tool, {})
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_executor_no_retry_on_value_error(self):
        call_count = [0]

        def bad_value():
            call_count[0] += 1
            raise ValueError("invalid value")

        executor = ToolExecutor()
        tool = ToolDefinition(
            name="bad",
            handler=bad_value,
            retry_policy={"max_retries": 3, "retry_delay": 0.001},
        )

        with pytest.raises(ToolExecutionError):
            await executor.execute(tool, {})
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_executor_retries_on_connection_error(self):
        call_count = [0]

        def conn_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("refused")
            return "connected"

        executor = ToolExecutor()
        tool = ToolDefinition(
            name="conn",
            handler=conn_fails,
            retry_policy={"max_retries": 3, "retry_delay": 0.001},
        )

        result = await executor.execute(tool, {})
        assert result == "connected"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_executor_default_no_retry(self):
        call_count = [0]

        def fail_once():
            call_count[0] += 1
            raise TimeoutError("fail")

        executor = ToolExecutor()
        tool = ToolDefinition(
            name="no_retry",
            handler=fail_once,
        )

        with pytest.raises(ToolTimeoutError):
            await executor.execute(tool, {})
        assert call_count[0] == 1


class TestNoBypassGateway:

    @pytest.mark.asyncio
    async def test_all_tool_paths_go_through_gateway(self):
        from enterprise_harness.gateway.gateway import ToolGateway
        from enterprise_harness.gateway.registry import ToolRegistry
        from enterprise_harness.gateway.router import ToolRouter
        from enterprise_harness.gateway.validator import ToolValidator
        from enterprise_harness.policy.engine import PolicyEngine
        from enterprise_harness.policy.rbac import Principal, RBAC, Role
        from enterprise_harness.observability.audit import AuditLogger

        audit_logger = AuditLogger()
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="safe_tool",
                description="Safe operation",
                risk_level="LOW",
                permissions=["basic:read"],
                handler=lambda: "ok",
            )
        )
        registry.register(
            ToolDefinition(
                name="risky_tool",
                description="Risky operation",
                risk_level="HIGH",
                permissions=["admin:write"],
                handler=lambda: "done",
            )
        )

        rbac = RBAC(
            roles=[
                Role(name="user", permissions=frozenset({"basic:read"})),
            ]
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            result_validator=DefaultResultValidator(),
            policy_engine=PolicyEngine(rbac=rbac),
            audit_logger=audit_logger,
        )

        principal = Principal(principal_id="u1", role="user")

        result = await gw.execute(
            tool_name="safe_tool",
            arguments={},
            principal=principal,
        )
        assert result == "ok"

        audit_events = audit_logger.get_events()
        assert len(audit_events) >= 1
        assert audit_events[0].tool == "safe_tool"
        assert audit_events[0].result.get("success") is True

        with pytest.raises(PermissionError):
            await gw.execute(
                tool_name="risky_tool",
                arguments={},
                principal=principal,
            )