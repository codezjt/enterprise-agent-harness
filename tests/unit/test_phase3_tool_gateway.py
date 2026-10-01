import pytest

from enterprise_harness.gateway.exceptions import (
    ToolApprovalRejectedError,
    ToolApprovalRequiredError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolTimeoutError,
    ToolValidationError,
    classify_error,
)
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.models import ToolDefinition
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.registry import ToolRegistry
from enterprise_harness.gateway.validator import ToolValidator


class TestToolExceptionHierarchy:

    def test_tool_execution_error_is_runtime_error(self):
        exc = ToolExecutionError("my_tool", "something went wrong")
        assert isinstance(exc, RuntimeError)
        assert exc.tool_name == "my_tool"
        assert "something went wrong" in str(exc)

    def test_tool_not_found_error_is_key_error(self):
        exc = ToolNotFoundError("missing_tool")
        assert isinstance(exc, KeyError)
        assert exc.tool_name == "missing_tool"

    def test_tool_validation_error_is_value_error(self):
        exc = ToolValidationError("my_tool", "missing required field")
        assert isinstance(exc, ValueError)
        assert exc.tool_name == "my_tool"
        assert "missing required field" in str(exc)

    def test_tool_policy_error_is_permission_error(self):
        exc = ToolPolicyError("my_tool")
        assert isinstance(exc, PermissionError)
        assert exc.tool_name == "my_tool"

    def test_tool_timeout_error_is_timeout_error(self):
        exc = ToolTimeoutError("my_tool", 30.0)
        assert isinstance(exc, TimeoutError)
        assert exc.tool_name == "my_tool"
        assert exc.timeout == 30.0

    def test_tool_approval_required_error_is_permission_error(self):
        exc = ToolApprovalRequiredError(
            approval_id="appr-001",
            run_id="run-001",
            tool_name="risky_tool",
            arguments={"x": 1},
        )
        assert isinstance(exc, PermissionError)
        assert exc.approval_id == "appr-001"
        assert exc.tool_name == "risky_tool"

    def test_tool_approval_rejected_error_is_permission_error(self):
        exc = ToolApprovalRejectedError("risky_tool", "appr-001")
        assert isinstance(exc, PermissionError)
        assert exc.tool_name == "risky_tool"
        assert exc.approval_id == "appr-001"

    def test_all_exceptions_have_to_dict(self):
        exceptions = [
            ToolExecutionError("t", "msg"),
            ToolNotFoundError("t"),
            ToolValidationError("t", "msg"),
            ToolPolicyError("t"),
            ToolTimeoutError("t", 10.0),
            ToolApprovalRequiredError(
                approval_id="a", run_id="r", tool_name="t", arguments={}
            ),
            ToolApprovalRejectedError("t", "a"),
        ]
        for exc in exceptions:
            d = exc.to_dict()
            assert "type" in d
            assert "message" in d

    def test_classify_error_returns_correct_codes(self):
        assert classify_error(ToolNotFoundError("t")) == "TOOL_NOT_FOUND"
        assert classify_error(ToolValidationError("t", "m")) == "TOOL_VALIDATION_FAILED"
        assert classify_error(ToolPolicyError("t")) == "TOOL_POLICY_DENIED"
        assert classify_error(ToolTimeoutError("t", 10.0)) == "TOOL_TIMEOUT"
        assert classify_error(
            ToolApprovalRequiredError(
                approval_id="a", run_id="r", tool_name="t", arguments={}
            )
        ) == "TOOL_APPROVAL_REQUIRED"
        assert classify_error(ToolApprovalRejectedError("t", "a")) == "TOOL_APPROVAL_REJECTED"
        assert classify_error(ToolExecutionError("t", "m")) == "TOOL_EXECUTION_FAILED"
        assert classify_error(PermissionError("nope")) == "PERMISSION_ERROR"
        assert classify_error(ValueError("bad")) == "VALIDATION_ERROR"
        assert classify_error(KeyError("missing")) == "RESOURCE_NOT_FOUND"

    def test_classify_error_unknown(self):
        assert classify_error(RuntimeError("unknown")) == "UNKNOWN_ERROR"


class TestToolRouterTypedError:

    def test_router_raises_tool_not_found(self):
        registry = ToolRegistry()
        router = ToolRouter(registry)

        with pytest.raises(ToolNotFoundError) as excinfo:
            router.route("nonexistent")
        assert excinfo.value.tool_name == "nonexistent"
        assert isinstance(excinfo.value, KeyError)


class TestToolValidatorTypedError:

    def test_validator_raises_tool_validation_error(self):
        validator = ToolValidator()
        tool = ToolDefinition(
            name="test_tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )

        with pytest.raises(ToolValidationError) as excinfo:
            validator.validate(tool, {})
        assert excinfo.value.tool_name == "test_tool"
        assert isinstance(excinfo.value, ValueError)


class TestToolExecutorTypedErrors:

    @pytest.mark.asyncio
    async def test_executor_raises_execution_error_for_missing_handler(self):
        executor = ToolExecutor()
        tool = ToolDefinition(
            name="no_handler",
            handler=None,
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            await executor.execute(tool, {})
        assert excinfo.value.tool_name == "no_handler"

    @pytest.mark.asyncio
    async def test_executor_raises_execution_error_on_handler_failure(self):
        executor = ToolExecutor()

        def failing_handler(**kwargs):
            raise ValueError("boom")

        tool = ToolDefinition(
            name="failing_tool",
            handler=failing_handler,
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            await executor.execute(tool, {})
        assert excinfo.value.tool_name == "failing_tool"
        assert "boom" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_executor_runs_successfully(self):
        executor = ToolExecutor()

        def add(x: int, y: int) -> int:
            return x + y

        tool = ToolDefinition(
            name="add",
            handler=add,
        )

        result = await executor.execute(tool, {"x": 2, "y": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_executor_supports_async_handler(self):
        executor = ToolExecutor()

        async def async_add(**kwargs):
            return kwargs["x"] + kwargs["y"]

        tool = ToolDefinition(
            name="async_add",
            handler=async_add,
        )

        result = await executor.execute(tool, {"x": 5, "y": 6})
        assert result == 11