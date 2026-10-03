import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.deepagent_runtime import DeepAgentRuntime
from enterprise_harness.context import ContextBuilder, ContextProvider
from enterprise_harness.gateway.deepagent import DeepAgentToolAdapter
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.registry import ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.gateway.models import ToolDefinition
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.rbac import Principal
from enterprise_harness.runtime import RunManager, RunStatus
from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime


def _create_gateway() -> ToolGateway:
    registry = ToolRegistry()

    def _handler(**kwargs):
        return {"called_with": kwargs}

    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        handler=_handler,
    )
    registry.register(tool_def)

    router = ToolRouter(registry)
    validator = ToolValidator()
    executor = ToolExecutor()
    policy_engine = PolicyEngine()

    return ToolGateway(
        registry=registry,
        router=router,
        validator=validator,
        executor=executor,
        policy_engine=policy_engine,
    )


def test_run_context_factory_includes_tenant_and_version():
    from enterprise_harness.runtime.context_factory import RunContextFactory
    from enterprise_harness.runtime.models import Run

    run = Run(
        run_id="run-001",
        agent_id="agent-001",
        agent_version="2.0.0",
        tenant_id="tenant-abc",
        task="test task",
    )

    ctx = RunContextFactory.create(run)

    assert ctx.run_id == "run-001"
    assert ctx.agent_id == "agent-001"
    assert ctx.agent_version == "2.0.0"
    assert ctx.tenant_id == "tenant-abc"


def test_run_context_has_required_fields():
    ctx = RunContext(
        run_id="run-001",
        agent_id="agent-001",
        agent_version="1.5.0",
        tenant_id="tenant-xyz",
        task="test task",
    )

    assert ctx.run_id == "run-001"
    assert ctx.agent_id == "agent-001"
    assert ctx.agent_version == "1.5.0"
    assert ctx.tenant_id == "tenant-xyz"


@pytest.mark.asyncio
async def test_run_context_propagates_to_deepagent():
    gateway = _create_gateway()

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="openai/gpt-4o-mini",
        tools=["test_tool"],
    )

    runtime = DeepAgentRuntime(
        config=config,
        tool_gateway=gateway,
        context_builder=ContextBuilder(),
        context_provider=ContextProvider(),
    )

    run_context = RunContext(
        run_id="run-prop-001",
        agent_id="test-agent",
        agent_version="2.0.0",
        tenant_id="tenant-abc",
        task="call test_tool with arg x=1",
        principal=Principal(
            principal_id="user-001",
            role="admin",
        ),
    )

    tools = runtime._build_tools_for_run(run_context)
    assert len(tools) == 1

    adapted_tool = tools[0]
    assert adapted_tool.name == "test_tool"


@pytest.mark.asyncio
async def test_deepagent_tool_adapter_receives_run_context_fields():
    gateway = _create_gateway()
    principal = Principal(principal_id="user-001", role="admin")

    adapter = DeepAgentToolAdapter(
        gateway=gateway,
        principal=principal,
        run_id="run-ctx-001",
        tenant_id="tenant-abc",
        context={"custom": "data"},
        parent_span_id="span-001",
    )

    assert adapter.run_id == "run-ctx-001"
    assert adapter.tenant_id == "tenant-abc"
    assert adapter.principal == principal
    assert adapter.context == {"custom": "data"}
    assert adapter.parent_span_id == "span-001"


@pytest.mark.asyncio
async def test_run_context_reaches_deepagent_and_gateway():
    gateway = _create_gateway()

    config = AgentConfig(
        agent_id="order-agent",
        name="Order Agent",
        model="openai/gpt-4o-mini",
        tools=["test_tool"],
    )

    runtime = DeepAgentRuntime(
        config=config,
        tool_gateway=gateway,
        context_builder=ContextBuilder(),
        context_provider=ContextProvider(),
    )

    manager = RunManager()

    run = await manager.create_run(
        agent_id="order-agent",
        task="call test_tool",
        agent_version="2.0.0",
        tenant_id="tenant-abc",
    )

    assert run.agent_version == "2.0.0"
    assert run.tenant_id == "tenant-abc"

    principal = Principal(
        principal_id="user-001",
        role="admin",
    )

    run_context = manager.context_builder.build(run, principal=principal)
    assert run_context.run_id == run.run_id
    assert run_context.agent_id == "order-agent"
    assert run_context.agent_version == "2.0.0"
    assert run_context.tenant_id == "tenant-abc"
    assert run_context.principal == principal
    assert run_context.trace_root_span_id is not None
    assert run_context.current_span_id is not None

    tools = runtime._build_tools_for_run(run_context)
    assert len(tools) == 1
    assert tools[0].name == "test_tool"


@pytest.mark.asyncio
async def test_start_run_with_langgraph_runtime():
    gateway = _create_gateway()

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="openai/gpt-4o-mini",
        tools=["test_tool"],
    )

    deepagent_rt = DeepAgentRuntime(
        config=config,
        tool_gateway=gateway,
        context_builder=ContextBuilder(),
        context_provider=ContextProvider(),
    )

    langgraph_rt = LangGraphRuntime(deepagent_runtime=deepagent_rt)

    manager = RunManager(runtime=langgraph_rt)

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    principal = Principal(
        principal_id="user-001",
        role="admin",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
        principal=principal,
    )

    assert result.run_id == run.run_id