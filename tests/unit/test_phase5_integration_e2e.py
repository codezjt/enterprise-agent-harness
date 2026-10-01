import pytest
from fastapi.testclient import TestClient

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.deepagent_runtime import DeepAgentRuntime
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.api.server import _route_task, app, init_api
from enterprise_harness.context import ContextBuilder, ContextProvider
from enterprise_harness.gateway.deepagent import DeepAgentToolAdapter
from enterprise_harness.gateway.executor import ToolExecutor
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.gateway.models import ToolDefinition
from enterprise_harness.gateway.registry import ToolRegistry
from enterprise_harness.gateway.router import ToolRouter
from enterprise_harness.gateway.validator import ToolValidator
from enterprise_harness.observability.audit import AuditLogger
from enterprise_harness.observability.manager import TraceManager
from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.rbac import Principal, RBAC, Role
from enterprise_harness.runtime import RunManager, RunStatus


def _build_gateway() -> ToolGateway:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a file from disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path"}
                },
                "required": ["path"],
            },
            risk_level="LOW",
            permissions=["file:read"],
            handler=lambda path: {"content": f"mock content of {path}"},
        )
    )

    registry.register(
        ToolDefinition(
            name="write_file",
            description="Write content to a file on disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "file path",
                    },
                    "content": {
                        "type": "string",
                        "description": "file content",
                    },
                },
                "required": ["path", "content"],
            },
            risk_level="HIGH",
            permissions=["file:write"],
            handler=lambda path, content: {"bytes": len(content)},
        )
    )

    rbac = RBAC(
        roles=[
            Role(
                name="viewer",
                permissions=frozenset({"file:read"}),
            ),
            Role(
                name="editor",
                permissions=frozenset({"file:read", "file:write"}),
            ),
        ]
    )

    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(rbac=rbac),
        trace_manager=TraceManager(),
        audit_logger=AuditLogger(),
        metric_collector=MetricCollector(),
    )


class TestRouteTask:

    def test_route_tool_read_file(self):
        gw = _build_gateway()
        route, payload = _route_task(
            "read D:/tmp/hello.txt", gw, ["read_file", "write_file"]
        )
        assert route == "tool"
        assert payload["tool_name"] == "read_file"

    def test_route_tool_write_file(self):
        gw = _build_gateway()
        route, payload = _route_task(
            "write 0000 to D:/tmp/hello.txt", gw, ["read_file", "write_file"]
        )
        assert route == "tool"
        assert payload["tool_name"] == "write_file"

    def test_route_to_agent_for_unknown(self):
        gw = _build_gateway()
        route, payload = _route_task(
            "what is the weather today", gw, ["read_file", "write_file"]
        )
        assert route == "agent"
        assert payload is None

    def test_route_agent_when_tool_not_in_list(self):
        gw = _build_gateway()
        route, payload = _route_task(
            "read D:/tmp/hello.txt", gw, ["other_tool"]
        )
        assert route == "agent"


class TestFullChainRunManagerToToolGateway:

    @pytest.mark.asyncio
    async def test_full_chain_low_risk_tool(self):
        gw = _build_gateway()

        config = AgentConfig(
            agent_id="file-agent",
            name="File Agent",
            model="openai/gpt-4o-mini",
            tools=["read_file"],
        )

        runtime = DeepAgentRuntime(
            config=config,
            tool_gateway=gw,
        )

        manager = RunManager()
        run = await manager.create_run(
            agent_id="file-agent",
            task="read D:/tmp/hello.txt",
        )

        principal = Principal(principal_id="u1", role="viewer")
        run = await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=principal,
        )

        assert run.status in (RunStatus.COMPLETED, RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_tool_gateway_policy_deny_viewer_write(self):
        gw = _build_gateway()
        principal = Principal(principal_id="u1", role="viewer")

        with pytest.raises(PermissionError):
            await gw.execute(
                tool_name="write_file",
                arguments={"path": "/tmp/test.txt", "content": "hello"},
                principal=principal,
            )

    @pytest.mark.asyncio
    async def test_tool_gateway_editor_write_approval(self):
        gw = _build_gateway()
        principal = Principal(principal_id="u2", role="editor")

        from enterprise_harness.gateway.exceptions import ApprovalRequiredError

        with pytest.raises(ApprovalRequiredError) as excinfo:
            await gw.execute(
                tool_name="write_file",
                arguments={"path": "/tmp/test.txt", "content": "hello"},
                principal=principal,
            )

        approval_id = excinfo.value.approval_id
        assert approval_id is not None

    @pytest.mark.asyncio
    async def test_tool_gateway_approval_approve_and_retry(self):
        gw = _build_gateway()
        principal = Principal(principal_id="u2", role="editor")

        from enterprise_harness.gateway.exceptions import ApprovalRequiredError

        with pytest.raises(ApprovalRequiredError) as excinfo:
            await gw.execute(
                tool_name="write_file",
                arguments={"path": "/tmp/test.txt", "content": "hello"},
                principal=principal,
            )

        approval_id = excinfo.value.approval_id

        await gw.approval_manager.approve(approval_id, comment="ok")

        result = await gw.execute(
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
            principal=principal,
            approval_id=approval_id,
        )

        assert result == {"bytes": 5}

    @pytest.mark.asyncio
    async def test_audit_logger_records_events(self):
        audit_logger = AuditLogger()
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="echo",
                description="Echo",
                handler=lambda msg: msg,
            )
        )

        gw = ToolGateway(
            registry=registry,
            router=ToolRouter(registry),
            validator=ToolValidator(),
            executor=ToolExecutor(),
            policy_engine=PolicyEngine(),
            audit_logger=audit_logger,
        )

        principal = Principal(principal_id="u1", role="admin")
        result = await gw.execute(
            tool_name="echo",
            arguments={"msg": "hello world"},
            principal=principal,
        )
        assert result == "hello world"

        records = audit_logger.get_events()
        assert len(records) == 1
        assert records[0].tool == "echo"
        assert records[0].result is not None
        assert records[0].result.get("success") is True


class TestRunManagerEndToEnd:

    @pytest.mark.asyncio
    async def test_create_and_start_run_with_trace(self):
        gw = _build_gateway()
        trace_manager = TraceManager()

        config = AgentConfig(
            agent_id="trace-agent",
            name="Trace Agent",
            model="openai/gpt-4o-mini",
            tools=["read_file"],
        )

        runtime = DeepAgentRuntime(
            config=config,
            tool_gateway=gw,
        )

        manager = RunManager(trace_manager=trace_manager)
        run = await manager.create_run(
            agent_id="trace-agent",
            task="read /tmp/test.txt",
        )

        principal = Principal(principal_id="u1", role="viewer")
        run = await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=principal,
        )

        assert run.run_id is not None
        assert run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.WAITING_APPROVAL)

        spans = trace_manager.get_run_spans(run.run_id)
        assert len(spans) >= 1


class TestDeepAgentToolAdapterFullChain:

    def test_adapter_creates_valid_tool(self):
        gw = _build_gateway()
        principal = Principal(principal_id="u1", role="viewer")

        adapter = DeepAgentToolAdapter(
            gateway=gw,
            principal=principal,
            run_id="run-001",
            tenant_id="t1",
            context={"custom": "data"},
        )

        read_tool_def = gw.registry.get("read_file")
        adapted = adapter.adapt(read_tool_def)

        assert adapted.name == "read_file"
        assert adapted.description is not None

    def test_adapter_stores_all_fields(self):
        gw = _build_gateway()
        principal = Principal(principal_id="u1", role="admin")

        adapter = DeepAgentToolAdapter(
            gateway=gw,
            principal=principal,
            run_id="run-ctx-001",
            tenant_id="tenant-abc",
            context={"env": "prod"},
            parent_span_id="span-001",
        )

        assert adapter.principal == principal
        assert adapter.run_id == "run-ctx-001"
        assert adapter.tenant_id == "tenant-abc"
        assert adapter.context == {"env": "prod"}
        assert adapter.parent_span_id == "span-001"


class TestApprovalManagerFullChain:

    @pytest.mark.asyncio
    async def test_approval_create_approve_validate(self):
        from enterprise_harness.policy.approval import ApprovalStatus

        approval_manager = ApprovalManager()

        request = await approval_manager.create_request(
            run_id="run-001",
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
        )

        assert request.approval_id is not None
        assert request.status == ApprovalStatus.PENDING

        await approval_manager.approve(request.approval_id, comment="approved")

        validated = await approval_manager.validate_approval(
            approval_id=request.approval_id,
            run_id="run-001",
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
        )
        assert validated.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_approval_create_reject_invalid(self):
        approval_manager = ApprovalManager()

        request = await approval_manager.create_request(
            run_id="run-002",
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "hello"},
        )

        await approval_manager.reject(request.approval_id, comment="denied")

        with pytest.raises(PermissionError):
            await approval_manager.validate_approval(
                approval_id=request.approval_id,
                run_id="run-002",
                tool_name="write_file",
                arguments={"path": "/tmp/test.txt", "content": "hello"},
            )


class TestApiServerE2E:

    @pytest.fixture(autouse=True)
    def setup_api(self):
        gw = _build_gateway()
        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="api-agent",
                name="API Agent",
                model="openai/gpt-4o-mini",
                tools=["read_file", "write_file"],
            ),
            description="Test agent for API E2E",
        )

        manager = RunManager(trace_manager=TraceManager())
        approval_manager = ApprovalManager()

        init_api(
            registry=registry,
            run_manager=manager,
            trace_manager=TraceManager(),
            approval_manager=approval_manager,
            tool_gateway=gw,
        )

    def test_health(self):
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    def test_list_agents(self):
        with TestClient(app) as client:
            r = client.get("/v1/agents")
            assert r.status_code == 200
            agents = r.json()
            assert len(agents) >= 1
            assert agents[0]["agent_id"] == "api-agent"

    def test_get_agent(self):
        with TestClient(app) as client:
            r = client.get("/v1/agents/api-agent")
            assert r.status_code == 200
            assert r.json()["agent_id"] == "api-agent"

    def test_get_agent_not_found(self):
        with TestClient(app) as client:
            r = client.get("/v1/agents/nonexistent")
            assert r.status_code == 404

    def test_list_tools(self):
        with TestClient(app) as client:
            r = client.get("/v1/tools")
            assert r.status_code == 200
            data = r.json()
            assert data["total"] == 2

    def test_create_run(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/runs",
                json={"agent_id": "api-agent", "task": "read file"},
            )
            assert r.status_code == 200
            assert r.json()["agent_id"] == "api-agent"
            assert r.json()["status"] == "CREATED"

    def test_create_run_agent_not_found(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/runs",
                json={"agent_id": "nonexistent", "task": "test"},
            )
            assert r.status_code == 404

    def test_tool_execute_read(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/tools/execute",
                json={
                    "tool_name": "read_file",
                    "arguments": {"path": "/tmp/test.txt"},
                    "principal_id": "u1",
                    "principal_role": "viewer",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
            assert "content" in str(data["result"])

    def test_tool_execute_viewer_denied(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/tools/execute",
                json={
                    "tool_name": "write_file",
                    "arguments": {"path": "/tmp/test.txt", "content": "x"},
                    "principal_id": "u1",
                    "principal_role": "viewer",
                },
            )
            assert r.status_code == 403

    def test_tool_approval_flow(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/tools/execute",
                json={
                    "tool_name": "write_file",
                    "arguments": {"path": "/tmp/test.txt", "content": "hello"},
                    "principal_id": "u2",
                    "principal_role": "editor",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "REQUIRE_APPROVAL"
            approval_id = data["approval_id"]

            r_approve = client.post(
                "/v1/tools/approval/approve",
                json={"approval_id": approval_id, "comment": "ok"},
            )
            assert r_approve.status_code == 200

            r_retry = client.post(
                "/v1/tools/execute",
                json={
                    "tool_name": "write_file",
                    "arguments": {"path": "/tmp/test.txt", "content": "hello"},
                    "principal_id": "u2",
                    "principal_role": "editor",
                    "approval_id": approval_id,
                },
            )
            assert r_retry.status_code == 200
            assert r_retry.json()["status"] == "ok"


class TestChatApiE2E:

    @pytest.fixture(autouse=True)
    def setup_chat_api(self):
        gw = _build_gateway()
        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="chat-agent",
                name="Chat Agent",
                model="openai/gpt-4o-mini",
                tools=["read_file", "write_file"],
            ),
            description="Chat endpoint agent",
        )

        manager = RunManager(trace_manager=TraceManager())

        init_api(
            registry=registry,
            run_manager=manager,
            trace_manager=TraceManager(),
            approval_manager=ApprovalManager(),
            tool_gateway=gw,
        )

    def test_chat_direct_tool_read(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={
                    "agent_id": "chat-agent",
                    "message": "read D:/tmp/hello.txt",
                    "principal_id": "u1",
                    "principal_role": "viewer",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["routed_to"].startswith("tool:read_file")
            assert data["status"] == "ok"

    def test_chat_direct_tool_write_approval(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={
                    "agent_id": "chat-agent",
                    "message": "write hello to D:/tmp/test.txt",
                    "principal_id": "u2",
                    "principal_role": "editor",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "WAITING_APPROVAL"

    def test_chat_unknown_goes_to_agent(self):
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={
                    "agent_id": "chat-agent",
                    "message": "what is the weather in Shanghai",
                    "principal_id": "u1",
                    "principal_role": "viewer",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["routed_to"] == "deepagent"


class TestContextMemoryRagE2E:

    @pytest.mark.asyncio
    async def test_context_provider_enrichment_in_full_chain(self):
        from enterprise_harness.memory import MemoryManager, MemoryScope, ScopedShortTermMemory
        from enterprise_harness.rag import InMemoryRetriever, Document

        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="File API",
                content="The read_file tool reads file contents. The write_file tool writes file contents.",
            )
        )

        memory_manager = MemoryManager(short_term=ScopedShortTermMemory())
        await memory_manager.store(
            {"role": "user", "content": "need to read a file"},
            scope=MemoryScope(tenant_id="t1", agent_id="agent-1"),
        )

        context_provider = ContextProvider(
            memory_manager=memory_manager,
            retriever=retriever,
        )

        gw = _build_gateway()

        config = AgentConfig(
            agent_id="context-agent",
            name="Context Agent",
            model="openai/gpt-4o-mini",
            tools=["read_file"],
        )

        runtime = DeepAgentRuntime(
            config=config,
            tool_gateway=gw,
        )

        manager = RunManager(
            context_provider=context_provider,
            context_builder=ContextBuilder(),
        )

        run = await manager.create_run(
            agent_id="context-agent",
            task="read D:/tmp/hello.txt",
            tenant_id="t1",
        )

        principal = Principal(principal_id="u1", role="viewer")
        run = await manager.start_run(
            run_id=run.run_id,
            runtime=runtime,
            principal=principal,
        )

        assert run.run_id is not None
        assert run.status in (RunStatus.COMPLETED, RunStatus.FAILED)