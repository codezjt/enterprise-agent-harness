import pytest
from fastapi.testclient import TestClient

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.api.server import app, init_api
from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.observability import TraceManager
from enterprise_harness.policy import PolicyEngine
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import Run, RunStatus


def _create_test_tool_gateway():
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="echo", description="Echo tool",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        handler=lambda message: {"echo": message},
    ))
    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(),
        trace_manager=TraceManager(),
    )


@pytest.fixture
def client():
    registry = AgentRegistry()
    registry.register(AgentConfig(
        agent_id="test-agent", name="Test Agent", model="test-model",
        version="1.0.0", tools=["echo"],
    ))
    run_manager = RunManager(trace_manager=TraceManager())
    trace_manager = TraceManager()
    gateway = _create_test_tool_gateway()
    init_api(
        registry=registry,
        run_manager=run_manager,
        trace_manager=trace_manager,
        tool_gateway=gateway,
    )
    return TestClient(app)


class TestRunStateMachine:

    def test_created_to_running_valid(self):
        assert RunStatus.is_valid_transition(RunStatus.CREATED, RunStatus.RUNNING)

    def test_completed_to_running_invalid(self):
        assert not RunStatus.is_valid_transition(RunStatus.COMPLETED, RunStatus.RUNNING)

    def test_failed_to_running_invalid(self):
        assert not RunStatus.is_valid_transition(RunStatus.FAILED, RunStatus.RUNNING)

    def test_cancelled_to_running_invalid(self):
        assert not RunStatus.is_valid_transition(RunStatus.CANCELLED, RunStatus.RUNNING)

    def test_running_to_completed_valid(self):
        assert RunStatus.is_valid_transition(RunStatus.RUNNING, RunStatus.COMPLETED)

    def test_running_to_cancelled_valid(self):
        assert RunStatus.is_valid_transition(RunStatus.RUNNING, RunStatus.CANCELLED)

    def test_running_to_waiting_approval_valid(self):
        assert RunStatus.is_valid_transition(RunStatus.RUNNING, RunStatus.WAITING_APPROVAL)

    def test_completed_is_terminal(self):
        for status in RunStatus:
            assert not RunStatus.is_valid_transition(RunStatus.COMPLETED, status)

    def test_failed_is_terminal(self):
        for status in RunStatus:
            assert not RunStatus.is_valid_transition(RunStatus.FAILED, status)


class TestCancelDrivesRuntime:

    @pytest.mark.asyncio
    async def test_cancel_sets_event(self):
        run_manager = RunManager()
        run = await run_manager.create_run(agent_id="test", task="task1")

        assert not run_manager.is_cancelled(run.run_id)

        await run_manager.cancel_run(run.run_id)

        assert run_manager.is_cancelled(run.run_id)

    @pytest.mark.asyncio
    async def test_cancel_invalid_state_raises(self):
        run_manager = RunManager()
        run = await run_manager.create_run(agent_id="test", task="task1")
        await run_manager.cancel_run(run.run_id)

        with pytest.raises(ValueError, match="Cannot cancel"):
            await run_manager.cancel_run(run.run_id)

    @pytest.mark.asyncio
    async def test_cancel_event_survives_create(self):
        run_manager = RunManager()
        run = await run_manager.create_run(agent_id="test", task="task1")

        assert not run_manager.is_cancelled(run.run_id)
        assert run_manager._cancel_events[run.run_id] is not None


class TestRejectDrivesRuntime:

    def test_reject_sets_run_to_failed(self, client):
        client.post("/v1/agents", json={
            "agent_id": "rej-test", "name": "Rej Agent", "model": "test",
        })

        resp = client.post("/v1/runs", json={
            "agent_id": "rej-test", "task": "hello",
        })
        assert resp.status_code == 200
        data = resp.json()
        run_id = data["run_id"]

        resp = client.post(f"/v1/runs/{run_id}/reject", json={"comment": "not needed"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FAILED"
        assert "rejected" in data.get("error", "")


class TestAgentCRUD:

    def test_create_agent(self, client):
        resp = client.post("/v1/agents", json={
            "agent_id": "crud-test", "name": "CRUD Agent", "model": "gpt-4",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "crud-test"
        assert data["status"] == "ACTIVE"

    def test_list_agents(self, client):
        client.post("/v1/agents", json={
            "agent_id": "list-test", "name": "List Agent", "model": "gpt-4",
        })
        resp = client.get("/v1/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert any(a["agent_id"] == "list-test" for a in agents)

    def test_get_agent(self, client):
        client.post("/v1/agents", json={
            "agent_id": "get-test", "name": "Get Agent", "model": "gpt-4",
        })
        resp = client.get("/v1/agents/get-test")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "get-test"

    def test_get_agent_not_found(self, client):
        resp = client.get("/v1/agents/nonexistent")
        assert resp.status_code == 404

    def test_disable_agent(self, client):
        client.post("/v1/agents", json={
            "agent_id": "disable-test", "name": "Disable Agent", "model": "gpt-4",
        })
        resp = client.post("/v1/agents/disable-test/disable")
        assert resp.status_code == 200
        assert resp.json()["agent_status"] == "INACTIVE"

    def test_enable_agent(self, client):
        client.post("/v1/agents", json={
            "agent_id": "enable-test", "name": "Enable Agent", "model": "gpt-4",
        })
        resp = client.post("/v1/agents/enable-test/disable")
        assert resp.json()["agent_status"] == "INACTIVE"

        resp = client.post("/v1/agents/enable-test/enable")
        assert resp.status_code == 200
        assert resp.json()["agent_status"] == "ACTIVE"

    def test_update_agent(self, client):
        client.post("/v1/agents", json={
            "agent_id": "update-test", "name": "Old Name", "model": "gpt-4",
        })
        resp = client.put("/v1/agents/update-test", json={
            "name": "New Name", "description": "updated",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "New Name"


class TestRunCRUD:

    def test_create_run(self, client):
        resp = client.post("/v1/runs", json={
            "agent_id": "test-agent", "task": "echo hello",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "test-agent"
        assert data["status"] == "CREATED"

    def test_get_run(self, client):
        resp = client.post("/v1/runs", json={
            "agent_id": "test-agent", "task": "echo hello",
        })
        run_id = resp.json()["run_id"]

        resp = client.get(f"/v1/runs/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

    def test_get_run_not_found(self, client):
        resp = client.get("/v1/runs/nonexistent")
        assert resp.status_code == 404

    def test_list_runs(self, client):
        client.post("/v1/runs", json={"agent_id": "test-agent", "task": "task1"})
        client.post("/v1/runs", json={"agent_id": "test-agent", "task": "task2"})

        resp = client.get("/v1/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) >= 2

    def test_cancel_run(self, client):
        resp = client.post("/v1/runs", json={
            "agent_id": "test-agent", "task": "cancel me",
        })
        run_id = resp.json()["run_id"]

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

    def test_cancel_completed_run_conflict(self, client):
        resp = client.post("/v1/runs", json={
            "agent_id": "test-agent", "task": "done",
        })
        run_id = resp.json()["run_id"]

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200

        resp = client.post(f"/v1/runs/{run_id}/cancel")
        assert resp.status_code == 409


class TestToolsAPI:

    def test_list_tools(self, client):
        resp = client.get("/v1/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        tool_names = [t["name"] for t in data["tools"]]
        assert "echo" in tool_names

    def test_execute_tool(self, client):
        resp = client.post("/v1/tools/execute", json={
            "tool_name": "echo",
            "arguments": {"message": "hello world"},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["result"]["echo"] == "hello world"


class TestObservabilityAPI:

    def test_observability_endpoint(self, client):
        resp = client.post("/v1/runs", json={
            "agent_id": "test-agent", "task": "trace me",
        })
        run_id = resp.json()["run_id"]

        resp = client.get(f"/v1/runs/{run_id}/observability")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert "traces" in data
        assert "metrics" in data


class TestHealthCheck:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"