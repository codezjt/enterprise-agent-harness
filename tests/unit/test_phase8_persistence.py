import os
import tempfile

import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.policy.approval import ApprovalStatus
from enterprise_harness.policy.approval_manager import ApprovalManager
from enterprise_harness.repositories import (
    InMemoryAgentRepository,
    InMemoryApprovalRepository,
    InMemoryCheckpointRepository,
    InMemoryRunRepository,
    SqliteAgentRepository,
    SqliteApprovalRepository,
    SqliteCheckpointRepository,
    SqliteRunRepository,
)
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus


class TestInMemoryRunRepository:

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        repo = InMemoryRunRepository()
        manager = RunManager(run_repository=repo)

        run = await manager.create_run(
            agent_id="test-agent",
            task="hello",
            tenant_id="tenant-a",
        )

        loaded = await repo.get(run.run_id)
        assert loaded is not None
        assert loaded.agent_id == "test-agent"
        assert loaded.task == "hello"
        assert loaded.tenant_id == "tenant-a"

    @pytest.mark.asyncio
    async def test_simulate_restart_recovery(self):
        repo = InMemoryRunRepository()

        manager_a = RunManager(run_repository=repo)
        run = await manager_a.create_run(
            agent_id="test-agent",
            task="hello",
            tenant_id="tenant-a",
        )
        run.status = RunStatus.COMPLETED
        await repo.save(run)

        manager_b = RunManager(run_repository=repo)
        loaded = await repo.get(run.run_id)
        assert loaded is not None
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.task == "hello"


class TestSqliteRunRepository:

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        repo = SqliteRunRepository(":memory:")
        manager = RunManager(run_repository=repo)

        run = await manager.create_run(
            agent_id="test-agent",
            task="hello",
            tenant_id="tenant-a",
        )

        loaded = await repo.get(run.run_id)
        assert loaded is not None
        assert loaded.agent_id == "test-agent"
        assert loaded.tenant_id == "tenant-a"

    @pytest.mark.asyncio
    async def test_restart_recovery(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            repo_a = SqliteRunRepository(db_path)

            manager_a = RunManager(run_repository=repo_a)
            run_a = await manager_a.create_run(
                agent_id="agent-1",
                task="task-a",
                tenant_id="tenant-a",
            )
            run_b = await manager_a.create_run(
                agent_id="agent-2",
                task="task-b",
                tenant_id="tenant-b",
            )

            run_a.status = RunStatus.COMPLETED
            await repo_a.save(run_a)
            run_b.status = RunStatus.FAILED
            await repo_a.save(run_b)

            repo_b = SqliteRunRepository(db_path)
            manager_b = RunManager(run_repository=repo_b)

            loaded_a = await repo_b.get(run_a.run_id)
            assert loaded_a is not None
            assert loaded_a.status == RunStatus.COMPLETED
            assert loaded_a.tenant_id == "tenant-a"

            loaded_b = await repo_b.get(run_b.run_id)
            assert loaded_b is not None
            assert loaded_b.status == RunStatus.FAILED
            assert loaded_b.tenant_id == "tenant-b"

            await repo_a.close()
            await repo_b.close()
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_tenant_filtering(self):
        repo = SqliteRunRepository(":memory:")
        manager = RunManager(run_repository=repo)

        await manager.create_run(
            agent_id="a1", task="t1", tenant_id="tenant-a"
        )
        await manager.create_run(
            agent_id="a2", task="t2", tenant_id="tenant-a"
        )
        await manager.create_run(
            agent_id="a3", task="t3", tenant_id="tenant-b"
        )

        a_runs = await repo.list(tenant_id="tenant-a")
        assert len(a_runs) == 2

        b_runs = await repo.list(tenant_id="tenant-b")
        assert len(b_runs) == 1


class TestSqliteApprovalRepository:

    @pytest.mark.asyncio
    async def test_approval_restart_recovery(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            approval_repo = SqliteApprovalRepository(db_path)
            run_repo = SqliteRunRepository(db_path)

            manager_a = RunManager(run_repository=run_repo)
            run = await manager_a.create_run(
                agent_id="test-agent",
                task="needs approval",
            )

            approval_mgr_a = ApprovalManager(approval_repository=approval_repo)
            req = await approval_mgr_a.create_request(
                run_id=run.run_id,
                tool_name="update_order",
                arguments={"order_id": "1001"},
                tenant_id="tenant-a",
            )

            approval_repo_b = SqliteApprovalRepository(db_path)
            approval_mgr_b = ApprovalManager(approval_repository=approval_repo_b)

            loaded = await approval_repo_b.get(req.approval_id)
            assert loaded is not None
            assert loaded.tool_name == "update_order"
            assert loaded.tenant_id == "tenant-a"
            assert loaded.status == ApprovalStatus.PENDING

            await approval_mgr_b.approve(req.approval_id)
            approved = await approval_repo_b.get(req.approval_id)
            assert approved is not None
            assert approved.status == ApprovalStatus.APPROVED
            assert approved.resolved_at is not None

            await approval_repo_b.close()
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


class TestSqliteAgentRepository:

    @pytest.mark.asyncio
    async def test_agent_config_persistence(self):
        repo = SqliteAgentRepository(":memory:")

        config = AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            version="1.0.0",
            model="test-model",
            system_prompt="You are an order assistant.",
            tools=["query_order"],
            skills=["order_skill"],
            middleware=["filesystem"],
        )

        await repo.save(config)

        loaded = await repo.get("order-agent", "1.0.0")
        assert loaded is not None
        assert loaded.name == "Order Agent"
        assert loaded.model == "test-model"
        assert loaded.tools == ["query_order"]
        assert loaded.skills == ["order_skill"]
        assert loaded.middleware == ["filesystem"]

    @pytest.mark.asyncio
    async def test_agent_version_history(self):
        repo = SqliteAgentRepository(":memory:")

        v1 = AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            version="1.0.0",
            model="test-model",
        )
        v2 = AgentConfig(
            agent_id="order-agent",
            name="Order Agent v2",
            version="2.0.0",
            model="test-model-v2",
        )

        await repo.save(v1)
        await repo.save(v2)

        latest = await repo.get("order-agent")
        assert latest is not None
        assert latest.version == "2.0.0"

        old = await repo.get("order-agent", "1.0.0")
        assert old is not None
        assert old.version == "1.0.0"

        versions = await repo.list_versions("order-agent")
        assert versions == ["1.0.0", "2.0.0"]


class TestSqliteCheckpointRepository:

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_get(self):
        repo = SqliteCheckpointRepository(":memory:")

        checkpoint = {
            "thread_id": "run-001",
            "state": {"messages": []},
            "metadata": {"step": 1},
        }

        await repo.save("run-001", checkpoint)
        loaded = await repo.get("run-001")

        assert loaded is not None
        assert loaded["thread_id"] == "run-001"
        assert loaded["metadata"]["step"] == 1


class TestEndToEndRestartRecovery:

    @pytest.mark.asyncio
    async def test_run_survives_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            run_repo = SqliteRunRepository(db_path)

            manager = RunManager(run_repository=run_repo)
            run = await manager.create_run(
                agent_id="test-agent",
                task="survive restart",
                tenant_id="tenant-x",
            )
            run.status = RunStatus.COMPLETED
            run.result = {"answer": "done"}
            await run_repo.save(run)

            run_repo_b = SqliteRunRepository(db_path)
            new_manager = RunManager(run_repository=run_repo_b)
            recovered = await run_repo_b.get(run.run_id)
            assert recovered is not None
            assert recovered.status == RunStatus.COMPLETED
            assert recovered.result == {"answer": "done"}
            assert recovered.tenant_id == "tenant-x"

            await run_repo.close()
            await run_repo_b.close()
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_approval_survives_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            approval_repo = SqliteApprovalRepository(db_path)

            mgr_a = ApprovalManager(approval_repository=approval_repo)
            req = await mgr_a.create_request(
                run_id="run-001",
                tool_name="delete_order",
                arguments={"order_id": "999"},
                tenant_id="tenant-x",
            )

            approval_repo_b = SqliteApprovalRepository(db_path)
            mgr_b = ApprovalManager(approval_repository=approval_repo_b)
            recovered = await approval_repo_b.get(req.approval_id)
            assert recovered is not None
            assert recovered.status == ApprovalStatus.PENDING
            assert recovered.tool_name == "delete_order"

            await mgr_b.approve(req.approval_id)
            assert (await approval_repo_b.get(req.approval_id)).status == ApprovalStatus.APPROVED

            await approval_repo.close()
            await approval_repo_b.close()
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_survives_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            agent_repo = SqliteAgentRepository(db_path)

            config = AgentConfig(
                agent_id="prod-agent",
                name="Production Agent",
                version="3.0.0",
                model="gpt-4",
                tools=["tool_a", "tool_b"],
            )
            await agent_repo.save(config)

            new_repo = SqliteAgentRepository(db_path)
            recovered = await new_repo.get("prod-agent", "3.0.0")
            assert recovered is not None
            assert recovered.name == "Production Agent"
            assert recovered.model == "gpt-4"
            assert recovered.tools == ["tool_a", "tool_b"]

            await agent_repo.close()
            await new_repo.close()
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass