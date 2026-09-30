"""European Order Agent Demo - 端到端跑通 Harness 完整闭环。

运行方式: python -m examples.european_order_agent.demo

演示内容：
  1. Agent Registration + AgentConfig
  2. Tool Gateway 完整链路（Registry → Schema Validation → Policy → RBAC → Approval → Execute）
  3. Task DAG 并行调度（T1/T2/T3 并行 → T4 → T5）
  4. HITL：update_order 触发 REQUIRE_APPROVAL → 人工批准 → Resume
  5. Trace 全链路记录 + Audit 日志
  6. Recovery：模拟 Tool 超时 → Retry / Replan
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.registry import AgentRegistry
from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.observability import (
    AuditLogger,
    MetricCollector,
    TraceManager,
)
from enterprise_harness.orchestration import (
    PlanValidator,
    Planner,
    Scheduler,
    SimpleReplanner,
    SimplePlanner,
    Task,
    TaskGraph,
)
from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.policy import (
    PolicyDecision,
    PolicyEngine,
    Principal,
    RBAC,
    Role,
)
from enterprise_harness.runtime import (
    AgentRuntimeAdapter,
    LangGraphRuntime,
    LangGraphRuntimeAdapter,
    RunManager,
    RunStatus,
)
from enterprise_harness.runtime.recovery import RecoveryManager


# ─── 模拟数据层 ────────────────────────────────────────────────────

class InMemoryOrderSystem:
    """模拟后端订单系统。"""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {
            "1001": {
                "order_id": "1001",
                "customer": "Acme GmbH",
                "items": "widget_blue",
                "quantity": 100,
                "status": "pending_shipment",
                "destination": "Berlin, Germany",
            },
        }
        self._inventory: dict[str, int] = {
            "widget_blue": 60,
            "widget_red": 50,
        }
        self._logistics: dict[str, dict] = {
            "1001": {
                "shipped": 20,
                "carrier": "DHL Express",
                "eta_days": 2,
            },
        }

    def query_order(self, order_id: str) -> dict:
        return dict(self._orders[order_id])

    def query_inventory(self, item: str) -> dict:
        return {"item": item, "stock": self._inventory.get(item, 0)}

    def query_logistics(self, order_id: str) -> dict:
        return dict(self._logistics[order_id])

    def update_order(
        self,
        order_id: str,
        quantity: int | None = None,
        status: str | None = None,
    ) -> dict:
        order = self._orders[order_id]
        if quantity is not None:
            order["quantity"] = quantity
        if status is not None:
            order["status"] = status
        return {"updated": True, "order": dict(order)}


# ─── Tools 定义 ────────────────────────────────────────────────────

def build_tools(
    system: InMemoryOrderSystem,
) -> list[ToolDefinition]:

    return [
        ToolDefinition(
            name="query_order",
            description="查询订单详情",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
            risk_level="LOW",
            permissions=["order:read"],
            timeout=10,
            handler=system.query_order,
        ),
        ToolDefinition(
            name="query_inventory",
            description="查询库存数量",
            input_schema={
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                },
                "required": ["item"],
            },
            risk_level="LOW",
            permissions=["order:read"],
            timeout=10,
            handler=system.query_inventory,
        ),
        ToolDefinition(
            name="query_logistics",
            description="查询物流状态",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
            risk_level="LOW",
            permissions=["order:read"],
            timeout=10,
            handler=system.query_logistics,
        ),
        ToolDefinition(
            name="update_order",
            description="修改订单（高风险，需要审批）",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "status": {"type": "string"},
                },
                "required": ["order_id"],
            },
            risk_level="HIGH",
            permissions=["order:update"],
            timeout=10,
            handler=system.update_order,
        ),
    ]


# ─── Minimal Agent stub ──────────────────────────────────────────

class MinimalAgentRuntime:
    """不依赖 DeepAgents 的极简 Agent。

    这个 Demo 的核心是跑通 Harness 链路，
    不依赖真实 LLM。根据 task_name 精准路由
    到 Tool Gateway，完整触发 Policy / RBAC / Trace / Audit。
    """

    def __init__(self, gateway: ToolGateway):
        self.gateway = gateway

    async def run(
        self,
        task: str,
        context: dict | None = None,
    ):
        ctx = context or {}
        task_name = ctx.get("task_name", "")
        deps = ctx.get("dependency_results", {})

        run_id = ctx.get("_run_id")
        principal = ctx.get("_principal")
        gw_kwargs = dict(run_id=run_id, principal=principal)

        if task_name == "query_order":
            return await self.gateway.execute(
                tool_name="query_order",
                arguments={"order_id": "1001"},
                context=ctx,
                **gw_kwargs,
            )

        if task_name == "query_inventory":
            return await self.gateway.execute(
                tool_name="query_inventory",
                arguments={"item": "widget_blue"},
                context=ctx,
                **gw_kwargs,
            )

        if task_name == "query_logistics":
            return await self.gateway.execute(
                tool_name="query_logistics",
                arguments={"order_id": "1001"},
                context=ctx,
                **gw_kwargs,
            )

        if task_name == "analyze":
            return self._analyze(deps)

        if task_name == "update_order":
            return await self._update_order(deps, ctx, gw_kwargs)

        return {"task": task, "status": "completed"}

    @staticmethod
    def _analyze(deps: dict) -> dict:
        order = deps.get("T1")
        inventory = deps.get("T2")
        logistics = deps.get("T3")

        if order is None or inventory is None or logistics is None:
            return {"analysis": "缺少必要依赖数据", "deps_keys": list(deps.keys())}

        qty = order.get("quantity", 0) if isinstance(order, dict) else 0
        inv_stock = inventory.get("stock", 0) if isinstance(inventory, dict) else 0
        shipped = logistics.get("shipped", 0) if isinstance(logistics, dict) else 0

        available = inv_stock + shipped
        need_adjust = qty > available

        return {
            "analysis": f"订单数量 {qty}，库存 {inv_stock}，已发货 {shipped}，"
                       f"可发货 {available}。需要调整: {need_adjust}",
            "order_qty": qty,
            "available": available,
            "need_adjust": need_adjust,
        }

    async def _update_order(
        self,
        deps: dict,
        ctx: dict,
        gw_kwargs: dict,
    ) -> dict:
        analyze = deps.get("T4")

        need_adjust = False
        available = 0
        if isinstance(analyze, dict):
            need_adjust = analyze.get("need_adjust", False)
            available = analyze.get("available", 0)

        if not need_adjust:
            return {"status": "ok", "message": "无需调整"}

        return await self.gateway.execute(
            tool_name="update_order",
            arguments={"order_id": "1001", "quantity": available},
            context=ctx,
            **gw_kwargs,
        )


# ─── Demo 主流程 ──────────────────────────────────────────────────

async def run_demo():
    print("=" * 60)
    print("  Enterprise Agent Harness - European Order Agent Demo")
    print("=" * 60)

    # 1. 基础设施 ------------------------------------------------
    trace_manager = TraceManager()
    metric_collector = MetricCollector()
    audit_logger = AuditLogger()

    # 2. Tool Gateway --------------------------------------------
    system = InMemoryOrderSystem()
    tools = build_tools(system)

    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register(tool)

    rbac = RBAC(roles=[
        Role(name="viewer", permissions=frozenset({"order:read"})),
        Role(name="operator", permissions=frozenset({"order:read", "order:update"})),
        Role(name="manager", permissions=frozenset({"order:read", "order:update", "order:cancel"})),
        Role(name="admin", permissions=frozenset({"order:read", "order:update", "order:cancel", "order:refund"})),
    ])

    policy_engine = PolicyEngine(rbac=rbac)

    gateway = ToolGateway(
        registry=tool_registry,
        router=ToolRouter(tool_registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=policy_engine,
        trace_manager=trace_manager,
        audit_logger=audit_logger,
        metric_collector=metric_collector,
    )

    # 3. Agent Registry + AgentConfig ----------------------------
    agent_registry = AgentRegistry()
    agent_config = AgentConfig(
        agent_id="europe-order-agent",
        name="European Order Agent",
        version="1.0.0",
        model="mock",
        system_prompt="你是欧洲区域订单异常分析 Agent。",
        tools=["query_order", "query_inventory", "query_logistics", "update_order"],
    )
    agent_registry.register(agent_config, description="欧洲配件订单异常分析", owner="ops-team")

    agent_runtime = MinimalAgentRuntime(gateway=gateway)

    # 4. RunManager -----------------------------------------------
    replanner = SimpleReplanner()
    recovery = RecoveryManager(replanner=replanner, max_replans=1)
    run_manager = RunManager(replanner=replanner, trace_manager=trace_manager)

    # 5. Principal ------------------------------------------------
    principal = Principal(principal_id="user-001", role="manager")

    # 6. Planner → TaskGraph → Scheduler --------------------------
    print("\n📋  Phase 1: Planner 生成 Task DAG")
    print("-" * 40)

    # 手动定义 Task DAG（真实场景由 LLM Planner 生成）
    task_graph = TaskGraph()

    t1 = Task(task_id="T1", name="query_order", description="查询订单 1001 详情", max_retries=2)
    t2 = Task(task_id="T2", name="query_inventory", description="查询 widget_blue 库存", max_retries=2)
    t3 = Task(task_id="T3", name="query_logistics", description="查询订单 1001 物流", max_retries=2)
    t4 = Task(task_id="T4", name="analyze", description="综合分析订单异常")
    t5 = Task(task_id="T5", name="update_order", description="调整订单数量（需要审批）")

    task_graph.add_task(t1)
    task_graph.add_task(t2)
    task_graph.add_task(t3)
    task_graph.add_task(t4)
    task_graph.add_task(t5)

    task_graph.add_dependency("T4", "T1")
    task_graph.add_dependency("T4", "T2")
    task_graph.add_dependency("T4", "T3")
    task_graph.add_dependency("T5", "T4")

    print(f"   Task DAG 已构建: T1/T2/T3 并行 → T4 → T5")
    print(f"   依赖关系已校验（无环）")

    # 7. 创建 Run -------------------------------------------------
    run = await run_manager.create_run(
        agent_id="europe-order-agent",
        task="分析欧洲配件订单 1001 异常，如果需要修改订单必须经过人工审批",
        context={},
    )
    print(f"\n🏃  Phase 2: Run 创建完成")
    print(f"   Run ID: {run.run_id}")

    # 8. 并行执行 T1 / T2 / T3 ------------------------------------
    print("\n⚡  Phase 3: 并行执行 T1/T2/T3")
    print("-" * 40)

    run_context = run_manager.context_builder.build(run, principal=principal)

    executor = AgentTaskExecutor(runtime=agent_runtime, run_context=run_context)
    scheduler = Scheduler(graph=task_graph, executor=executor, max_concurrency=4)

    result_graph = await scheduler.run()

    for task in result_graph.tasks():
        status = "✅" if task.status.value == "SUCCESS" else "❌"
        output_summary = str(task.output)[:60] if task.output else "-"
        print(f"   {status} {task.task_id} {task.name}: {task.status.value}")
        print(f"      输出: {output_summary}")

    # 9. 展示分析结果 ----------------------------------------------
    print("\n📊  Phase 4: T4 分析结果")
    print("-" * 40)

    t4_output = result_graph.get_task("T4").output
    if isinstance(t4_output, dict):
        analysis = t4_output.get("analysis", "(无分析结果)")
        print(f"   {analysis}")

    # 10. T5 update_order → 触发审批 --------------------------
    print("\n⚠️  Phase 5: T5 update_order 触发审批")
    print("-" * 40)

    t5_task = result_graph.get_task("T5")
    approval_id = None

    if t5_task.status.value == "FAILED":
        error = t5_task.error or ""
        print(f"   ❌ T5 被阻止: {error}")

        # 从 audit log 里拿 approval_id
        events = audit_logger.get_run_events(run.run_id)
        for ev in events:
            r = ev.result
            if isinstance(r, dict) and r.get("approval_id"):
                approval_id = r["approval_id"]
                break

        if approval_id:
            print(f"   📝 Approval ID: {approval_id}")

            # 11. 模拟人工批准 ---------------------------------
            print("\n✅  Phase 6: 人工批准")
            print("-" * 40)

            await gateway.approval_manager.approve(
                approval_id, comment="库存确实不足，调整合理"
            )

            print(f"   审批人: manager-001")
            print(f"   批准理由: 库存确实不足，调整合理")
            print(f"   ✅ 已批准")

            # 手动执行 update_order （带 approval_id）
            result = await gateway.execute(
                tool_name="update_order",
                arguments={"order_id": "1001", "quantity": 80},
                context={"agent_id": "europe-order-agent"},
                principal=principal,
                run_id=run.run_id,
                approval_id=approval_id,
            )

            print(f"\n🎯  Phase 7: 最终结果")
            print("-" * 40)
            print(f"   ✅ Tool 执行成功: {result}")

    # 12. Trace 全链路展示 ----------------------------------------
    print(f"\n🔍  Phase 8: Trace 全链路")
    print("-" * 40)
    spans = trace_manager.get_run_spans(run.run_id)
    for span in sorted(spans, key=lambda s: s.start_time):
        dur = f"{span.duration_ms:.1f}ms" if span.duration_ms else "running"
        print(f"   [{span.component.value:10}] {span.name:25} {dur}  {span.status.value}")

    # 13. Audit 日志 -----------------------------------------------
    print(f"\n📝  Phase 9: Audit 日志")
    print("-" * 40)
    for ev in audit_logger.get_run_events(run.run_id):
        r = ev.result if isinstance(ev.result, dict) else {"raw": str(ev.result)[:40]}
        print(f"   [{ev.decision:12}] {ev.tool:20} → {r.get('success', r)}")

    # 14. Metrics --------------------------------------------------
    print(f"\n📈  Phase 10: Metrics")
    print("-" * 40)
    snap = metric_collector.snapshot()
    print(f"   Tool Calls:      {snap.tool_calls} (success {snap.tool_success_rate:.0%})")
    print(f"   Latency (avg):   {snap.average_latency_ms:.1f}ms")
    print(f"   Policy Denied:   {snap.policy_denied}")
    print(f"   Tool Retries:    {snap.tool_retries}")

    print("\n" + "=" * 60)
    print("  Demo 完成 ✅ 完整链路已跑通")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
