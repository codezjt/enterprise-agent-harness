from __future__ import annotations

from .planner import Plan, Planner
from .task import Task


class SimplePlanner(Planner):
    """
    用于测试和基础编排的确定性 Planner。

    当前仅将一个自然语言任务转换为单任务 Plan。
    """

    async def plan(self, task: str) -> Plan:
        return Plan(
            task=task,
            tasks=[
                Task(
                    task_id="T1",
                    name="执行任务",
                    description=task,
                )
            ],
        )