from typing import Any

from deepagents import create_deep_agent

from .config import AgentConfig
from .model import resolve_model
from .runtime import AgentRuntime


class DeepAgentRuntime(AgentRuntime):
    """基于 DeepAgents 的 AgentRuntime 实现。"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        model = resolve_model(config.model)

        self.agent = create_deep_agent(
            model=model,
            system_prompt=config.system_prompt,
            name=config.name,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """执行一次 Agent 任务。"""

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }

        result = await self.agent.ainvoke(input_data)

        return result