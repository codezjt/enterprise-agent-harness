from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Agent 的静态配置。

    AgentConfig 描述：
    - Agent 是谁
    - 使用什么模型
    - 使用哪些工具
    - 使用哪些 Skills
    - 使用哪些 Middleware
    """

    agent_id: str = Field(
        description="Agent 唯一标识"
    )

    name: str = Field(
        description="Agent 名称"
    )

    version: str = Field(
        default="1.0.0",
        description="Agent 版本"
    )

    model: str = Field(
        description="模型名称"
    )

    system_prompt: str = Field(
        default="",
        description="Agent 系统提示词"
    )

    tools: list[str] = Field(
        default_factory=list,
        description="Agent 可使用的 Tool 名称"
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Agent 使用的 Skill"
    )

    middleware: list[str] = Field(
        default_factory=list,
        description="Agent 使用的 Middleware"
    )

    metadata: dict = Field(
        default_factory=dict,
        description="扩展元数据"
    )