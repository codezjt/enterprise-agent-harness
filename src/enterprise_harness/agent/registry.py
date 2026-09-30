from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import AgentConfig


class AgentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"


class AgentProfile(BaseModel):
    """Agent 的完整运行画像。

    AgentProfile = AgentConfig + 版本 + 状态 + 元数据

    AgentConfig 是静态配置（模型、工具、Prompt）
    AgentProfile 是 Registry 管理对象（版本、状态、谁创建的、什么时候更新的）
    """

    agent_id: str
    name: str
    description: str = ""

    status: AgentStatus = AgentStatus.ACTIVE

    current_version: str = "1.0.0"
    versions: list[str] = Field(default_factory=list)

    owner: str | None = None
    tenant_id: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "current_version": self.current_version,
            "versions": self.versions,
            "owner": self.owner,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


class AgentRegistry:
    """Agent 注册中心。

    管理：
    - AgentProfile：Agent 元信息（版本、状态、归属）
    - AgentConfig：各版本的具体配置（模型、工具、Prompt）

    第一版内存实现，后续可替换为 Postgres。
    """

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        self._configs: dict[tuple[str, str], AgentConfig] = {}

    def register(
        self,
        config: AgentConfig,
        *,
        description: str = "",
        owner: str | None = None,
        tenant_id: str | None = None,
    ) -> AgentProfile:
        if config.agent_id in self._profiles:
            raise ValueError(
                f"Agent already registered: {config.agent_id}"
            )

        profile = AgentProfile(
            agent_id=config.agent_id,
            name=config.name,
            description=description,
            current_version=config.version,
            versions=[config.version],
            owner=owner,
            tenant_id=tenant_id,
        )

        self._profiles[config.agent_id] = profile
        self._configs[(config.agent_id, config.version)] = config

        return profile

    def update_config(
        self,
        config: AgentConfig,
    ) -> AgentProfile:
        profile = self.get_profile(config.agent_id)

        key = (config.agent_id, config.version)
        if key not in self._configs:
            profile.versions.append(config.version)
        self._configs[key] = config

        profile.current_version = config.version
        profile.name = config.name
        profile.updated_at = datetime.now(timezone.utc)

        return profile

    def get_profile(self, agent_id: str) -> AgentProfile:
        profile = self._profiles.get(agent_id)
        if profile is None:
            raise KeyError(f"Agent not found: {agent_id}")
        return profile

    def get_config(
        self,
        agent_id: str,
        version: str | None = None,
    ) -> AgentConfig:
        if version is None:
            profile = self.get_profile(agent_id)
            version = profile.current_version

        config = self._configs.get((agent_id, version))
        if config is None:
            raise KeyError(
                f"Agent config not found: {agent_id}@{version}"
            )
        return config

    def list_agents(
        self,
        *,
        tenant_id: str | None = None,
        status: AgentStatus | None = None,
    ) -> list[AgentProfile]:
        profiles = list(self._profiles.values())

        if tenant_id is not None:
            profiles = [
                p for p in profiles
                if p.tenant_id == tenant_id
            ]

        if status is not None:
            profiles = [
                p for p in profiles
                if p.status == status
            ]

        return profiles

    def deactivate(self, agent_id: str) -> AgentProfile:
        profile = self.get_profile(agent_id)
        profile.status = AgentStatus.INACTIVE
        profile.updated_at = datetime.now(timezone.utc)
        return profile
