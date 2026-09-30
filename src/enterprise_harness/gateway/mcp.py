from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MCPServer(ABC):
    """MCP Server 抽象。

    Tool Gateway 可以注册多个 MCP Server 作为 Tool 来源。
    第一版只做抽象接口，后续接入真正的 MCP SDK。
    """

    @abstractmethod
    async def discover_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        raise NotImplementedError


class InMemoryMCPServer(MCPServer):
    """用于测试的内存 MCP Server 实现。"""

    def __init__(
        self,
        name: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self._tools = tools or []

    async def discover_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        for tool in self._tools:
            if tool.get("name") == tool_name:
                handler = tool.get("handler")
                if handler is None:
                    raise RuntimeError(
                        f"MCP tool has no handler: {tool_name}"
                    )
                return handler(**arguments)

        raise KeyError(f"MCP tool not found: {tool_name}")


class MCPToolSource:
    """Tool Gateway 对 MCP Server 的统一接入层。"""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def register(
        self,
        name: str,
        server: MCPServer,
    ) -> None:
        self._servers[name] = server

    def unregister(self, name: str) -> None:
        if name not in self._servers:
            raise KeyError(f"MCP server not found: {name}")
        del self._servers[name]

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())

    async def discover_all(self) -> list[dict[str, Any]]:
        all_tools: list[dict[str, Any]] = []
        for name, server in self._servers.items():
            tools = await server.discover_tools()
            for tool in tools:
                tool["_source"] = name
            all_tools.extend(tools)
        return all_tools

    async def invoke(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        server = self._servers.get(server_name)
        if server is None:
            raise KeyError(f"MCP server not found: {server_name}")
        return await server.invoke_tool(tool_name, arguments)
