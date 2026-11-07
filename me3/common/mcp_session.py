"""A long-lived MCP client session usable from any asyncio task.

MCP sessions are anyio-task-bound (opened and closed in the same task), so a
dedicated owner task holds the session; callers just `await call(...)`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.client.client import Client

from common.mcp_util import tool_value


class PersistentMCP:
    def __init__(self, url: str):
        self.url = url
        self._client: Client | None = None
        self._owner: asyncio.Task | None = None
        self._closing = asyncio.Event()

    async def start(self) -> "PersistentMCP":
        if self._owner:
            return self
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def own():
            try:
                async with Client(self.url) as client:
                    self._client = client
                    ready.set_result(None)
                    await self._closing.wait()
            except BaseException as e:  # surface connection errors to start()
                if not ready.done():
                    ready.set_exception(e)
                raise
            finally:
                self._client = None

        self._owner = asyncio.create_task(own())
        await ready
        return self

    async def call(self, tool: str, args: dict[str, Any] | None = None) -> Any:
        if not self._client:
            await self.start()
        return tool_value(await self._client.call_tool(tool, args or {}))

    async def close(self) -> None:
        if self._owner:
            self._closing.set()
            try:
                await self._owner
            except BaseException:
                pass
            self._owner = None
