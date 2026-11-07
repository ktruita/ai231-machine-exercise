import pytest
from mcp.client.client import Client

from mcp_servers.memory.server import mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_remember_recall_forget():
    async with Client(mcp) as c:
        await c.call_tool("remember", {"fact": "User's dog is named Max"})
        await c.call_tool("remember", {"fact": "User prefers Celsius"})
        hits = (await c.call_tool("recall", {"query": "what is my dog's name"})).structured_content["result"]
        assert hits and "Max" in hits[0]
        await c.call_tool("forget", {"query": "dog Max"})
        hits = (await c.call_tool("recall", {"query": "dog"})).structured_content["result"]
        assert hits == []
