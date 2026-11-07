import asyncio

import pytest
from mcp.client.client import Client

from common.mcp_util import tool_value
from mcp_servers.delegate import server
from mcp_servers.delegate.adapters.subprocess_harness import SubprocessHarness

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def fake_harnesses(monkeypatch):
    monkeypatch.setitem(server.HARNESSES, "opencode", SubprocessHarness(
        "opencode", ["sh", "-c", "sleep 0.3; echo 'Built index.html.' > SUMMARY.md", "{goal}"], "m"))
    monkeypatch.setitem(server.HARNESSES, "hermes", SubprocessHarness(
        "hermes", ["sh", "-c", "sleep 30", "{goal}"], "m"))


async def test_delegate_returns_immediately_then_reports():
    async with Client(server.mcp) as c:
        started = tool_value(await c.call_tool("delegate_task", {"goal": "make a website"}))
        assert started["status"] == "running" and started["harness"] == "opencode"
        for _ in range(40):
            events = tool_value(await c.call_tool("poll_events", {}))
            if events:
                break
            await asyncio.sleep(0.1)
        assert events[0]["kind"] == "task_done"
        assert events[0]["result"] == "Built index.html."


async def test_cancel_is_not_reported_as_failure():
    async with Client(server.mcp) as c:
        tid = tool_value(await c.call_tool("delegate_task", {"goal": "x", "harness": "hermes"}))["task_id"]
        assert tool_value(await c.call_tool("cancel_task", {"task_id": tid})) == "Cancelled."
        await asyncio.sleep(0.3)
        status = tool_value(await c.call_tool("task_status", {"task_id": tid}))
        assert status[0]["status"] == "cancelled"
        assert tool_value(await c.call_tool("poll_events", {})) == []


async def _wait_done(c, tid):
    for _ in range(60):
        st = tool_value(await c.call_tool("task_status", {"task_id": tid}))[0]
        if st["status"] != "running":
            return st
        await asyncio.sleep(0.1)
    raise AssertionError("task did not finish")


async def test_follow_up_edits_the_same_folder(monkeypatch):
    monkeypatch.setitem(server.HARNESSES, "opencode", SubprocessHarness(
        "opencode", ["sh", "-c", "sleep 0.2; echo run >> work.txt; printf '%.40s' \"$0\" > SUMMARY.md",
                     "{goal}"], "m"))
    async with Client(server.mcp) as c:
        first = tool_value(await c.call_tool("delegate_task", {"goal": "website about Max (follow-up test)"}))
        st1 = await _wait_done(c, first["task_id"])
        assert "work.txt" in st1["files_so_far"]
        follow = tool_value(await c.call_tool("continue_task", {"instruction": "make it blue",
                                                                "task_id": first["task_id"]}))
        assert follow["status"] == "running" and follow["continues"] == first["task_id"]
        st2 = await _wait_done(c, follow["task_id"])
        assert st2["workdir"] == st1["workdir"] and st2["parent"] == first["task_id"]
        from pathlib import Path
        assert Path(st1["workdir"], "work.txt").read_text().count("run") == 2  # edited, not restarted
        assert st2["result"].startswith("This folder already contains")  # got the follow-up brief
        # no task_id: continues the most recent one (the follow-up), same folder again
        again = tool_value(await c.call_tool("continue_task", {"instruction": "add photos"}))
        assert again["continues"] == follow["task_id"]
        await _wait_done(c, again["task_id"])


async def test_follow_up_refused_while_running():
    async with Client(server.mcp) as c:
        tid = tool_value(await c.call_tool("delegate_task", {"goal": "slow job (refuse test)",
                                                             "harness": "hermes"}))["task_id"]
        out = tool_value(await c.call_tool("continue_task", {"instruction": "x", "task_id": tid}))
        assert out["status"] == "failed" and "still running" in out["error"]
        await c.call_tool("cancel_task", {"task_id": tid})


async def test_duplicate_request_is_not_started_twice():
    async with Client(server.mcp) as c:
        a = tool_value(await c.call_tool("delegate_task", {"goal": "Suggest books (dup test)", "harness": "hermes"}))
        b = tool_value(await c.call_tool("delegate_task", {"goal": "suggest books, dup test", "harness": "hermes"}))
        assert b["task_id"] == a["task_id"] and b["duplicate"] is True
        await c.call_tool("cancel_task", {"task_id": a["task_id"]})
