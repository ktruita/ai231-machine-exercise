import asyncio
import time

import pytest
from mcp.client.client import Client

from mcp_servers.home.server import mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def text(result) -> str:
    return result.content[0].text


async def test_all_option_b_intents_logged():
    start = time.time()
    calls = [
        ("music", {"action": "play"}, "PLAY_MUSIC"),
        ("get_time", {}, "TIME"),
        ("lights", {"action": "on"}, "LIGHT_ON"),
        ("lights", {"action": "off"}, "LIGHT_OFF"),
        ("music", {"action": "pause"}, "PAUSE"),
        ("music", {"action": "stop"}, "STOP"),
        ("music", {"action": "next"}, "NEXT"),
        ("music", {"action": "volume_up"}, "VOLUME_UP"),
        ("music", {"action": "volume_down"}, "VOLUME_DOWN"),
        ("call", {}, "CALL"),
        ("send_message", {}, "MESSAGE"),
        ("list_reminders", {}, "LIST_REMINDERS"),
        ("set_timer", {"duration_seconds": 30}, "TIMER"),
        ("set_alarm", {"time_24h": "06:00"}, "ALARM"),
        ("set_temperature", {"degrees": 22}, "TEMPERATURE"),
        ("lights", {"action": "brightness", "brightness_percent": 60}, "BRIGHTNESS"),
        ("lights", {"action": "color", "color": "Blue"}, "COLOR"),
        ("create_reminder", {"task": "Drink water"}, "CREATE_REMINDER"),
    ]
    async with Client(mcp) as c:
        for name, args, _ in calls:
            r = await c.call_tool(name, args)
            assert not r.is_error, (name, text(r))
        log = (await c.call_tool("intent_log_since", {"ts": start})).structured_content["result"]
    assert [e["intent"] for e in log] == [i for *_, i in calls]
    slots = {e["intent"]: e["slots"] for e in log}
    assert slots["TIMER"] == {"duration_seconds": 30}
    assert slots["ALARM"] == {"time": "06:00"}
    assert slots["COLOR"] == {"color": "blue"}
    assert slots["CREATE_REMINDER"] == {"task": "drink water"}


async def test_timer_fires_event():
    async with Client(mcp) as c:
        await c.call_tool("poll_events", {})
        await c.call_tool("set_timer", {"duration_seconds": 1})
        for _ in range(20):
            events = (await c.call_tool("poll_events", {})).structured_content["result"]
            if events:
                break
            await asyncio.sleep(0.2)
        assert events and events[0]["kind"] == "timer"


async def test_reminders_roundtrip():
    async with Client(mcp) as c:
        await c.call_tool("create_reminder", {"task": "Study"})
        r = await c.call_tool("list_reminders", {})
        assert "Study" in text(r)


async def test_call_rings_connects_and_hangs_up(monkeypatch):
    from common.mcp_util import tool_value
    from mcp_servers.home import server
    monkeypatch.setattr(server, "CALL_RING_S", 0.2)
    async with Client(mcp) as c:
        assert text(await c.call_tool("call", {"contact": "Mom"})) == "Calling Mom now."
        d = tool_value(await c.call_tool("device_state", {}))
        assert d["in_call"]["contact"] == "Mom" and d["in_call"]["status"] == "ringing"
        await asyncio.sleep(0.3)
        assert tool_value(await c.call_tool("device_state", {}))["in_call"]["status"] == "connected"
        assert text(await c.call_tool("end_call", {})).startswith("Call with Mom ended")
        d = tool_value(await c.call_tool("device_state", {}))
        assert d["in_call"] is None and d["last_call"]["contact"] == "Mom"
        assert text(await c.call_tool("end_call", {})) == "There's no call to end."


async def test_ringing_timer_is_dismissed():
    from common.mcp_util import tool_value
    from mcp_servers.home import server
    server.state.set("ringing", [{"kind": "timer", "label": "1 minute timer", "at": time.time()}])
    async with Client(mcp) as c:
        d = tool_value(await c.call_tool("device_state", {}))
        assert d["ringing"][0]["label"] == "1 minute timer" and "now" in d and d["tz"]
        assert text(await c.call_tool("dismiss_alarm", {})) == "Stopped the 1 minute timer."
        assert tool_value(await c.call_tool("device_state", {}))["ringing"] == []
        assert text(await c.call_tool("dismiss_alarm", {})) == "Nothing is ringing."


async def test_ringing_stops_by_itself(monkeypatch):
    from common.mcp_util import tool_value
    from mcp_servers.home import server
    server.state.set("ringing", [{"kind": "alarm", "label": "7:00 AM", "at": time.time() - 61}])
    async with Client(mcp) as c:
        assert tool_value(await c.call_tool("device_state", {}))["ringing"] == []


async def test_web_search_cleans_answer_and_keeps_sources(monkeypatch):
    from common.mcp_util import tool_value
    from mcp_servers.home import server
    seen = {}

    async def fake(query):
        seen["q"] = query
        return {"choices": [{"message": {
            "content": "The **Lakers** won 112-104 last night ([espn.com](https://www.espn.com/nba/x)). See https://nba.com/y",
            "annotations": [{"type": "url_citation", "url_citation": {"url": "https://www.espn.com/nba/x"}},
                            {"type": "url_citation", "url_citation": {"url": "https://nba.com/y"}}]}}]}

    monkeypatch.setattr(server, "_openrouter_search", fake)
    monkeypatch.setitem(server.cfg, "search", {"enabled": True})
    async with Client(mcp) as c:
        out = text(await c.call_tool("web_search", {"query": "Lakers result last night"}))
        assert out.startswith("The Lakers won 112-104 last night") and "http" not in out and "**" not in out
        d = tool_value(await c.call_tool("device_state", {}))
        assert d["last_search"]["sources"] == ["espn.com", "nba.com"] and seen["q"] == "Lakers result last night"


async def test_web_search_off_and_errors(monkeypatch):
    import httpx
    from mcp_servers.home import server
    monkeypatch.setitem(server.cfg, "search", {"enabled": False})
    async with Client(mcp) as c:
        assert text(await c.call_tool("web_search", {"query": "x"})) == "Online search is turned off on this device."

    async def boom(query):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(server, "_openrouter_search", boom)
    monkeypatch.setitem(server.cfg, "search", {"enabled": True})
    async with Client(mcp) as c:
        assert "couldn't search online" in text(await c.call_tool("web_search", {"query": "x"}))
