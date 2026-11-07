"""home-mcp: the 19 "easy" intents from Dataset Schema Option B as MCP tools.

Tools are grouped (music, lights, ...) to keep the LLM's tool list short; every
call is logged with its Option-B intent label + slots for benchmark scoring.

Run:  python -m mcp_servers.home.server
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from mcp.server.mcpserver import MCPServer

from common.config import env, load_config, path
from mcp_servers.home.player import Player
from mcp_servers.home.state import HomeState

cfg = load_config()
TZ = ZoneInfo(cfg["location"]["timezone"])
state = HomeState(path("data_dir") / "home.db")
player = Player(path("music_dir"), path("data_dir") / "mpv.sock",
                browser=cfg.get("audio_out", "browser") == "browser")

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "a thunderstorm",
    96: "a thunderstorm with hail", 99: "a thunderstorm with hail",
}


RING_MAX_S = 60        # a ringing timer/alarm stops by itself after a minute
CALL_RING_S = 6        # simulated call: ringback for 6 s, then "connected"
CALL_MAX_S = 120       # simulated calls hang up by themselves after 2 minutes


async def _scheduler() -> None:
    """Fire due timers/alarms: an event for the voice agent to announce + a ringing entry for
    the UI (it beeps on the laptop until dismissed)."""
    while True:
        for kind, label in state.pop_due(time.time()):
            state.add_event(kind, f"Your {label} is done. It is ringing until the user says stop."
                            if kind == "timer" else f"Alarm: it's {label}. It is ringing until the user says stop.")
            state.set("ringing", (state.get("ringing") or []) + [{"kind": kind, "label": label, "at": time.time()}])
        await asyncio.sleep(0.5)


def _ringing() -> list[dict]:
    ringing = [r for r in (state.get("ringing") or []) if time.time() - r["at"] < RING_MAX_S]
    if ringing != (state.get("ringing") or []):
        state.set("ringing", ringing)
    return ringing


def _call() -> dict | None:
    c = state.get("in_call")
    if c and time.time() - c["since"] >= CALL_MAX_S:
        _finish_call(c)
        return None
    if c:
        c = dict(c, status="ringing" if time.time() - c["since"] < CALL_RING_S else "connected")
    return c


def _finish_call(c: dict) -> int:
    seconds = max(0, round(time.time() - c["since"] - CALL_RING_S))
    state.set("last_call", {"contact": c["contact"], "at": time.time(), "duration": seconds})
    state.set("in_call", None)
    return seconds


@contextlib.asynccontextmanager
async def lifespan(_server: MCPServer):
    task = asyncio.create_task(_scheduler())
    try:
        yield {}
    finally:
        task.cancel()
        await player.close()


mcp = MCPServer(
    name="home",
    instructions="Smart-home and assistant tools: music, weather, time, lights, "
    "temperature, calls/messages, timers, alarms, reminders. Results are short spoken confirmations.",
    lifespan=lifespan,
)


def _humanize_seconds(seconds: int) -> str:
    if seconds % 3600 == 0 and seconds >= 3600:
        n, unit = seconds // 3600, "hour"
    elif seconds % 60 == 0 and seconds >= 60:
        n, unit = seconds // 60, "minute"
    else:
        n, unit = seconds, "second"
    return f"{n} {unit}{'s' if n != 1 else ''}"


@mcp.tool()
async def music(
    action: Literal["play", "pause", "stop", "next", "volume_up", "volume_down"],
) -> str:
    """Control music playback: play, pause, stop, skip to next song, or raise/lower volume."""
    intent = {
        "play": "PLAY_MUSIC", "pause": "PAUSE", "stop": "STOP", "next": "NEXT",
        "volume_up": "VOLUME_UP", "volume_down": "VOLUME_DOWN",
    }[action]
    state.log_intent(intent, {})
    if action == "play":
        return await player.play()
    if action == "pause":
        return await player.pause()
    if action == "stop":
        return await player.stop()
    if action == "next":
        return await player.next()
    return await player.change_volume(10 if action == "volume_up" else -10)


@mcp.tool()
async def get_weather() -> str:
    """Current weather at the user's location."""
    state.log_intent("WEATHER", {})
    loc = cfg["location"]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": loc["timezone"],
                },
            )
            r.raise_for_status()
            cur = r.json()["current"]
    except httpx.HTTPError as e:
        return f"Sorry, I couldn't reach the weather service ({type(e).__name__})."
    desc = WMO_CODES.get(cur["weather_code"], "mixed conditions")
    state.set("weather", {"place": loc["name"], "temp": round(cur["temperature_2m"]),
                          "feels": round(cur["apparent_temperature"]), "desc": desc,
                          "code": cur["weather_code"], "wind": round(cur["wind_speed_10m"]),
                          "at": time.time()})  # for the UI's weather card
    return (
        f"In {loc['name']} it's {round(cur['temperature_2m'])} degrees with {desc}, "
        f"feels like {round(cur['apparent_temperature'])}."
    )


@mcp.tool()
def get_time() -> str:
    """Current local time and date."""
    state.log_intent("TIME", {})
    now = datetime.now(TZ)
    return f"It's {now.strftime('%-I:%M %p')} on {now.strftime('%A, %B %-d')}."


@mcp.tool()
def lights(
    action: Literal["on", "off", "brightness", "color"],
    brightness_percent: int | None = None,
    color: str | None = None,
) -> str:
    """Control the lights: turn on/off, set brightness (0-100 percent), or set color (e.g. red)."""
    if action == "on":
        state.log_intent("LIGHT_ON", {})
        state.set("lights_on", True)
        return "Lights are on."
    if action == "off":
        state.log_intent("LIGHT_OFF", {})
        state.set("lights_on", False)
        return "Lights are off."
    if action == "brightness":
        if brightness_percent is None:
            return "What brightness would you like, in percent?"
        pct = max(0, min(100, brightness_percent))
        state.log_intent("BRIGHTNESS", {"percent": pct})
        state.set("brightness", pct)
        state.set("lights_on", pct > 0)
        return f"Brightness set to {pct} percent."
    if not color:
        return "Which color would you like?"
    color = color.strip().lower()
    state.log_intent("COLOR", {"color": color})
    state.set("color", color)
    state.set("lights_on", True)
    return f"Lights changed to {color}."


@mcp.tool()
def set_temperature(degrees: int) -> str:
    """Set the thermostat / aircon temperature in degrees Celsius."""
    state.log_intent("TEMPERATURE", {"degrees": degrees})
    state.set("temperature", degrees)
    return f"Temperature set to {degrees} degrees."


SEARCH_SYSTEM = ("You answer for a voice assistant. Use the web results. Reply in at most two short spoken "
                 "sentences: plain text, no links, no markdown, no lists. Mention when the information is "
                 "from if it matters (e.g. 'as of this morning'). If the results don't answer it, say so. "
                 "Today is {today}.")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\(?https?://\S+\)?")


async def _openrouter_search(query: str) -> dict:
    """One web-grounded completion via OpenRouter's web plugin (Exa by default for this model)."""
    sc = cfg.get("search", {})
    async with httpx.AsyncClient(timeout=float(sc.get("timeout_s", 20))) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"},
            json={
                "model": sc.get("model", cfg.get("llm", {}).get("model")),
                "plugins": [{"id": "web", "engine": sc.get("engine", "exa"),
                             "max_results": int(sc.get("max_results", 3))}],
                "temperature": 0.2, "max_tokens": 160,
                "messages": [
                    {"role": "system", "content": SEARCH_SYSTEM.format(
                        today=datetime.now(TZ).strftime("%A %B %-d, %Y"))},
                    {"role": "user", "content": query},
                ],
            },
        )
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def web_search(query: str) -> str:
    """Look up CURRENT information online: news, sports scores, prices, recent events, opening hours,
    or anything the user asks you to look up / search. Not for general knowledge you already know.
    query: a short, complete search query (e.g. 'Lakers game result last night')."""
    if not cfg.get("search", {}).get("enabled", False):
        return "Online search is turned off on this device."
    try:
        data = await _openrouter_search(query)
        msg = data["choices"][0]["message"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        return f"Sorry, I couldn't search online right now ({type(e).__name__})."
    answer = _URL.sub("", _MD_LINK.sub(r"\1", msg.get("content") or "")).replace("**", "").strip()
    sources = []
    for a in msg.get("annotations") or []:
        url = (a.get("url_citation") or {}).get("url", "")
        host = re.sub(r"^www\.", "", url.split("/")[2]) if url.count("/") >= 2 else ""
        if host and host not in sources:
            sources.append(host)
    state.set("last_search", {"query": query, "answer": answer, "sources": sources[:3], "at": time.time()})
    return answer or "I couldn't find anything useful about that."


@mcp.tool()
def call(contact: str | None = None) -> str:
    """Start a phone call (simulated). Ask who to call if the contact is unknown."""
    state.log_intent("CALL", {"contact": contact} if contact else {})
    if not contact:
        return "Who would you like to call?"
    if _call():
        _finish_call(state.get("in_call"))
    state.set("in_call", {"contact": contact, "since": time.time()})
    return f"Calling {contact} now."


@mcp.tool()
def end_call() -> str:
    """Hang up / end the current phone call ("hang up", "stop the call", "end the call")."""
    c = _call()
    if not c:
        return "There's no call to end."
    seconds = _finish_call(c)
    return f"Call with {c['contact']} ended" + (f" after {seconds} seconds." if seconds else ".")


@mcp.tool()
def dismiss_alarm() -> str:
    """Stop a ringing timer or alarm ("stop", "okay", "turn it off" while it rings)."""
    ringing = _ringing()
    if not ringing:
        return "Nothing is ringing."
    state.set("ringing", [])
    return f"Stopped the {ringing[-1]['label']}." if ringing[-1]["kind"] == "timer" else "Alarm stopped."


@mcp.tool()
def send_message(contact: str | None = None, text: str | None = None) -> str:
    """Send a text message (simulated). Ask for the recipient and content if missing."""
    state.log_intent("MESSAGE", {k: v for k, v in {"contact": contact, "text": text}.items() if v})
    if not contact:
        return "Who should I send the message to?"
    if not text:
        return f"What should the message to {contact} say?"
    state.set("last_message", {"contact": contact, "text": text, "at": time.time()})
    return f"Message sent to {contact}."


@mcp.tool()
def set_timer(duration_seconds: int) -> str:
    """Start a countdown timer. Convert the spoken duration to seconds (e.g. '1 minute' -> 60)."""
    label = f"{_humanize_seconds(duration_seconds)} timer"
    state.log_intent("TIMER", {"duration_seconds": duration_seconds})
    state.schedule("timer", label, time.time() + duration_seconds)
    return f"Timer set for {_humanize_seconds(duration_seconds)}."


@mcp.tool()
def set_alarm(time_24h: str) -> str:
    """Set an alarm at a local time given as 24-hour HH:MM (e.g. '06:00', '21:00')."""
    try:
        hour, minute = (int(x) for x in time_24h.split(":"))
    except ValueError:
        return "Sorry, what time should the alarm be for?"
    now = datetime.now(TZ)
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    spoken = due.strftime("%-I:%M %p")
    state.log_intent("ALARM", {"time": f"{hour:02d}:{minute:02d}"})
    state.schedule("alarm", spoken, due.timestamp())
    day = "today" if due.date() == now.date() else "tomorrow"
    return f"Alarm set for {spoken} {day}."


@mcp.tool()
def create_reminder(task: str) -> str:
    """Save a reminder, e.g. 'drink water'."""
    task = task.strip().rstrip(".")
    state.log_intent("CREATE_REMINDER", {"task": task.lower()})
    state.add_reminder(task)
    return f"Okay, I'll remind you to {task}."


@mcp.tool()
def list_reminders() -> str:
    """List the user's saved reminders."""
    state.log_intent("LIST_REMINDERS", {})
    items = state.reminders()
    if not items:
        return "You don't have any reminders."
    if len(items) == 1:
        return f"You have one reminder: {items[0]}."
    return f"You have {len(items)} reminders: " + ", ".join(items[:-1]) + f", and {items[-1]}."


# --- internal tools (used by the agent runtime / benchmark, hidden from the LLM) ---

@mcp.tool()
def poll_events() -> list[dict]:
    """[internal] Return undelivered events (fired timers/alarms) and mark them delivered."""
    return state.take_events()


@mcp.tool()
def intent_log_since(ts: float) -> list[dict]:
    """[internal] Intents logged since unix timestamp `ts` (benchmark scoring)."""
    return state.intents_since(ts)


@mcp.tool()
def music_track_ended() -> str:
    """[internal] The web UI finished playing a track; advance to the next one."""
    player.track_ended()
    return player.info()["music_title"] or ""


@mcp.tool()
def device_state() -> dict:
    """[internal] Current simulated device state."""
    return state.devices() | player.info() | {
        "ringing": _ringing(),
        "in_call": _call(),
        "now": time.time(),                      # the UI clock shows the Pi's time
        "tz": cfg["location"]["timezone"],
        "schedules": state.pending_schedules(),
        "reminders": state.reminders(),
    }


INTERNAL_TOOLS = {"poll_events", "intent_log_since", "device_state", "music_track_ended"}


if __name__ == "__main__":
    s = cfg["mcp_servers"]["home"]
    mcp.run("streamable-http", host=s["host"], port=s["port"])
