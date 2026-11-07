"""delegate-mcp: hand complex tasks to an agent harness (OpenCode, Hermes, ...) in the background.

`delegate_task` returns a task id immediately so the voice conversation never blocks;
the agent runtime polls `poll_events` and announces results when tasks finish.
`continue_task` runs a follow-up ("make it blue") in the SAME folder, so the harness edits
its earlier work instead of starting over. `task_status` reports live progress (files made
so far and the harness's latest step).

Run:  python -m mcp_servers.delegate.server
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer

from common.config import load_config, path
from mcp_servers.delegate.adapters.subprocess_harness import HarnessRun, SubprocessHarness

cfg = load_config()["delegate"]
HARNESSES = {
    name: SubprocessHarness(name, spec["argv"], cfg["harness_model"])
    for name, spec in cfg["harnesses"].items()
}


@dataclass
class Task:
    id: str
    goal: str
    harness: str
    created: float = field(default_factory=time.time)
    finished: float | None = None
    status: Literal["running", "done", "failed", "cancelled"] = "running"
    result: str = ""
    run: HarnessRun | None = None
    parent: str | None = None  # set for follow-ups: the task whose folder this one edits

    def public(self) -> dict:
        out = {
            "task_id": self.id, "goal": self.goal, "harness": self.harness, "status": self.status,
            "created": self.created,
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
            "result": self.result, "workdir": str(self.run.workdir) if self.run else None,
            "parent": self.parent,
        }
        if self.run:
            out.update(progress(self.run))
        return out


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def progress(run: HarnessRun) -> dict:
    """Live view of a harness run for "how's it going?": files so far + its latest step."""
    found: list[tuple[float, str]] = []
    try:
        for root, dirs, names in os.walk(run.workdir):
            # prune instead of walking into them: node_modules can hold thousands of files
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv"}]
            for n in names:
                if n.startswith(".") or n == "SUMMARY.md":
                    continue
                full = os.path.join(root, n)
                found.append((os.path.getmtime(full), os.path.relpath(full, run.workdir)))
                if len(found) > 500:
                    break
    except OSError:
        pass
    files = [rel for _, rel in sorted(found, reverse=True)]
    last = ""
    try:
        tail = run.log_path.read_bytes()[-4000:].decode(errors="replace")
        lines = [_ANSI.sub("", l).strip(" >←→·-*") for l in tail.splitlines()]
        last = next((l for l in reversed(lines) if len(l) > 3), "")[:120]
    except OSError:
        pass
    return {"files_so_far": files[:10], "last_step": last}


tasks: dict[str, Task] = {}
events: list[dict] = []
_watchers: set[asyncio.Task] = set()


async def _watch(task: Task) -> None:
    try:
        code = await asyncio.wait_for(task.run.waiter, cfg["task_timeout_s"])
        if task.status == "cancelled":
            return
        await asyncio.sleep(0.3)  # let helper processes finish writing SUMMARY.md
        task.status = "done" if code == 0 else "failed"
        task.result = SubprocessHarness.result(task.run)
    except asyncio.TimeoutError:
        await SubprocessHarness.cancel(task.run)
        task.status, task.result = "failed", "The task timed out."
    except asyncio.CancelledError:
        return
    task.finished = time.time()
    events.append({"kind": "task_" + task.status, **task.public()})


@contextlib.asynccontextmanager
async def lifespan(_server: MCPServer):
    try:
        yield {}
    finally:
        for t in tasks.values():
            if t.run and t.status == "running":
                await SubprocessHarness.cancel(t.run)


mcp = MCPServer(
    name="delegate",
    instructions="Delegate long, multi-step tasks (build a website, write a report, research, "
    "coding) to a background agent harness. Returns immediately; results arrive later.",
    lifespan=lifespan,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


DUPLICATE_WINDOW_S = 60.0
NOTE = ("The user already heard 'I'll work on that in the background'. Don't repeat it; "
        "reply with a few words at most, or just continue the conversation.")


async def _launch(goal: str, harness_goal: str, name: str, workdir: Path,
                  parent: str | None = None) -> Task:
    task = Task(id=uuid.uuid4().hex[:8], goal=goal, harness=name, parent=parent)
    task.run = await HARNESSES[name].start(harness_goal, workdir)
    tasks[task.id] = task
    watcher = asyncio.create_task(_watch(task))
    _watchers.add(watcher)
    watcher.add_done_callback(_watchers.discard)
    return task


def _normalize(goal: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", goal.lower()))


@mcp.tool()
async def delegate_task(goal: str, harness: Literal["auto", "opencode", "hermes"] = "auto") -> dict:
    """Start a NEW complex, multi-step task in the background (e.g. 'create a website about my
    dog', 'write a report on ...'). Describe the goal completely. Returns a task_id right away;
    tell the user you're working on it and keep talking. You'll be notified when it finishes.
    To change something an earlier task made, use continue_task instead."""
    name = cfg["default_harness"] if harness == "auto" else harness
    if not HARNESSES[name].available():
        return {"status": "failed", "error": f"Harness '{name}' is not installed on this device."}
    # guard: the same request again within a minute is a repeat, not a new job (saves credit)
    for t in tasks.values():
        if _normalize(t.goal) == _normalize(goal) and time.time() - t.created < DUPLICATE_WINDOW_S:
            return {"task_id": t.id, "status": t.status, "harness": t.harness, "duplicate": True,
                    "note": "This task was already started a moment ago; don't start it again. " + NOTE}
    task_id = uuid.uuid4().hex[:8]
    workdir = path("workspaces_dir") / f"{task_id}-{_slug(goal)}"
    workdir.mkdir(parents=True)
    task = await _launch(goal, goal, name, workdir)
    return {"task_id": task.id, "status": "running", "harness": name, "note": NOTE}


FOLLOW_UP = ("This folder already contains your earlier work for this request: {original}\n"
             "Earlier result: {result}\n\n"
             "Now do this change: {instruction}\n"
             "Edit the existing files in place; do not start over or create a new project.")


@mcp.tool()
async def continue_task(instruction: str, task_id: str | None = None) -> dict:
    """Change or extend something an earlier background task made, in the same folder
    (e.g. 'make the website blue', 'add a photo section', 'also add a conclusion').
    task_id: the task to continue; omit it to continue the most recent task.
    Returns right away like delegate_task; you'll be notified when it finishes."""
    if task_id and task_id not in tasks:
        return {"status": "failed", "error": f"No task with id {task_id}."}
    if not tasks:
        return {"status": "failed", "error": "There is no earlier task to continue; use delegate_task."}
    base = tasks[task_id] if task_id else max(tasks.values(), key=lambda t: t.created)
    busy = [t for t in tasks.values() if t.status == "running" and t.run and base.run
            and t.run.workdir == base.run.workdir]
    if busy:
        return {"status": "failed", "error": "That task is still running. Wait until it finishes, "
                "then ask again (or cancel it first).", "task_id": busy[0].id}
    # follow-ups of follow-ups still describe the very first request
    root = base
    while root.parent and root.parent in tasks:
        root = tasks[root.parent]
    harness_goal = FOLLOW_UP.format(original=root.goal, result=base.result or "(none)",
                                    instruction=instruction)
    (base.run.workdir / "SUMMARY.md").unlink(missing_ok=True)  # the follow-up writes a fresh one
    task = await _launch(instruction, harness_goal, base.harness, base.run.workdir, parent=base.id)
    return {"task_id": task.id, "status": "running", "harness": base.harness,
            "continues": base.id, "note": NOTE}


@mcp.tool()
def task_status(task_id: str | None = None) -> list[dict]:
    """Status of one task (by id) or of all tasks, newest first."""
    selected = [tasks[task_id]] if task_id in tasks else sorted(
        tasks.values(), key=lambda t: t.created, reverse=True
    )
    return [t.public() for t in selected]


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """Cancel a running background task."""
    task = tasks.get(task_id)
    if not task or task.status != "running":
        return "No running task with that id."
    # Mark first: the watcher wakes as soon as the process exits and must see this.
    task.status, task.finished, task.result = "cancelled", time.time(), "Cancelled by the user."
    await SubprocessHarness.cancel(task.run)
    return "Cancelled."


@mcp.tool()
def poll_events() -> list[dict]:
    """[internal] Task completion events not yet delivered to the voice agent."""
    out = events[:]
    events.clear()
    return out


INTERNAL_TOOLS = {"poll_events"}


if __name__ == "__main__":
    s = load_config()["mcp_servers"]["delegate"]
    mcp.run("streamable-http", host=s["host"], port=s["port"])
