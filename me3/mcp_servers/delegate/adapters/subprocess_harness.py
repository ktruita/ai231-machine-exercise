"""Generic harness adapter: run a harness CLI one-shot as a background subprocess.

Every harness (OpenCode, Hermes, ...) is treated as a blackbox: we hand it a goal
in a fresh workspace dir, wait for it to exit, and read SUMMARY.md (which the goal
prompt asks it to write) or fall back to the tail of its stdout.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

GOAL_SUFFIX = (
    "\n\nWork only inside the current directory. When you are finished, write a "
    "2-3 sentence plain-language summary of what you did and where the main output "
    "file is to SUMMARY.md. The summary will be read aloud, so avoid markdown and code."
)


@dataclass
class HarnessRun:
    proc: asyncio.subprocess.Process
    workdir: Path
    log_path: Path
    waiter: asyncio.Task = field(repr=False)


class SubprocessHarness:
    def __init__(self, name: str, argv_template: list[str], model: str):
        self.name = name
        self.argv_template = argv_template
        self.model = model

    def available(self) -> bool:
        return shutil.which(self.argv_template[0]) is not None

    async def start(self, goal: str, workdir: Path) -> HarnessRun:
        argv = [
            a.format(goal=goal + GOAL_SUFFIX, model=self.model, workdir=str(workdir))
            for a in self.argv_template
        ]
        log_path = workdir / ".harness.log"
        log = open(log_path, "wb")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=workdir, stdout=log, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, env=os.environ.copy(),
        )
        log.close()  # the child holds its own fd
        return HarnessRun(proc=proc, workdir=workdir, log_path=log_path,
                          waiter=asyncio.create_task(proc.wait()))

    @staticmethod
    def result(run: HarnessRun) -> str:
        summary = run.workdir / "SUMMARY.md"
        if summary.exists():
            return summary.read_text().strip()
        tail = run.log_path.read_text(errors="replace").strip()[-600:] if run.log_path.exists() else ""
        return tail or "The task finished but produced no summary."

    @staticmethod
    async def cancel(run: HarnessRun) -> None:
        if run.proc.returncode is None:
            run.proc.terminate()
            try:
                await asyncio.wait_for(run.waiter, 5)
            except asyncio.TimeoutError:
                run.proc.kill()
