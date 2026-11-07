"""Music playback: real mpv (JSON IPC) when available, otherwise a simulated player.

Browser mode (audio_out: browser): the Pi only keeps the player state (track, playing,
volume); the web UI on the laptop plays the actual file from /music/<name> and follows
that state, so music comes out of the laptop speakers like the agent's voice.

Playing audio files is not model inference, so it is fine on the Pi.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".wav", ".m4a", ".opus"}


class Player:
    def __init__(self, music_dir: Path, socket_path: Path, browser: bool = False):
        self.music_dir = music_dir
        self.socket_path = socket_path
        self.proc: asyncio.subprocess.Process | None = None
        self.browser = browser
        self.simulated = browser or shutil.which("mpv") is None
        self.playing = False
        self.stopped = True
        self.volume = 60
        self.track_index = 0

    def tracks(self) -> list[Path]:
        return sorted(p for p in self.music_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)

    async def _ensure_mpv(self) -> None:
        if self.simulated or (self.proc and self.proc.returncode is None):
            return
        self.socket_path.unlink(missing_ok=True)
        self.proc = await asyncio.create_subprocess_exec(
            "mpv", "--idle=yes", "--no-video", "--no-terminal",
            f"--volume={self.volume}", f"--input-ipc-server={self.socket_path}",
        )
        for _ in range(50):  # wait up to ~2.5s for the socket
            if self.socket_path.exists():
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("mpv did not open its IPC socket")

    async def _cmd(self, *command) -> None:
        if self.simulated:
            return
        await self._ensure_mpv()
        reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
        writer.write((json.dumps({"command": list(command)}) + "\n").encode())
        await writer.drain()
        await reader.readline()
        writer.close()

    def _track_name(self) -> str:
        tracks = self.tracks()
        if not tracks:
            return "a demo track"
        return tracks[self.track_index % len(tracks)].stem

    async def play(self) -> str:
        tracks = self.tracks()
        if tracks and not self.playing:
            await self._cmd("loadfile", str(tracks[self.track_index % len(tracks)]))
            for t in tracks[self.track_index + 1:] + tracks[: self.track_index]:
                await self._cmd("loadfile", str(t), "append")
        await self._cmd("set_property", "pause", False)
        self.playing, self.stopped = True, False
        return f"Playing {self._track_name()}."

    async def pause(self) -> str:
        await self._cmd("set_property", "pause", True)
        self.playing = False
        return "Paused."

    async def stop(self) -> str:
        await self._cmd("stop")
        self.playing, self.stopped = False, True
        return "Stopped the music."

    async def next(self) -> str:
        self.track_index += 1
        if self.playing:
            await self._cmd("playlist-next", "force")
        else:
            self.playing = False
            return await self.play()
        return f"Skipping to {self._track_name()}."

    async def change_volume(self, delta: int) -> str:
        self.volume = max(0, min(100, self.volume + delta))
        await self._cmd("set_property", "volume", self.volume)
        return f"Volume at {self.volume} percent."

    def track_ended(self) -> None:
        """The browser finished a track: move on to the next one (and keep playing)."""
        if self.tracks():
            self.track_index = (self.track_index + 1) % len(self.tracks())

    def info(self) -> dict:
        """State for the UI. music_file is the file the browser should play."""
        tracks = self.tracks()
        current = tracks[self.track_index % len(tracks)] if tracks else None
        return {
            "music_playing": self.playing,
            "music_stopped": self.stopped,
            "music_title": current.stem if current else None,
            "music_file": str(current.relative_to(self.music_dir)) if current else None,
            "music_browser": self.browser,
            "volume": self.volume,
        }

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            await self.proc.wait()
