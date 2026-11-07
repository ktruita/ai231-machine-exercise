"""Config + path helpers shared by the agent, MCP servers and benchmark."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    path = Path(os.environ.get("ME3_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a `paths:` entry relative to the repo root and make sure it exists."""
    p = Path(load_config()["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def server_url(name: str) -> str:
    s = load_config()["mcp_servers"][name]
    return f"http://{s['host']}:{s['port']}/mcp"


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable {name} (set it in .env)")
    return value
