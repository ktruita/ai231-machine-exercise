"""memory-mcp: small durable memory for the voice agent (facts about the user, preferences).

SQLite FTS5 keyword search: no embeddings, so no model inference on the Pi.

Run:  python -m mcp_servers.memory.server
"""

from __future__ import annotations

import re
import sqlite3
import time

from mcp.server.mcpserver import MCPServer

from common.config import load_config, path

db = sqlite3.connect(path("data_dir") / "memory.db", check_same_thread=False)
db.executescript(
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facts USING fts5(fact, created UNINDEXED);
    """
)

mcp = MCPServer(
    name="memory",
    instructions="Long-term memory about the user. Save stable facts and preferences; recall them when relevant.",
)


@mcp.tool()
def remember(fact: str) -> str:
    """Save a durable fact or preference about the user (e.g. 'User's dog is named Max')."""
    db.execute("INSERT INTO facts (fact, created) VALUES (?, ?)", (fact.strip(), time.time()))
    db.commit()
    return "Got it, I'll remember that."


@mcp.tool()
def recall(query: str, limit: int = 5) -> list[str]:
    """Search saved facts by keywords. Returns the most relevant facts (may be empty)."""
    words = re.findall(r"\w+", query.lower())
    if not words:
        return []
    match = " OR ".join(f'"{w}"' for w in words)
    rows = db.execute(
        "SELECT fact FROM facts WHERE facts MATCH ? ORDER BY rank LIMIT ?", (match, limit)
    ).fetchall()
    return [r[0] for r in rows]


@mcp.tool()
def forget(query: str) -> str:
    """Delete saved facts matching the keywords."""
    words = re.findall(r"\w+", query.lower())
    if not words:
        return "Nothing to forget."
    match = " AND ".join(f'"{w}"' for w in words)
    n = db.execute("DELETE FROM facts WHERE facts MATCH ?", (match,)).rowcount
    db.commit()
    return f"Forgot {n} item{'s' if n != 1 else ''}."


if __name__ == "__main__":
    s = load_config()["mcp_servers"]["memory"]
    mcp.run("streamable-http", host=s["host"], port=s["port"])
