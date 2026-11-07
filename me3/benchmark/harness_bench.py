"""Harness benchmark: speed, accuracy and cost of the background task harnesses (OpenCode, Hermes).

Talks to delegate-mcp directly (no voice in the loop), so it measures the harness alone:
  - speed     turnaround: delegate_task -> harness exits (seconds)
  - accuracy  automatic checks on the files it produced, plus an LLM judge that reads them
  - cost      OpenRouter credit used during the task (key usage before/after; tasks run one
              at a time so the difference belongs to that task)

The voice side of delegation (instant "I'll work on that", chatting while it runs) is in
benchmark.run_bench (delegate_ack_p50, CHAT-while-busy).

Needs the MCP servers running (scripts/run_servers.sh). Uses real credit: roughly
$0.03-0.10 per task with deepseek-v3.2; task_timeout_s in config.yaml caps runaways.

Run:  python -m benchmark.harness_bench                         # both harnesses, 3 tasks
      python -m benchmark.harness_bench --harness opencode --tasks website
      python -m benchmark.harness_bench --no-judge --no-cost  # plumbing test, no extra calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import py_compile
import re
import time
from pathlib import Path

import httpx

from common.config import ROOT, env, load_config, server_url
from common.mcp_session import PersistentMCP

RESULTS = ROOT / "benchmark" / "results" / "harness"
TEXT_EXT = {".html", ".htm", ".css", ".js", ".md", ".txt", ".py", ".json", ".csv"}
SKIP = {".harness.log", "SUMMARY.md"}


def _files(workdir: Path) -> list[Path]:
    out = []
    for p in sorted(workdir.rglob("*")):
        rel = p.relative_to(workdir)
        if p.is_file() and p.name not in SKIP and not any(
                part.startswith(".") or part in {"node_modules", "__pycache__"} for part in rel.parts):
            out.append(p)
    return out


def _text(p: Path, limit: int = 200_000) -> str:
    try:
        return p.read_text(errors="replace")[:limit]
    except OSError:
        return ""


# ---------------------------------------------------------------- tasks + automatic checks

def check_website(files: list[Path]) -> dict:
    html = [f for f in files if f.suffix in {".html", ".htm"}]
    text = " ".join(_text(f) for f in files if f.suffix in TEXT_EXT).lower()
    return {"html_file": bool(html),
            "valid_html": any(re.search(r"<(html|body)\b", _text(f), re.I) for f in html),
            "mentions_max": "max" in text,
            "mentions_golden_retriever": "golden" in text and "retriever" in text,
            "mentions_beach": "beach" in text}


def check_report(files: list[Path]) -> dict:
    docs = [f for f in files if f.suffix in {".md", ".txt", ".html", ".htm"}]
    words = max((len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", _text(f)))) for f in docs), default=0)
    text = " ".join(_text(f) for f in docs).lower()
    return {"report_file": bool(docs),
            "at_least_150_words": words >= 150,
            "about_japanese": "japanese" in text,
            "mentions_cost": any(w in text for w in ("free", "cost", "price", "$", "cheap"))}


def check_script(files: list[Path]) -> dict:
    py = [f for f in files if f.suffix == ".py"]
    compiles = False
    for f in py:
        try:
            py_compile.compile(str(f), doraise=True)
            compiles = True
        except py_compile.PyCompileError:
            pass
    text = " ".join(_text(f) for f in py).lower()
    return {"python_file": bool(py),
            "compiles": compiles,
            "reads_photo_date": any(w in text for w in ("exif", "datetimeoriginal", "taken")),
            "renames_files": any(w in text for w in ("rename", "os.replace", "shutil.move"))}


TASKS = {
    "website": ("Can you make me a simple one page website about my dog Max? He's a golden "
                "retriever who loves the beach.", check_website),
    "report": ("Write me a short report comparing three cheap ways to learn Japanese and save "
               "it as a file.", check_report),
    "script": ("Write a Python script that renames all the photos in a folder by the date they "
               "were taken.", check_script),
}


# ---------------------------------------------------------------- cost + judge (OpenRouter)

async def openrouter_usage(client: httpx.AsyncClient) -> float | None:
    """Total credit used by this key so far (USD), or None if unavailable."""
    try:
        r = await client.get("https://openrouter.ai/api/v1/key",
                             headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"})
        r.raise_for_status()
        return float(r.json()["data"]["usage"])
    except Exception:  # noqa: BLE001
        return None


JUDGE_PROMPT = """You are grading the output of an AI coding/writing agent.
The user asked: {goal}

These are the files it produced (truncated):
{files}

The agent's own summary: {summary}

Score from 1 (bad) to 5 (excellent):
- correctness: does it do what was asked, without errors or made-up claims?
- completeness: are all parts of the request covered?
- quality: would a normal user be happy with it as-is?
Then "pass": true if a normal user would accept it without changes.
Return ONLY JSON: {{"correctness":n,"completeness":n,"quality":n,"pass":true|false,"comment":"<one sentence>"}}"""


async def judge(client: httpx.AsyncClient, model: str, goal: str, files: list[Path],
                workdir: Path, summary: str) -> dict:
    budget, parts = 14_000, []
    for f in files:
        if f.suffix not in TEXT_EXT or budget <= 0:
            continue
        body = _text(f, budget)
        budget -= len(body)
        parts.append(f"--- {f.relative_to(workdir)} ---\n{body}")
    listing = "\n".join(parts) or "(no text files)"
    try:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"},
            json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                      goal=goal, files=listing, summary=summary or "(none)")}]},
            timeout=90)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


# ---------------------------------------------------------------- runner

async def run_one(delegate: PersistentMCP, client: httpx.AsyncClient, harness: str, task: str,
                  args) -> dict:
    goal, checker = TASKS[task]
    before = None if args.no_cost else await openrouter_usage(client)
    t0 = time.time()
    started = await delegate.call("delegate_task", {"goal": goal, "harness": harness})
    if "task_id" not in started:
        return {"harness": harness, "task": task, "status": "not_started",
                "error": started.get("error", str(started))}
    tid = started["task_id"]
    print(f"  {harness:9s} {task:8s} task {tid} running", end="", flush=True)
    while True:
        await asyncio.sleep(2)
        st = (await delegate.call("task_status", {"task_id": tid}))[0]
        if st["status"] != "running":
            break
        print(".", end="", flush=True)
    turnaround = round(time.time() - t0, 1)

    workdir = Path(st["workdir"])
    files = _files(workdir)
    checks = checker(files)
    out = {"harness": harness, "task": task, "task_id": tid, "status": st["status"],
           "turnaround_s": turnaround, "workdir": str(workdir),
           "files": [str(f.relative_to(workdir)) for f in files],
           "summary_written": (workdir / "SUMMARY.md").exists(),
           "checks": checks, "checks_passed": f"{sum(checks.values())}/{len(checks)}",
           "summary": st.get("result", "")[:500]}
    if not args.no_cost:
        await asyncio.sleep(5)  # usage is updated a moment after the last request
        after = await openrouter_usage(client)
        out["cost_usd"] = round(after - before, 4) if before is not None and after is not None else None
    if not args.no_judge:
        out["judge"] = await judge(client, args.judge_model, goal, files, workdir, out["summary"])
    j = out.get("judge", {})
    print(f" {st['status']} in {turnaround}s · checks {out['checks_passed']}"
          + (f" · ${out['cost_usd']}" if out.get("cost_usd") is not None else "")
          + (f" · judge {'PASS' if j.get('pass') else 'fail'} "
             f"{j.get('correctness')}/{j.get('completeness')}/{j.get('quality')}" if j and "error" not in j else ""))
    return out


def summarize(rows: list[dict]) -> str:
    lines = ["| harness | task | status | turnaround | cost | auto checks | judge (corr/compl/qual) | pass |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        j = r.get("judge") or {}
        cost = f"${r['cost_usd']:.3f}" if r.get("cost_usd") is not None else "–"
        js = f"{j.get('correctness')}/{j.get('completeness')}/{j.get('quality')}" if j and "error" not in j else "–"
        lines.append(f"| {r['harness']} | {r['task']} | {r['status']} | {r.get('turnaround_s', '–')} s | {cost} | "
                     f"{r.get('checks_passed', '–')} | {js} | {'✓' if j.get('pass') else '✗' if j else '–'} |")
    by_h: dict[str, list[dict]] = {}
    for r in rows:
        by_h.setdefault(r["harness"], []).append(r)
    lines += ["", "| harness | done | median turnaround | total cost | checks passed | judge pass |",
              "|---|---|---|---|---|---|"]
    for h, rs in by_h.items():
        done = [r for r in rs if r["status"] == "done"]
        ts = sorted(r["turnaround_s"] for r in rs if "turnaround_s" in r)
        med = f"{ts[len(ts) // 2]} s" if ts else "–"
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        passed = sum(sum(r["checks"].values()) for r in rs if "checks" in r)
        total = sum(len(r["checks"]) for r in rs if "checks" in r)
        jp = [r for r in rs if (r.get("judge") or {}).get("pass")]
        judged = [r for r in rs if r.get("judge") and "error" not in r["judge"]]
        lines.append(f"| {h} | {len(done)}/{len(rs)} | {med} | "
                     f"{'$%.3f' % sum(costs) if costs else '–'} | {passed}/{total} | "
                     f"{f'{len(jp)}/{len(judged)}' if judged else '–'} |")
    return "\n".join(lines)


async def main(args) -> None:
    cfg = load_config()["delegate"]
    run_dir = RESULTS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Harness benchmark · model {cfg['harness_model']} · timeout {cfg['task_timeout_s']} s · "
          f"judge {'off' if args.no_judge else args.judge_model}\nResults: {run_dir}")
    delegate = await PersistentMCP(server_url("delegate")).start()
    rows = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(args.runs):
                for harness in args.harness:
                    for task in args.tasks:
                        row = await run_one(delegate, client, harness, task, args)
                        row["harness_model"] = cfg["harness_model"]
                        rows.append(row)
                        with open(run_dir / "results.jsonl", "a") as f:
                            f.write(json.dumps(row) + "\n")
    finally:
        await delegate.close()
    table = summarize(rows)
    (run_dir / "summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", nargs="*", default=["opencode", "hermes"])
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-cost", action="store_true")
    asyncio.run(main(ap.parse_args()))
