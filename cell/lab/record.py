"""Keep what was said: one JSON line per record in the session's run folder, next to narration.txt.

    page.jsonl         every message on the page (the person, the robot, actions, the log)
    chat_llm.jsonl     every turn of the chat model: what it was given (the status) and what it answered
    ventral_llm.jsonl  every call of the ventral stream's LLM: the new messages (text only) and its answer

Nothing here may stop the robot: a failed write is dropped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from lab import viz


def folder() -> Path | None:
    log = getattr(viz, "LOG", None)
    return Path(log.name).parent if log is not None else None


def write(name: str, record: dict) -> None:
    try:
        d = folder()
        if d is None:
            return
        with open(d / name, "a") as f:
            f.write(json.dumps({"t": time.time(), **record}, default=str) + "\n")
    except Exception:
        pass


def text_only(messages: list) -> list:
    """Messages without pictures (data URLs are megabytes): text parts kept, an image becomes "[picture]"."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [p.get("text", "") if p.get("type") == "text" else "[picture]" for p in c]
        out.append({k: v for k, v in {**m, "content": c}.items() if k != "content" or v is not None})
    return out
