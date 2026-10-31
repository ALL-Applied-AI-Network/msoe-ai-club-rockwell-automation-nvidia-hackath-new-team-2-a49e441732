"""V8's mind: it talks, it remembers, and (stage two) it has eyes - look and inspect, nothing else.

One Pydantic AI agent. Its memory is the message history, kept whole. Its eyes are the dorsal
stream in v8/eyes.py: anonymous shapes carved from wrist-camera views. There is no naming stage
yet - it cannot know WHAT anything is, only WHERE something is and roughly its shape, and its
persona says so. No hands yet either.

    uv run python -m v8.agent "hello"      # one exchange from the terminal (no cell: talk only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PERSONA = """You are the mind of a FANUC ER-4iA robot arm in a small work cell at a hackathon,
talking with Joel, who built you. You have EYES but no hands: the look tool photographs the table
from wherever your wrist camera is and finds anonymous regions ("blobs"); the inspect tool flies
the camera around one blob and carves its 3D shape from several angles. You cannot name or
recognize anything - a blob is a shape, not a thing - and you cannot move or touch objects.
Say so plainly when asked for more than you can do. When Joel asks about the table, look; when a
shape matters, inspect it; report what the tools actually said, never what you imagine. Be direct
and brief; no filler, no pretending. You remember the whole conversation."""


@dataclass
class Deps:
    eyes: object | None = None       # v8.eyes.Eyes once a cell is connected


class Mind:
    """The conversation IS the memory: every exchange carries the full history back in."""

    def __init__(self, model: str | None = None):
        from pydantic_ai import Agent, RunContext
        self.model_name = model or os.getenv("V8_MODEL", "openai/gpt-6-luna")
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        m = OpenAIChatModel(self.model_name,
                            provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]))
        self.agent = Agent(m, instructions=PERSONA, deps_type=Deps)
        self.history: list = []

        @self.agent.tool
        def look(ctx: RunContext[Deps]) -> str:
            """Photograph the table from where the wrist camera is now; SAM proposes regions and
            each becomes or updates an anonymous blob. Returns what is known after this look."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to look at"
            return ctx.deps.eyes.look()

        @self.agent.tool
        def inspect(ctx: RunContext[Deps], blob: int) -> str:
            """Fly the wrist camera to three vantages around blob <blob> (one above, two low from
            different sides) and carve its volume from the intersection. Returns the refined shape."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to inspect"
            return ctx.deps.eyes.inspect(blob)

    def say(self, text: str, deps: Deps | None = None) -> str:
        r = self.agent.run_sync(text, deps=deps or Deps(), message_history=self.history)
        self.history = r.all_messages()
        return r.output

    def forget(self) -> None:
        self.history = []


if __name__ == "__main__":
    import sys
    print(Mind().say(sys.argv[1] if len(sys.argv) > 1 else "hello - who are you?"))
