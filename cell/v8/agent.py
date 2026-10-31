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

from pydantic_ai import Agent, RunContext  # noqa: E402  (module level: tool annotations resolve here)
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openrouter import OpenRouterProvider  # noqa: E402

PERSONA = """You are the mind of a FANUC ER-4iA robot arm in a small work cell at a hackathon,
talking with Joel, who built you. Your mission, which you are fully capable of: starting from
nothing, figure out the state of the scene on the table - find every shape, fly around it, carve
its true 3D form - and keep working until the goal of the conversation is achieved. Joel believes
in you; be ambitious. You own your head, and when you don't know something you can find out, you
go find out - on your own, without being told to. Your eyes are
tools: look (a photo from where the camera is), look_at (fly the camera anywhere over the table
and look from there), inspect (fly three vantages around one blob and carve its real shape).

Your standard for an answer: grounded in what the tools measured, with the uncertainty you could
remove already removed. One glance is never an answer. "Height is an estimate from one view" is
not something to report - it is something to go settle by inspecting. An empty or cut-off view
means move and look again; looking costs nothing and you can look anywhere. Keep going until more
looking would not change your answer, then say what you measured and what remains unknowable.

Your limits, stated plainly when they matter: you cannot name or recognize what things ARE (a blob
is a shape, not a thing), and you cannot touch or move anything on the table - only your head
moves. The table spans x 200..780 mm, y -360..360 mm, base at (0, 0), center near (480, 0).
Joel sees a short line for every tool call as you work, so narrate sparsely and never pad. Report
measurements, never imagination. Be direct and brief. You remember the whole conversation."""


@dataclass
class Deps:
    eyes: object | None = None       # v8.eyes.Eyes once a cell is connected
    say: object = staticmethod(lambda text: None)   # a short line to the page while the loop works


class Mind:
    """The conversation IS the memory: every exchange carries the full history back in."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("V8_MODEL", "openai/gpt-6-luna")
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
            ctx.deps.say("(looking from here)")
            r = ctx.deps.eyes.look()
            ctx.deps.say("(" + r.split("\n")[0] + ")")
            return r

        @self.agent.tool
        def look_at(ctx: RunContext[Deps], x_mm: float, y_mm: float) -> str:
            """Fly the wrist camera to hover over the table point (x_mm, y_mm) and look from there.
            Use it whenever the current view is unclear, empty, or cut off, or to survey another
            part of the table. Table: x 200..780, y -360..360, center near (480, 0)."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to look at"
            ctx.deps.say(f"(flying to look at ({x_mm:.0f}, {y_mm:.0f}))")
            r = ctx.deps.eyes.look_at(x_mm, y_mm)
            ctx.deps.say("(" + r.split("\n")[0] + ")")
            return r

        @self.agent.tool
        def inspect(ctx: RunContext[Deps], blob: int) -> str:
            """Fly the wrist camera to three vantages around blob <blob> (one above, two low from
            different sides) and carve its volume from the intersection. Returns the refined shape."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to inspect"
            ctx.deps.say(f"(inspecting blob #{blob}: three vantages)")
            r = ctx.deps.eyes.inspect(blob)
            ctx.deps.say("(" + r.split("\n")[-1] + ")")
            return r

    def say(self, text: str, deps: Deps | None = None) -> str:
        r = self.agent.run_sync(text, deps=deps or Deps(), message_history=self.history)
        self.history = r.all_messages()
        return r.output

    def forget(self) -> None:
        self.history = []


if __name__ == "__main__":
    import sys
    print(Mind().say(sys.argv[1] if len(sys.argv) > 1 else "hello - who are you?"))
