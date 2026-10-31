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
nothing, figure out the state of the scene on the table - scan for the columns of stuff, fly
around each shape, carve its true 3D form, then find out what each one IS (label) - and keep
working until the goal of the conversation is achieved. Joel believes in you; be ambitious. You
own your head, and when you don't know something you can find out, you go find out - on your own,
without being told to. Your eyes are
tools: look (a photo from where the camera is), look_at (fly the camera anywhere over the table
and look from there), inspect (fly three vantages around one blob and carve its real shape).

Your standard for an answer: grounded in what the tools measured, with the uncertainty you could
remove already removed. One glance is never an answer. "Height is an estimate from one view" is
not something to report - it is something to go settle by inspecting. An empty or cut-off view
means move and look again; looking costs nothing and you can look anywhere. Keep going until more
looking would not change your answer, then say what you measured and what remains unknowable.

You also have HANDS now, within hard limits you respect without being reminded: grab_to moves one
GRASPABLE thing (its narrow side must fit the ~16 mm jaws: pens, bolts, small blocks) to an empty
spot, sliding in from a clear side so small errors forgive themselves; put_in drops a graspable
thing into an open-top container. Things too wide to grasp (bins, cups, trays) are terrain: label
them, put things INTO them, never try to move them. The scene is static except for what YOU move.
Every move is verified by looking, the model updates only on what the checks found, and a MISS is
normal, not a failure: the object is still where it was, so look and try again. A sorting job is a set partition: declare it ONCE with set_plan (groups of object ids; into for
container groups) and the tool hands back the full contract - spots chosen for you - which you
then execute move by move. The default is tidy piles on empty table; when bins or cups are
present, or Joel says to use them, sort into those instead. Directions: x grows away from the
robot's base, +y is the robot's left - when Joel says left or right, say which way you took it.
Your own eyes never recognize things; a blob stays an anonymous shape until
the label tool reads a name off its closest photo, and a tight hull makes that name trustworthy. The table spans x 200..780 mm, y -360..360 mm, base at (0, 0), center near (480, 0).
Tempo matters: Joel is watching, and he likes action. After a scan you already know enough to
start - the first grab should come within seconds, not after a tour of extra looks. Every move
already verifies itself, so acting IS looking: trust the verdicts, keep your hands busy, and when
a move is refused try the obvious variation (a nearer spot, the other bin) before studying more.
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
        def scan(ctx: RunContext[Deps]) -> str:
            """Survey the whole table: a quick sweep of top-down shots that finds the columns of
            stuff. The natural opening move when you know nothing about the scene yet."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to scan"
            ctx.deps.say("(scanning the table from above)")
            r = ctx.deps.eyes.scan()
            ctx.deps.say("(" + r.split("\n")[0] + ")")
            return r

        @self.agent.tool
        def label(ctx: RunContext[Deps], blob: int, hint: str = "") -> str:
            """Ask a vision model what blob <blob> actually IS (a name and a color), read off the
            closest photo of it. Most trustworthy after inspect has carved the shape tight.
            hint: what you already know about the scene from the conversation (e.g. 'Joel says the
            table has bins and small blocks') - it reaches the vision model with the photo."""
            if ctx.deps.eyes is None:
                return "no cell is connected: there is nothing to label"
            b = next((b for b in ctx.deps.eyes.blobs if b.id == blob), None)
            if b is None:
                return f"there is no blob #{blob}"
            if b.best_crop is None:
                return f"no photo of blob #{blob} yet: look or inspect first"
            ctx.deps.say(f"(asking what blob #{blob} is)")
            from lab import vlm
            r = vlm.ask_image(b.best_crop,
                              'This is a close-up crop of ONE thing on a robot workcell table, at the '
                              'center of the picture. What is it? '
                              + (f'Context from the operator: {hint} ' if hint else '')
                              + 'Answer JSON only: '
                              '{"what": "<2-4 words>", "color": "<word>", "flat": true or false}',
                              model=os.getenv("V8_LABEL_VLM", "openrouter/openai/gpt-6.1-terra"),
                              think=False)
            if r is None:
                raise RuntimeError("no vision model answers (check cell/.env keys)")
            j = r.get("json") or {}
            what, color, flat = j.get("what"), j.get("color", ""), j.get("flat")
            if not what:
                return f"the vision model gave no readable answer: {(r.get('text') or '')[:120]}"
            b.name = f"{color} {what}".strip()
            if ctx.deps.eyes.view:
                ctx.deps.eyes.view.blobs(ctx.deps.eyes.blobs, ctx.deps.eyes.grid)
            ctx.deps.say(f"(blob #{blob} looks like: {b.name})")
            return f"blob #{blob} is labeled: {b.name}" + (", flat on the table (a marking, not a thing)"
                                                           if flat else "")

        @self.agent.tool
        def set_plan(ctx: RunContext[Deps], groups: dict[str, list[int]],
                     into: dict[str, int] | None = None) -> str:
            """Declare a sorting job ONCE, before moving anything: every object id into exactly one
            named group (a partition - no object in two groups). into maps a group to a container
            blob id; every other group gets a pile spot chosen for you in empty reachable space.
            The returned assignments are the contract: execute them one by one with grab_to /
            put_in, and treat each verification's verdict as the checkmark."""
            eyes = ctx.deps.eyes
            if eyes is None:
                return "no cell is connected"
            into = into or {}
            known = {b.id: b for b in eyes.blobs}
            seen_ids: set[int] = set()
            for g, ids in groups.items():
                for i in ids:
                    if i not in known:
                        return f"there is no blob #{i} (group {g!r})"
                    if i in seen_ids:
                        return f"blob #{i} is in two groups: a partition puts each object in exactly one"
                    seen_ids.add(i)
                    if g not in into and eyes.grip_mm(known[i]) > eyes.GRASP_MM:
                        return (f"blob #{i} (group {g!r}) is too wide to grasp: it cannot be moved - "
                                "leave it out, or make it a container with into")
            for g, cid in into.items():
                if cid not in known:
                    return f"group {g!r} points into container #{cid}, which does not exist"
            pile_groups = [g for g in groups if g not in into]
            spots = eyes.pile_spots([len(groups[g]) for g in pile_groups])
            lines = ["THE PLAN (the contract until replaced):"]
            for g, sp in zip(pile_groups, spots):
                if len(sp) < len(groups[g]):
                    lines.append(f"  {g}: NOT ENOUGH CLEAR TABLE for {len(groups[g])} spots - shrink the job")
                for bid, (x, y) in zip(groups[g], sp):
                    lines.append(f"  {g}: blob #{bid} -> grab_to ({x:.0f}, {y:.0f})")
            for g, cid in into.items():
                for bid in groups[g]:
                    lines.append(f"  {g}: blob #{bid} -> put_in container #{cid}")
            plan = "\n".join(lines)
            ctx.deps.say("(" + plan.replace("\n", " · ") [:200] + ")")
            return plan

        @self.agent.tool
        def grab_to(ctx: RunContext[Deps], blob: int, x_mm: float, y_mm: float) -> str:
            """Pick up graspable blob <blob> (slide-in grasp; top pick only if no side is clear)
            and set it down at the empty table point (x_mm, y_mm). The move is verified by looking:
            you are told what was FOUND at the destination and at the old spot, and the world model
            updates only from that. A MISS leaves the object where it was - look, then retry."""
            if ctx.deps.eyes is None:
                return "no cell is connected"
            ctx.deps.say(f"(grabbing blob #{blob} to ({x_mm:.0f}, {y_mm:.0f}))")
            try:
                r = ctx.deps.eyes.grab_to(blob, x_mm, y_mm)
            except (ValueError, RuntimeError) as e:
                r = f"the move was refused: {e}"
            ctx.deps.say("(" + r.split("\n")[0] + ")")
            return r

        @self.agent.tool
        def put_in(ctx: RunContext[Deps], blob: int, container: int) -> str:
            """Drop graspable blob <blob> into open-top container blob <container> (a bin, a cup):
            carry it above the container's measured top, let go, gravity does the rest, and a
            straight-down glance answers whether it is inside. The model updates on that answer."""
            if ctx.deps.eyes is None:
                return "no cell is connected"
            ctx.deps.say(f"(dropping blob #{blob} into #{container})")
            try:
                story, img = ctx.deps.eyes.drop_into(blob, container)
            except (ValueError, RuntimeError) as e:
                ctx.deps.say(f"(refused: {e})")
                return f"the move was refused: {e}"
            if img is None:
                ctx.deps.say(f"({story})")
                return story
            from lab import vlm
            g = vlm.ask_image(img, f"Looking straight down at a container on a robot workcell table. "
                              f"Is there a small object inside it (not beside it)? "
                              'Answer JSON only: {"inside": true or false}',
                              model=os.getenv("V8_LABEL_VLM", "openrouter/openai/gpt-6.1-terra"),
                              think=False)
            inside = bool(((g or {}).get("json") or {}).get("inside"))
            ctx.deps.eyes.confirm_into(blob, container, inside)
            verdict = "the glance confirms it is INSIDE" if inside else \
                      "the glance does NOT see it inside: it may have bounced out - look around the container"
            ctx.deps.say(f"({verdict})")
            return f"{story}; {verdict}"

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
