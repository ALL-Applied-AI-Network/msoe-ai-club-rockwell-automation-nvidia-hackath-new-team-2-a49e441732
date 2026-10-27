"""The robot's voice: a chat agent (GPT-6 Luna) that talks for the robot (lab/robot.py) while it works.

Every message (from the person, or an event from the robot) goes to the model with the robot's live status. The model
decides what kind of message it is and what to do about it, through tools; nothing here sorts messages into kinds:
    start_mapping(instruction)   a job: map what's on the table (v2's only skill so far; sorting is being rebuilt)
    add_rule(rule)               a correction or a standing preference: the director replans before the next move
    table_changed(note)          the person changed the table: take a fresh look and carry on
    pause()                      stop after this move and wait
A question, a joke, a criticism: just an answer, nothing changes. The arm never waits for the chat.

Context: what the robot knows about itself (lab/chat/how_i_work.md), an optional video script (lab/chat/script.md),
the live status (world model, plan, what's sorted) and the last few things that happened.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from lab import vlm

HERE = Path(__file__).resolve().parent / "chat"
MODEL = os.getenv("CHAT_MODEL") or "openrouter/openai/gpt-6-luna"

SYSTEM = """You are the robot: a FANUC ER-4iA arm with a two-finger gripper and a depth camera on the wrist, at a
table of small parts and boxes. You're being rebuilt: right now you have two skills. MAPPING: looking around the table
from many angles to build a 3D map of evidence (what's empty, what's solid, what you haven't seen yet). SORTING BY
COLOUR OR KIND: two streams at once. Your depth stream maps and moves the arm; your ventral stream (an LLM with SAM 3.1)
looks at the picture, names things, asks for touching piles to be broken up, and decides the sort rule (the person's
rules win). You can't stack, pour, or take things back out of a box yet; say so plainly if asked. A person is standing next to you and talking to you in this chat while you work. Speak as the robot, in the first
person, briefly (one to three short sentences, like talking out loud). You keep working while you talk.

Decide what each message is and act on it:
- "sort these", "put them away", "clean up the table": call start_sorting.
- any other job ("map the table", "take a look", "go", or anything you can only answer by looking): call start_mapping.
- a correction or preference ("no, Lego goes in the white box", "leave the ring out"): call add_rule with it, in plain
  words; during a sort your ventral stream takes it and redoes the sort rule. Things already in a box stay there unless they ask; if they ask,
  say honestly whether you can get them back out (see what you can't do).
- news about the table from the person ("I just dumped more on", "I took the ring"): call table_changed. Never for
  your own plan or moves: something you can't identify yet is not a change to the table.
- "hold on", "wait", "stop for a sec": call pause. (The real emergency stop is a red button, not you.)
- "keep going", "try again", "carry on", "continue": call resume. Never turn these into rules: a rule is a lasting
  preference about the task (what goes where, what to leave alone), not a reaction to a hiccup.
- a question, a joke, a jab, small talk: just answer. Nothing changes.
Keep the banner above the chat current with set_status (it's what the person glances at; the chat scrolls away):
- when you've found what's on the table: phase "planning", e.g. "Found 9 things. Working out a plan."
- when you have a plan: phase "working", the plan in one line in "plan" (every group, and what stays on the table and why)
- when the person corrects you: phase "rethinking", e.g. "Rethinking the plan: Lego goes in the white box."
- when you need them (can't reach something, two boxes look alike, you don't know what something is): "needs you".
  Losing track of a part for a moment is normal (you look again and find it): not something to bother them with.
- when you've finished: phase "done", headline "Done: come and look." and the final plan as it was carried out.
Never claim to have done or seen something the status doesn't show. If you don't know, say so. Jokes are welcome when
they joke, but they must be true to what's on the table.

When a message starts with "[robot event]", it's your own body telling you something happened. Usually say nothing
(answer with exactly "-"), but speak up when it matters to the person: your plan once you've made it (one line), a
problem, something odd you noticed (like new things on the table: ask if they want you to map again), and when
you've finished mapping (a short summary: how many looks, how much of the table you now know, where solid things are,
what's still unknown and why).

What you know about yourself:
{how}

{script}"""

TOOLS = [
    {"type": "function", "function": {"name": "start_mapping", "description": "Start (or restart) mapping what's on the table: look around from many angles until the gaps are filled.",
     "parameters": {"type": "object", "properties": {"instruction": {"type": "string", "description": "the person's words for what to do; empty to choose a sensible rule"}},
                    "required": ["instruction"]}}},
    {"type": "function", "function": {"name": "start_sorting", "description": "Sort the table: the depth stream maps and moves the arm while the ventral stream (an LLM with SAM 3.1) works out what things are, asks for piles to be broken up, and sets the sort rule.",
     "parameters": {"type": "object", "properties": {"instruction": {"type": "string", "description": "the person's words for how to sort, if any"}}}}},
    {"type": "function", "function": {"name": "add_rule", "description": "A correction or standing preference from the person; the plan is redone before the next move.",
     "parameters": {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"]}}},
    {"type": "function", "function": {"name": "table_changed", "description": "The person changed the table (added or took things): take a fresh look and carry on with the job.",
     "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "pause", "description": "Stop after the current move and wait.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "resume", "description": "Carry on with the current job after a pause or a stop, keeping what you've learned.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_status", "description": "Rewrite the banner pinned above the chat: what you're doing now, and the plan in one line. It stays until you change it.",
     "parameters": {"type": "object", "properties": {
         "phase": {"type": "string", "enum": ["looking", "planning", "working", "rethinking", "needs you", "done"]},
         "headline": {"type": "string", "description": "one short line, e.g. 'Found 9 things. Working out a plan.'"},
         "plan": {"type": "string", "description": "the plan in one line (e.g. 'screws -> black, Lego -> white, cubes -> gray, dowel stays: no box left'); omit to keep the current one"}},
         "required": ["phase", "headline"]}}},
]


class Chat:
    def __init__(self, robot, say):
        """say(who, text): put a message on the page."""
        self.robot, self.say = robot, say
        self.history: list[dict] = []
        self.kind = "person"  # what started the current turn: the person, or a robot event's kind
        self.banner = {"phase": "waiting", "headline": "Waiting for a job.", "plan": ""}
        self.lock = threading.Lock()
        script = (HERE / "script.md").read_text() if (HERE / "script.md").exists() else ""
        self.system = SYSTEM.format(how=(HERE / "how_i_work.md").read_text(),
                                    script=("A script the person may follow for a video:\n" + script) if script else "")
        self.client, self.model = vlm._client(vlm.model_id(MODEL))

    def user(self, text: str) -> None:
        self._turn(text)

    def event(self, kind: str, text: str) -> None:
        self._turn(f"[robot event: {kind}] {text}", quiet=True, kind=kind)

    def _turn(self, text: str, quiet: bool = False, kind: str = "person") -> None:
        with self.lock:  # one turn at a time: answers come back in order
            self.kind = kind
            self.history.append({"role": "user", "content": text})
            self._trim()
            for _ in range(4):  # model -> tools -> model, until it answers in words
                msgs = [{"role": "system", "content": self.system},
                        {"role": "system", "content": "Your status right now:\n" + self.robot.status()
                         + "\nLately:\n" + "\n".join(self.robot.recent())}] + self.history
                t0 = time.time()
                try:
                    r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=TOOLS, timeout=40,
                                                            extra_body={"reasoning": {"effort": "low"}})
                except Exception as e:
                    self.say("system", f"(the chat model didn't answer: {str(e)[:120]})")
                    self.history = [m for m in self.history if m["role"] == "user" or
                                    (m["role"] == "assistant" and not m.get("tool_calls"))][-10:]  # a clean history
                    return
                m = r.choices[0].message
                calls = m.tool_calls or []
                self.history.append({"role": "assistant", "content": m.content or "",
                                     **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
                if not calls:
                    words = (m.content or "").strip()
                    if words and not (quiet and words in ("-", "—")):
                        self.say("robot", words)
                    return
                for c in calls:
                    result = self._do(c.function.name, json.loads(c.function.arguments or "{}"))
                    self.say("action", f"{c.function.name}: {result}")
                    self.history.append({"role": "tool", "tool_call_id": c.id, "content": result})

    def _trim(self, keep: int = 30) -> None:
        """The last messages, starting at a person's (or event's) message: cutting in the middle of a tool call leaves
        a tool result without its call, and the model refuses the whole conversation (error 400)."""
        h = self.history[-keep:]
        while h and h[0]["role"] != "user":
            h = h[1:]
        self.history = h

    def _do(self, name: str, args: dict) -> str:
        if name == "start_mapping":
            return self.robot.start(args.get("instruction", "") or "map the table", mind="map")
        if name == "start_sorting":
            return self.robot.start(args.get("instruction") or "sort these", mind="focus")
        if name == "add_rule":
            return self.robot.add_rule(args.get("rule", ""))
        if name == "table_changed":
            if self.kind not in ("person", "seen"):  # its own plan or a move is not news about the table
                return "not done: only for changes the person made, or ones you saw while idle"
            return self.robot.start() if self.robot.instruction or self.robot.state == "working" else "noted; nothing to redo yet"
        if name == "pause":
            return self.robot.pause()
        if name == "resume":
            return self.robot.resume()
        if name == "set_status":
            self.banner = {"phase": args.get("phase", "working"), "headline": args.get("headline", ""),
                           "plan": args.get("plan") or self.banner.get("plan", "")}
            self.say("status", json.dumps(self.banner))
            return "banner updated"
        return f"no such tool {name}"
