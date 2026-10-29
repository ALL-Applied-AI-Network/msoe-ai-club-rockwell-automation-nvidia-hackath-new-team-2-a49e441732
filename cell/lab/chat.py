"""The robot's voice: a chat agent (GPT-6 Luna) that talks for the robot (lab/robot.py) while it works.

Every message (from the person, or an event from the robot) goes to the model with the robot's live status. The model
decides what kind of message it is and what to do about it, through tools; nothing here sorts messages into kinds:
    start_sorting(instruction)   a job: sort the table (the focus mind: SAM picks where to look, depth and SAM must agree)
    look_at(x_mm, y_mm | thing | word)  "look over there": one close look, ahead of everything else; it reports back
    start_mapping(instruction)   the old whole-table depth map (only when asked for a map by name)
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
table of small parts and cups. How you work now: your ventral stream (an LLM with SAM 3.1) looks at the table camera's
picture and outlines what's worth a look. You then look closely from straight above each place with the wrist camera:
SAM outlines things in that picture, and you keep only what SAM's outline AND the wrist depth agree on (depth must
stand up inside the outline). The outline gives the shape, depth gives the height. Where they disagree you look again
from another spot: an outline with nothing standing in it is flat (a sticker, a shadow, the paper); raised depth with
no outline is usually sensor noise, and only counts (as an "unknown", steered around, never picked) if a second look
finds it again. Grips are planned on the measured outline: the jaws (20 mm apart when open) must fit across the part
with both fingers clear of everything around it. Then the sort rule (the ventral stream's; the person's rules win)
sends each part to a cup: a fresh close look above it, grip, drop in the cup, look to check. You can't stack, pour, or
take things back out of a cup yet; say so plainly if asked.

Your status below lists every part and cup you've measured (number, colour, size, height, where, how much SAM and depth
agree), what only depth sees, the sort rule, the queue, and what you gave up on and why. Explain from it when asked
("why didn't you pick the blue one?", "what do you see?"): give the real reason, in plain words, with the numbers that
matter. Places are in mm in your base frame: x is the distance out from your base, +y is to your left (the person may
stand anywhere: if "left" is unclear, use the things on the table as landmarks instead).

A person is standing next to you and talking to you in this chat while you work. Speak as the robot, in the first
person, briefly (one to three short sentences, like talking out loud). You keep working while you talk.

Decide what each message is and act on it:
- "sort these", "put them away", "clean up the table", "go": call start_sorting.
- "look over there", "what's that by the red cup?", "check the corner", "is there a screw?", or a question you can only
  answer by looking: call look_at. Give a place in mm (work it out from the things in your status: "left of the blue
  cup" is the cup's place plus about 60 mm in the direction meant), a thing's number to look at it again, or a word for
  SAM ("screw", "washer") to find it on the table picture (with a place too, to search only near there). With no job
  running, look_at only looks: it doesn't start a sort. Say you're looking; the findings come back as a "looked" event.
- what's on the table, from the person ("that pile is 4 screws and 2 washers", "there are 6 cubes", "the cubes are
  15 mm"): call tell, with each kind as a short word SAM can find ("screw", "washer", "red cube") and the count and
  size if they gave them. It's not a rule and not a change to the table. You use their words to search, check your
  counts against theirs before you say you're done, and say so when what you measure disagrees with what they said
  (their word helps you look; it never replaces measuring).
- "do a data run", "gather data", "record a sweep": call data_run (no sorting: camera sweeps, everything recorded);
  "how fast can you move?", "speed test": call data_run with kind "speed". Not while a sort is running.
- "map the whole table": call start_mapping (the old depth-only map; it stops a sort in progress).
- a correction or preference ("no, Lego goes in the white box", "leave the ring out"): call add_rule with it, in plain
  words; during a sort your ventral stream takes it and redoes the sort rule. Things already in a box stay there unless they ask; if they ask,
  say honestly whether you can get them back out (see what you can't do).
- news that the person CHANGED the table ("I just dumped more on", "I took the ring"): call table_changed. Never for
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

When a message starts with "[robot event: looked]", it's the result of a look the person asked for: always tell them
what you found there in a sentence or two (what both streams agree on, with sizes; what only depth sees; or that you
couldn't look, and why). Don't list millimetre coordinates unless they ask.
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
    {"type": "function", "function": {"name": "look_at", "description": "Look closely at one place, ahead of everything else, and report what's there. Give x_mm and y_mm (base frame), or thing (a number from your status), or word (what SAM should find on the table picture; with x_mm/y_mm, only near there). With no job running it only looks (no sort).",
     "parameters": {"type": "object", "properties": {
         "x_mm": {"type": "number", "description": "distance out from the base, 150-650"},
         "y_mm": {"type": "number", "description": "+ is the robot's left, -450 to 450"},
         "thing": {"type": "integer", "description": "a measured thing's number, to look at it again"},
         "word": {"type": "string", "description": "a short SAM prompt, e.g. 'screw'"}}}}},
    {"type": "function", "function": {"name": "tell", "description": "What the person says is on the table (kinds, counts, sizes). Used to search with their words, to check counts before finishing, and to explain a measurement that disagrees.",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "description": "their words"},
         "items": {"type": "array", "items": {"type": "object", "properties": {
             "word": {"type": "string", "description": "a short SAM word for one kind, singular: 'screw', 'red cube'"},
             "count": {"type": "integer", "description": "how many, if they said"},
             "size_mm": {"type": "number", "description": "its width in mm, if they said"}}, "required": ["word"]}}},
         "required": ["text", "items"]}}},
    {"type": "function", "function": {"name": "data_run", "description": "No sorting. kind 'sweep': camera sweeps over the parts recording pictures at full rate (about 2-3 min). kind 'speed': a speed test of the arm, out and back at 100-500 mm/s, 150 mm above the table.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["sweep", "speed"]},
         "x_mm": {"type": "number", "description": "sweep: centre of the sweeps (default: where the parts are)"},
         "y_mm": {"type": "number"}}, "required": ["kind"]}}},
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
        if name == "look_at":
            return self.robot.look_at(args.get("x_mm"), args.get("y_mm"), args.get("thing"), (args.get("word") or "").strip())
        if name == "data_run":
            return self.robot.data_job(args.get("kind") or "sweep", args.get("x_mm"), args.get("y_mm"))
        if name == "tell":
            return self.robot.tell(args.get("text", ""), args.get("items") or [])
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
