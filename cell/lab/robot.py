"""The robot as a long-running session: a body (the Isaac cell, standing in for the real one), a mind (the sort policy:
world model, VLM close-ups, director), and a voice (lab/chat.py talks for it). lab/app.py puts it on a page.

It runs in its own thread and never waits for the chat: a job (an instruction) starts the sort policy; rules from the
person change its plan between moves; when there's nothing to do it watches the table, and a change there (someone
pouring more parts on, taking something away) is reported, not acted on: the chat decides what to do about it.

Everything it narrates goes to Rerun as before (served live: the page embeds the web viewer, and native viewers can
connect too), and the lines that matter to a person standing next to it become events for the chat.
"""
from __future__ import annotations

import collections
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lab import primitives as P, viz
from lab.isaaccell import IsaacCell
from lab.policies.sort import SortPolicy
from lab.trial import build, load_trial

RUNS = Path(__file__).resolve().parents[1] / "runs"
WATCH_S = 8.0          # idle: look at the table this often (wall clock)
CHANGE_PX = 120        # this many table-depth pixels changing by > 5 mm: something on the table changed


class Robot:
    def __init__(self, trial: str = "chat_table", seed: int = 0):
        self.cell = IsaacCell(build(load_trial(trial), seed))
        self.dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_chat"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.viewer_uri = viz.start_served("robot", save_to=self.dir / "replay.rrd")
        viz.truth_setup(self.cell)
        self.anim = viz.Animator(self.cell)
        self.lock = threading.Lock()
        self.events: collections.deque = collections.deque(maxlen=500)  # (time, kind, text): for the chat and the page
        self.listeners: list = []
        viz.LISTENERS.append(self._heard)
        self.instruction, self.rules = "", []
        self.policy: SortPolicy | None = None
        self.pending: str | None = None   # "start" | "pause": picked up by the robot thread between moves
        self.state = "idle"
        self.distract_s = 0.0             # hold still this long before the next move (talking to it slows it down)
        self.baseline = None              # the table depth when it last finished: changes are measured from it
        self.step = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ commands (from the chat or the page; any thread)

    def start(self, instruction: str | None = None) -> str:
        with self.lock:
            if instruction is not None:
                self.instruction = instruction
            self.pending = "start"
        return f"starting: \"{self.full_instruction() or 'sort these'}\""

    def add_rule(self, rule: str) -> str:
        with self.lock:
            self.rules.append(rule)
            pol = self.policy
        if pol is not None:  # the director replans with it before the next move
            pol.instruction = self.full_instruction()
            pol.director.instruction = pol.instruction
            pol.director.again()
            return "rule added; replanning"
        return "rule added; it applies to the next sort"

    def pause(self) -> str:
        with self.lock:
            self.pending = "pause"
        return "pausing after this move"

    def full_instruction(self) -> str:
        rules = "".join(f"\nRule from the person, overriding anything above: {r}" for r in self.rules)
        return (self.instruction + rules).strip()

    def distract(self, seconds: float) -> None:
        self.distract_s += seconds

    # ------------------------------------------------------------ what it knows (for the chat)

    def status(self) -> str:
        """A plain account of the robot's state and its world model, for the chat agent."""
        lines = [f"State: {self.state}. Instruction: \"{self.instruction or '(none yet)'}\"."]
        if self.rules:
            lines.append("Standing rules from the person: " + "; ".join(self.rules))
        pol = self.policy
        if pol is not None:
            plan = getattr(pol.director, "plan", None) or {}
            if plan:
                lines.append(f"Current plan: rule \"{plan.get('rule')}\"; groups: "
                             + "; ".join(f"{g.get('name')} -> box {g.get('box')}" for g in plan.get("groups", [])))
            for t in sorted(pol.world.things.values(), key=lambda t: t.id):
                lab = getattr(t, "label", None) or {}
                name = f"{lab.get('color', '')} {lab.get('name', '')}".strip() or "not named yet"
                where = f"({t.centroid[0] * 1000:.0f}, {t.centroid[1] * 1000:.0f}) mm"
                if t.status == "bin":
                    lines.append(f"  box #{t.id} at {where}: {getattr(t, 'colour', None) or name}")
                else:
                    lines.append(f"  #{t.id} at {where}: {name}; {t.status}; plan: {getattr(t, 'plan', '') or '-'}")
            for s in pol.world.sorted:
                lines.append(f"  sorted: #{s['id']} {s.get('color') or ''} {s.get('name') or '?'} -> box #{s['bin']}")
        lines.append(f"Robot time so far: {self.cell.t:.0f} s.")
        return "\n".join(lines)

    def recent(self, n: int = 12) -> list[str]:
        return [f"{kind}: {text}" for _, kind, text in list(self.events)[-n:]]

    # ------------------------------------------------------------ the body's loop

    def _emit(self, kind: str, text: str) -> None:
        self.events.append((time.time(), kind, text))
        for f in list(self.listeners):
            try:
                f(kind, text)
            except Exception:
                pass

    def _heard(self, panel: str, text: str, level: str) -> None:
        """Narration / VLM lines that a person standing there would care about become events."""
        if level == "DEBUG" or text.startswith("("):
            return
        if panel == "vlm" and text.startswith("Director ("):
            self._emit("plan", re.sub(r"^Director \([^)]*\): ", "", text))
        elif panel == "vlm" and text.startswith("Instruction vs what I see"):
            self._emit("check", text)
        elif panel == "narration" and re.match(r"Move \d+\.", text):
            self._emit("move", re.sub(r"^Move \d+\. ", "", text).split(": ")[0])
        elif panel == "narration" and (text.startswith(("Done", "Nothing more", "Sorted:", "Lost")) or level == "WARN"):
            self._emit("note", text)

    def _run(self) -> None:
        obs, fresh = None, True
        last_watch = 0.0
        while True:
            with self.lock:
                cmd, self.pending = self.pending, None
            if cmd == "start":
                self._new_policy()
                obs, fresh = self.cell.observe(), True
            elif cmd == "pause" and self.policy is not None:
                self.policy, self.state = None, "paused"
                self._emit("state", "paused")
            if self.policy is None:
                if time.time() - last_watch > WATCH_S:
                    last_watch = time.time()
                    self._watch()
                time.sleep(0.2)
                continue
            try:
                obs, fresh = self._act(obs, fresh)
            except Exception as e:  # a crash in the policy must not kill the body
                self._emit("note", f"I hit an error and stopped: {type(e).__name__}: {e}")
                self.policy, self.state = None, "idle"

    def _new_policy(self) -> None:
        pol = SortPolicy(instruction=self.full_instruction())
        pol.reset()
        if pol.labeler is not None:
            pol.labeler.save_dir = self.dir / "labels"
        self.cell.image_size = tuple(getattr(pol, "IMAGE_SIZE", self.cell.image_size))
        self.cell.instruction = self.full_instruction()
        self.policy, self.state = pol, "working"
        self._emit("state", "working")

    def _act(self, obs, fresh):
        pol = self.policy
        viz.set_step(self.step)
        viz.set_sim_time(self.cell.t)
        self.step += 1
        chunk = pol.act(obs)
        if getattr(pol, "done", False):
            self.policy, self.state = None, "idle"
            self._emit("done", self.status())
            self.baseline = self._table_depth()
            return obs, True
        if self.distract_s > 0:  # someone's talking to it: it slows down
            n = int(min(self.distract_s, 6.0) / P.DT)
            self.distract_s = 0.0
            o = self.cell.observe()
            self.cell.play(P.hold(np.asarray(o["tcp_pose"][:3], float), float(o["tcp_pose"][5]), int(o["joints"][6] >= 0.5), n),
                           on_frame=self.anim)
        self.cell.play(chunk, on_frame=self.anim)
        fresh = getattr(pol, "wants_obs", True)
        return (self.cell.observe() if fresh else obs), fresh

    # ------------------------------------------------------------ watching the table when idle

    def _table_depth(self):
        return self.cell.observe()["depth"]["table"].astype(np.int32)

    def _watch(self) -> None:
        d = self._table_depth()
        if self.baseline is None:
            self.baseline = d
            return
        both = (d > 0) & (self.baseline > 0)
        changed = both & (np.abs(d - self.baseline) > 5)
        if changed.sum() > CHANGE_PX:
            nearer = (changed & (d < self.baseline)).sum() > (changed & (d > self.baseline)).sum()
            self._emit("seen", "Something changed on the table while I was idle: "
                       + ("there's more on it than before." if nearer else "something's gone from it."))
            self.baseline = d

    # ------------------------------------------------------------ Joel's hands (the page's panel)

    def objects(self) -> list[dict]:
        out = []
        for i in self.cell.instances:
            p = self.cell.poses.get(i.id)
            if p is None:
                continue
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color, "box": i.kind.startswith("box"),
                        "x": round(float(p[0]) * 1000), "y": round(float(p[1]) * 1000)})
        return out

    def pour(self, order: list[tuple[str, str]]) -> str:
        new = self.cell.pour(order)
        viz.truth_setup(self.cell)
        return f"poured {len(new)} things onto the table"

    def take(self, ids) -> str:
        self.cell.take(ids)
        return f"took {len(ids)} thing(s) off the table"

    def tip_box(self, box_id: int) -> str:
        out = self.cell.tip_box(box_id)
        return f"tipped box {box_id} out: {len(out)} things back on the table"

    def add_box(self, color: str) -> str:
        b = self.cell.add_box(color, at=(0.30, 0.22))
        viz.truth_setup(self.cell)
        return f"put a {color} box down at (300, 220) mm"
