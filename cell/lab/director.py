"""The director: the expert on top of the physical intelligence. A VLM that looks at the whole table and decides.

The physical side knows where things are, how big, whether each has moved as one piece. The per-thing close-ups
(lab/labels.py) say what each looks like. The director puts it together, the way an expert glances at a tray of parts:
    - what is this ("a pile of mixed hardware", "six cubes, already apart")
    - the sorting rule (the user's, if they gave one; else its own) and which box each group goes into
    - which things it is sure about: one familiar object, alone. The robot carries those straight to their box, and
      arriving there in one piece is the physical check (nothing left behind = it was one rigid object)
    - which things to check first: might be several, attached, or unfamiliar. Those get the slow physical tests.

It sees the table camera's picture with every thing outlined and numbered ("set of marks"), plus a text list of what
was measured and what each close-up showed. It never says where to move: only what things are and where they belong.
It runs in the background; the arm keeps working on the last plan while a new one is thought out.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np

from lab import vlm
from lab.labels import measure
from lab.viz import id_color, project_to_image

PLAN = """You are the planner for a robot arm that sorts objects on a table. You are the expert: you recognise things
at a glance, so the robot can skip slow physical tests for things you're sure about. Be honest about what you're not
sure of: those get tested, and a wrong "sure" puts something in the wrong box.

The user's instruction: "{instruction}"

What the robot knows (sizes from a depth camera; "moved as one piece" means it was carried and nothing was left behind,
so it is certainly one object; "untouched" might still be several things touching; names from close-up photos):
{things}
Boxes to sort into:
{boxes}
Already sorted: {sorted}
{previous}
The image is the table from above; every thing is outlined and numbered with its id.

Answer in JSON only:
{{
  "scene": "one sentence: what is on the table",
  "state": "pile | partly spread | spread out",
  "rule": "the sorting rule, one sentence (the user's if they gave one)",
  "groups": [{{"name": "group name", "box": 0, "ids": [1, 2]}}],
  "sure": [1, 2],
  "check": [{{"id": 3, "why": "<your reason>"}}],
  "instruction_check": "<your answer, or null>"
}}
"check": things (not boxes) to test before sorting, each with your own reason: they might be several objects touching,
look attached, or you don't know what they are.
"instruction_check": if the user said what's on the table, does that match what you see? Otherwise null.
If the user gave no rule, choose the one a person tidying this table would: separate the main kinds of object
(by what they are first; by colour only if they're all the same kind), using every box that helps, and leave on the
table only what doesn't fit. A box's colour means nothing unless the user's instruction mentions it.
Rules: put every thing id (not the boxes) in exactly one group, or in "check" if you can't tell what it is yet.
"box" is a box id from the list, or null to leave that group on the table; use each box for one group only.
"sure" lists the ids you'd bet are ONE familiar object each, so the robot sorts them without testing.
Keep the same rule and groups as your previous plan unless new information changes them."""


class Director:
    def __init__(self, labeler, cams: dict, instruction: str = ""):
        self.labeler, self.cams, self.instruction = labeler, cams, instruction
        self.think = True  # slower (~1 min), but without it plans flip between calls and put a thing in two groups
        self.plan: dict | None = None
        self.pending: Future | None = None
        self.n = 0
        self.known: set[int] = set()       # thing ids the current plan was made with
        self.named_then: dict[int, str] = {}  # ... and what they were called then

    def busy(self) -> bool:
        return self.pending is not None

    def again(self) -> None:
        """Make the next stale() true: the robot is stuck waiting on this plan."""
        self.known = set()

    def stale(self, world) -> bool:
        """A thing exists that the plan doesn't mention (new pieces after a split), or new names came in."""
        ids = {t.id for t in world.things.values() if t.status in ("blob", "one")}
        named = {t.id: _name(t) for t in world.things.values() if t.id in ids and getattr(t, "label", None)}
        return self.plan is None or bool(ids - self.known) or any(self.named_then.get(i) != n for i, n in named.items())

    def request(self, world, obs: dict) -> str | None:
        if self.pending is not None or not self.labeler.on:
            return None
        things = [t for t in world.things.values() if t.status in ("blob", "one")]
        bins = world.bins()
        if not things:
            return None
        lines = []
        for t in sorted(things, key=lambda t: t.id):
            m = measure(t.points)
            lab = getattr(t, "label", None)
            seen = (f'close-up: "{lab.get("color", "")} {lab.get("name")}", {lab.get("category")}, count {lab.get("count", 1)}, '
                    f'confidence {float(lab.get("confidence") or 0):.2f}' + (", looks attached" if lab.get("attached") else "")
                    if lab else "close-up: not back yet")
            lines.append(f"  #{t.id}: {'moved as one piece' if t.status == 'one' else 'untouched'}, "
                         f"{m['L']:.0f} x {m['W']:.0f} x {m['H']:.0f} mm, {'alone' if world.isolated(t) else 'touching others'}; {seen}")
        blines = []
        for b in bins:
            lab = getattr(b, "label", None) or {}
            m = measure(b.points)
            blines.append(f"  box #{b.id}: {lab.get('color') or 'colour unknown'}, {m['L']:.0f} x {m['W']:.0f} mm, {m['H']:.0f} mm tall")
        done = ", ".join(f"#{s['id']} ({s.get('color') or ''} {s.get('name') or '?'}) -> box #{s['bin']}" for s in world.sorted) or "nothing yet"
        prev = ""
        if self.plan:
            prev = ("Your previous plan: rule \"" + str(self.plan.get("rule")) + "\"; groups: "
                    + "; ".join(f"{g.get('name')} -> box {g.get('box')}" for g in self.plan.get("groups", [])))
        prompt = PLAN.format(instruction=self.instruction or "Sort these.", things="\n".join(lines),
                             boxes="\n".join(blines) or "  none: groups stay on the table", sorted=done, previous=prev)
        image = self.marked(world, obs)
        self.known = {t.id for t in things}
        self.named_then = {t.id: _name(t) for t in things if getattr(t, "label", None)}
        self.n += 1
        self.pending = self.labeler.pool.submit(self._call, self.n, image, prompt)
        return f"Asking the director for a plan ({len(things)} things, {len(bins)} boxes)."

    def marked(self, world, obs: dict) -> np.ndarray:
        """The table camera's picture, cropped to the things and boxes, each outlined and numbered."""
        img = obs["images"]["table"].copy()
        h, w = img.shape[:2]
        boxes = []
        for t in world.things.values():
            uv = project_to_image("table", self.cams, t.points[::3, :3], (w, h))
            uv = uv[np.all(np.isfinite(uv), 1)]
            if len(uv) < 3:
                continue
            hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
            col = tuple(int(c) for c in id_color(t.id))
            cv2.polylines(img, [hull], True, col, 1)
            boxes.append(np.r_[uv.min(0), uv.max(0)])
        if not boxes:
            return img
        b = np.array(boxes)
        x0, y0 = np.maximum(b[:, :2].min(0) - 40, 0).astype(int)
        x1, y1 = np.minimum(b[:, 2:].max(0) + 40, [w, h]).astype(int)
        crop = img[y0:y1, x0:x1]
        s = 1400 / max(crop.shape[1], 1)
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        for t in world.things.values():  # numbers drawn after scaling, so they stay readable
            uv = project_to_image("table", self.cams, t.points[::3, :3], (w, h))
            uv = uv[np.all(np.isfinite(uv), 1)]
            if len(uv) < 3:
                continue
            p = ((uv.mean(0) - [x0, y0]) * s).astype(int)
            txt = f"#{t.id}" + (" box" if t.status == "bin" else "")
            cv2.putText(crop, txt, (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(crop, txt, (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return crop

    def _call(self, n: int, image: np.ndarray, prompt: str) -> dict:
        t0, r, model, tries = time.time(), None, None, []
        for model in self.labeler.models:
            try:
                r = vlm.ask_image(image, prompt, model=model, max_tokens=6000, think=self.think)
                if r and isinstance(r["json"], dict) and r["json"].get("groups") is not None:
                    tries.append(f"{model.split('/')[-1]}: ok")
                    break
                tries.append(f"{model.split('/')[-1]}: no usable plan")
            except Exception as e:
                tries.append(f"{model.split('/')[-1]}: {str(e)[:150]}")
            r = None
        out = {"n": n, "plan": r["json"] if r else None, "seconds": time.time() - t0, "model": model, "tries": tries,
               "image": image, "text": (r or {}).get("text", "")}
        d = self.labeler.save_dir
        if d is not None:
            d = Path(d)
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"director_{n:02d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            (d / f"director_{n:02d}.json").write_text(json.dumps({k: v for k, v in out.items() if k != "image"}
                                                                 | {"prompt": prompt}, indent=1, default=float))
        return out

    def poll(self, wait: float = 0.0) -> dict | None:
        """The new plan's answer if it has come back (waiting up to `wait` s), else None."""
        if self.pending is None or (not wait and not self.pending.done()):
            return None
        try:
            r = self.pending.result(timeout=wait or None)
        except Exception:
            return None
        self.pending = None
        if r["plan"]:
            self.plan = r["plan"]
        return r

    # ------------------------------------------------------------ reading the plan

    def group_of(self, tid: int) -> dict | None:
        gs = [g for g in (self.plan or {}).get("groups", []) if tid in (g.get("ids") or [])]
        return gs[0] if len(gs) == 1 else None

    def conflict(self, tid: int) -> str | None:
        """The plan contradicts itself about this thing (two groups, or a group and the check list)."""
        gs = [g.get("name") for g in (self.plan or {}).get("groups", []) if tid in (g.get("ids") or [])]
        if len(gs) > 1:
            return f"the plan puts it in {' and '.join(map(str, gs))}"
        return None  # in a group and on the check list: checking wins (the robot tests it, then sorts it)

    def sure(self, tid: int) -> bool:
        return tid in ((self.plan or {}).get("sure") or [])

    def check(self, tid: int) -> str | None:
        for c in (self.plan or {}).get("check") or []:
            if isinstance(c, dict) and c.get("id") == tid:
                return c.get("why") or "the director wants a closer look"
        return None


def _name(t) -> str:
    lab = getattr(t, "label", None) or {}
    return f"{lab.get('color', '')} {lab.get('name', '')} {lab.get('count', 1)}"
