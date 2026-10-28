"""The board between the two streams: what each has worked out, what one asks of the other, and the news.

    dorsal (lab.minds.twostream)  writes: the things (3D segments), a clear table picture, events ("map_done", "done: ...")
                                  reads:  requests (front of its queue: they skip its own work), the sort rule, labels
    ventral (lab.ventral)         writes: SAM masks + the agreement, labels, the sort rule, requests
                                  reads:  everything, and waits for events

Thread-safe: the dorsal stream runs in the robot's thread, the ventral one in its own.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

ACTIONS = ("break_up", "look_closer", "put_away", "look_for")


@dataclass
class Request:
    id: int
    action: str                   # break_up | look_closer | put_away | look_for (a SAM word on the table picture, in note)
    target: int                   # a thing id
    note: str = ""                # "keep the pieces close", "4 bolts in there"
    expect: int | None = None     # how many things the ventral stream thinks are in it (break_up)
    to: str | None = None         # put_away: "box #2" or "table"
    xy: tuple | None = None       # look_closer at a place instead of a thing (m, base frame)
    status: str = "queued"        # queued | doing | done | refused
    result: str = ""
    who: str = "ventral"          # "ventral", or "person" (from the chat: the only ones done while a job holds)
    on_done: object = None        # called with the request when it's done or refused


@dataclass
class Rule:
    text: str                     # in words, as the ventral stream (or the person) said it
    groups: list = field(default_factory=list)  # [{"name": "red cubes", "ids": [3, 7], "to": "box #2" | "table"}]
    source: str = "ventral"
    boxes: frozenset = frozenset()  # the box ids on the board when it was set (a cup found later makes it stale)


class Board:
    def __init__(self):
        self.cond = threading.Condition()
        self.things: dict = {}        # id -> lab.things.Thing (known and maybe), as the dorsal stream last read them
        self.known_ids: set = set()
        self.image = None             # the table camera's latest picture, and the arm's place in it
        self.arm = None
        self.image_t = 0.0
        self.masks: list = []
        self.agreement = None
        self.labels: dict = {}        # thing id -> what the ventral stream calls it
        self.vouched: set = set()
        self.contents: dict = {}      # box id -> what went in it (names), as the dorsal stream put them     # ids the ventral stream calls one thing by eye (SAM wouldn't outline them)
        self.rule: Rule | None = None
        self.user_rules: list[str] = []
        self.facts: list[dict] = []   # what the person says is on the table: {"text", "items": [{"word", "count", "size_mm"}]}
        self.requests: list[Request] = []
        self.events: list[tuple] = []  # (seq, t, kind, text)
        self._seq = itertools.count(1)
        self._rid = itertools.count(1)
        self.stopped = False
        self.mode = "segments"        # "focus": things are measured objects; segment() finds regions to look at
        self.regions = None           # focus: [{"xy", "prompt", "area_px"}] from the table picture
        self.prompts = None           # focus: the SAM prompts the ventral stream chose
        self.unknown: list[str] = []  # focus: things only depth sees, found again from a second look (no outline yet)
        self.real = False             # the real cell: SAM stays off unless SAM_REAL=1 (lab images to Meta: the team's call)

    # ------------------------------------------------------------ news

    def post(self, kind: str, text: str = "") -> int:
        with self.cond:
            seq = next(self._seq)
            self.events.append((seq, time.time(), kind, text))
            self.cond.notify_all()
            return seq

    def last_seq(self) -> int:
        with self.cond:
            return self.events[-1][0] if self.events else 0

    def wait(self, after: int, kinds=None, timeout: float = 60.0) -> list[tuple]:
        """Events after seq `after` (of these kinds, if given); blocks until there is one, the timeout, or a stop."""
        end = time.time() + timeout
        with self.cond:
            while True:
                new = [e for e in self.events if e[0] > after and (kinds is None or e[2] in kinds)]
                if new or self.stopped or time.time() >= end:
                    return new
                self.cond.wait(timeout=max(0.05, end - time.time()))

    def stop(self) -> None:
        with self.cond:
            self.stopped = True
            self.cond.notify_all()

    # ------------------------------------------------------------ the dorsal stream's side

    def set_things(self, known, maybe, image=None, arm=None) -> None:
        with self.cond:
            self.things = {t.id: t for t in list(known) + list(maybe)}
            self.known_ids = {t.id for t in known}
            for t in self.things.values():
                t.label = self.labels.get(t.id)
            if image is not None:
                self.image, self.arm, self.image_t = image, arm, time.time()

    def forget_near(self, xy, radius: float) -> list[int]:
        """Things near where the arm just worked may be different things now (ids are kept by position): their names
        and their places in the rule are dropped until the ventral stream looks again. Returns the ids dropped."""
        import numpy as np
        with self.cond:
            ids = [i for i, t in self.things.items() if np.linalg.norm(np.asarray(t.xy) - np.asarray(xy)) < radius]
            for i in ids:
                self.labels.pop(i, None)
                self.vouched.discard(i)
                if i in self.things:
                    self.things[i].label = None
            if self.rule is not None:
                for g in self.rule.groups:
                    g["ids"] = [i for i in g["ids"] if i not in ids]
            if self.agreement is not None:
                self.agreement.rows = [r for r in self.agreement.rows if r.seg not in ids]
        return ids

    def take_request(self, who: str | None = None) -> Request | None:
        with self.cond:
            r = next((r for r in self.requests if r.status == "queued" and who in (None, r.who)), None)
            if r is not None:
                r.status = "doing"
            return r

    def finish(self, r: Request, status: str, result: str) -> None:
        r.status, r.result = status, result
        self.post(f"request_{status}", f"request {r.id} ({r.action} #{r.target}): {result}")
        if r.on_done is not None:
            r.on_done(r)

    # ------------------------------------------------------------ the ventral stream's side

    def request(self, action: str, target: int | None, note: str = "", expect: int | None = None, to: str | None = None,
                xy=None, who: str = "ventral", on_done=None) -> Request:
        if action not in ACTIONS:
            raise ValueError(f"no such action {action}: one of {', '.join(ACTIONS)}")
        with self.cond:
            if xy is None and action != "look_for" and target not in self.things:
                raise ValueError(f"there is no thing #{target} on the board")
            r = Request(next(self._rid), action, -1 if target is None else int(target), note, expect, to,
                        None if xy is None else (float(xy[0]), float(xy[1])), who=who, on_done=on_done)
            self.requests.append(r)
        what = f"#{target}" if xy is None else f"({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm"
        what = f"'{note}' " + (what if xy is not None else "on the table") if action == "look_for" else what
        self.post("request", f"request {r.id}: {action} {what}" + (f" ({note})" if note and action != "look_for" else ""))
        return r

    def name(self, tid: int, name: str) -> None:
        with self.cond:
            self.labels[int(tid)] = name
            if int(tid) in self.things:
                self.things[int(tid)].label = name

    def set_rule(self, rule: Rule) -> None:
        with self.cond:
            rule.boxes = frozenset(t.id for t in self.things.values() if t.kind == "box")
            self.rule = rule
        self.post("rule", rule.text)

    def add_fact(self, text: str, items: list[dict]) -> None:
        with self.cond:
            self.facts.append({"text": text, "items": list(items)})
        self.post("user_fact", text)

    def told(self) -> dict:
        """The person's words for what's on the table -> {"count": n or None, "size_mm": s or None}; later facts win."""
        out: dict = {}
        with self.cond:
            for f in self.facts:
                for it in f["items"]:
                    w = (it.get("word") or "").strip().lower()
                    if w:
                        d = out.setdefault(w, {"count": None, "size_mm": None})
                        d.update({k: it[k] for k in ("count", "size_mm") if it.get(k) not in (None, "", 0)})
        return out

    def add_user_rule(self, text: str) -> None:
        with self.cond:
            self.user_rules.append(text)
        self.post("user_rule", text)

    def queue_text(self) -> str:
        with self.cond:
            live = [r for r in self.requests if r.status in ("queued", "doing")]
            return "; ".join(f"{r.id}: {r.action} " + (f"#{r.target}" if r.xy is None else "a place") + f" [{r.status}]"
                             for r in live) or "empty"
