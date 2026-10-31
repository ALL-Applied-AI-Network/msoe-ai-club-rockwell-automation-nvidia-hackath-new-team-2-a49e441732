"""The robot's one idea of the world: solid things and ghosts.

A solid is evidence: a carved object a camera actually confirmed at that spot. A ghost is a prediction:
where a move should have left something, believed but not yet checked. Picking a thing up doesn't move
it here - the solid stays where it was last seen and a ghost appears at the destination; the confirming
look afterwards is what resolves them (predict, then update). Until it does, BOTH occupy space, so the
arm plans around the prediction and the evidence alike and never sets one thing down on another.

    w = World()
    w.look(carve.objects(...))                # merge a fresh carve into the things (ids stay put)
    g = w.plan_move(3, 520.0, -180.0)         # thing 3 is being carried: ghost at the destination
    armcheck.check(rows, q0, w.boxes(ignore=(3,)))
    w.confirm(3, found=(521.0, -178.5))       # the look where it should be: ghost hardens, source frees
    print(w.mirror())                         # the plain-text state the LLM reasons over
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from calibration.geometry import SURFACE_Z
from lab.armcheck import Box

MATCH_MM = 25.0   # a carved object this close to a known thing is that thing
PAD_MM = 4.0      # grown around every thing when it becomes an obstacle box


@dataclass
class Thing:
    id: int
    name: str                      # the SAM word that found it
    x_mm: float
    y_mm: float
    size_mm: tuple[float, float]   # footprint along its own axes
    top_mm: float                  # highest solid voxel above the table
    yaw: float = 0.0               # footprint direction (rad, base frame)
    state: str = "solid"           # "solid" (seen there) | "ghost" (predicted there)
    seen: float = 0.0              # when a camera last confirmed it
    of: int | None = None          # a ghost remembers which solid it is the prediction of
    views: int = 1                 # how many outlines have constrained it: 1 = a one-glance prism,
                                   # its height is slack for the agent to tighten when an action needs it
    note: str = ""

    def dist_mm(self, x: float, y: float) -> float:
        return float(np.hypot(self.x_mm - x, self.y_mm - y))


class World:
    def __init__(self) -> None:
        self.things: list[Thing] = []
        self.events: list[str] = []
        self._ids = itertools.count(1)
        self._lock = threading.RLock()

    # ---- update (evidence in) ----

    def look(self, carved: list[dict], names_too: bool = True) -> list[Thing]:
        """Merge one carve into the world: a carved object near a known solid updates it (the id survives,
        the V4 lesson); anything new becomes a new solid. Ghosts are never touched here - only their
        confirming look resolves them."""
        now = time.time()
        out = []
        claimed: set[int] = set()  # each thing matches at most one blob per look (a hull spike next to a
        with self._lock:           # box must not overwrite the box's measured size)
            for ob in sorted(carved, key=lambda o: -o.get("voxels", 0)):
                near = [t for t in self.things if t.state == "solid" and t.id not in claimed
                        and t.dist_mm(ob["x_mm"], ob["y_mm"]) < MATCH_MM]
                if near:
                    t = min(near, key=lambda t: t.dist_mm(ob["x_mm"], ob["y_mm"]))
                    claimed.add(t.id)
                    t.x_mm, t.y_mm, t.yaw = ob["x_mm"], ob["y_mm"], ob.get("yaw", t.yaw)
                    t.size_mm, t.top_mm, t.seen = tuple(ob["size_mm"]), ob["top_mm"], now
                    t.views = max(t.views, int(ob.get("views", 1)))
                    if names_too and ob.get("name") and ob["name"] not in ("?", "blob"):
                        t.name = ob["name"]
                else:
                    t = Thing(next(self._ids), ob.get("name") or "blob", ob["x_mm"], ob["y_mm"],
                              tuple(ob["size_mm"]), ob["top_mm"], ob.get("yaw", 0.0), seen=now,
                              views=int(ob.get("views", 1)))
                    claimed.add(t.id)
                    self.things.append(t)
                    self._say(f"new: {t.name} #{t.id} at ({t.x_mm:.0f}, {t.y_mm:.0f})")
                out.append(t)
        return out

    # ---- predict (a move is about to happen) ----

    def plan_move(self, tid: int, x_mm: float, y_mm: float, yaw: float | None = None) -> Thing:
        """The arm is about to carry thing tid there. The solid STAYS (the evidence still says so);
        a ghost appears at the destination. Both take up space until the confirming look."""
        with self._lock:
            t = self[tid]
            g = replace(t, id=next(self._ids), x_mm=x_mm, y_mm=y_mm,
                        yaw=t.yaw if yaw is None else yaw, state="ghost", of=t.id, seen=0.0)
            self.things.append(g)
            self._say(f"predict: {t.name} #{t.id} -> ({x_mm:.0f}, {y_mm:.0f}) as ghost #{g.id}")
            return g

    def confirm(self, tid: int, found: tuple[float, float] | None) -> bool:
        """The look where the ghost of thing tid should be. Found there: the ghost hardens into the thing
        (the evidence moves at last) and the old spot frees. Not found: the prediction was wrong - the
        ghost dies, the solid stands, and someone should go look."""
        with self._lock:
            g = next((t for t in self.things if t.state == "ghost" and t.of == tid), None)
            t = self[tid]
            if g is None:
                return False
            self.things.remove(g)
            if found is not None:
                t.x_mm, t.y_mm, t.seen = float(found[0]), float(found[1]), time.time()
                t.yaw = g.yaw
                self._say(f"confirm: {t.name} #{t.id} is at ({t.x_mm:.0f}, {t.y_mm:.0f})")
                return True
            self._say(f"miss: {t.name} #{t.id} is NOT at ({g.x_mm:.0f}, {g.y_mm:.0f}); last seen ({t.x_mm:.0f}, {t.y_mm:.0f})")
            return False

    def forget(self, tid: int) -> None:
        with self._lock:
            self.things = [t for t in self.things if t.id != tid and t.of != tid]

    def rename(self, tid: int, name: str) -> None:
        """The LLM recognizes a blob and names it; from then on everyone speaks that name."""
        with self._lock:
            t = self[tid]
            self._say(f"named: {t.name} #{t.id} is now {name!r}")
            t.name = name

    # ---- what everyone else reads ----

    def boxes(self, ignore: tuple[int, ...] = ()) -> list[Box]:
        """Every thing - solid AND ghost - as an armcheck obstacle, padded. The thing being carried
        goes in ignore (armcheck carries it as held instead)."""
        with self._lock:
            out = []
            for t in self.things:
                if t.id in ignore or t.of in ignore:
                    continue
                h = max(t.top_mm, 4.0) / 1000.0
                half = np.array([(t.size_mm[0] / 2 + PAD_MM) / 1000.0,
                                 (t.size_mm[1] / 2 + PAD_MM) / 1000.0, h / 2])
                c = np.array([t.x_mm / 1000.0, t.y_mm / 1000.0, SURFACE_Z + h / 2])
                out.append(Box(c, t.yaw, half, f"{t.name} #{t.id} ({t.state})"))
            return out

    def mirror(self) -> str:
        """The world as the LLM reads it: one line per thing, plain text, mm."""
        with self._lock:
            if not self.things:
                return "nothing known yet"
            lines = []
            for t in sorted(self.things, key=lambda t: t.id):
                age = f"{time.time() - t.seen:.0f}s ago" if t.seen else "never"
                tag = "" if t.state == "solid" else f"  GHOST of #{t.of}, not yet checked"
                sure = "" if t.views >= 2 else "  (one glance: height is a guess, look closer to be sure)"
                lines.append(f"{t.name} #{t.id}: ({t.x_mm:.0f}, {t.y_mm:.0f}) mm, "
                             f"{t.size_mm[0]:.0f} x {t.size_mm[1]:.0f} mm, {t.top_mm:.0f} mm tall, "
                             f"seen {age}{tag}{sure}{'  - ' + t.note if t.note else ''}")
            return "\n".join(lines)

    def __getitem__(self, tid: int) -> Thing:
        t = next((t for t in self.things if t.id == tid), None)
        if t is None:
            raise KeyError(f"no thing #{tid}")
        return t

    def _say(self, s: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')} {s}")
