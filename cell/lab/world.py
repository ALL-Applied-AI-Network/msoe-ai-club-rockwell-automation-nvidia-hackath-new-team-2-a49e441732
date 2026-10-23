"""The robot's working model of the table: the things it has found, what it believes about each, and what changed.

Every thing starts as a hypothesis, "this blob is one rigid object" (status "blob": not yet tested). The test is to
move it: grab part of it and carry it somewhere clear, or rake through it. Then a fresh scan is compared with what was
expected (common fate: what moved together is one thing):

    carried, and nothing was left behind      -> it was one object: same id, status "one"
    carried, but part of it stayed behind     -> it was several: the carried piece and what's left are new "blob"s
                                                 (a piece that stayed together once might still be two touching
                                                 things, so it is only confirmed when carried again as a whole)
    carried, but it landed as several pieces  -> at least two things that moved together: each is a new "blob"
    nothing arrived and the source looks same -> the grip slipped: same blob, one more try counted
    raked                                     -> whatever is in the raked area now is a new "blob" each

Things nobody touched are matched to the fresh scan by footprint overlap and colour and keep their ids ("unchanged");
things that got nudged (close by, similar size, same colour) keep their ids too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.geometry import SURFACE_Z
from lab.fusion import CELL, X_RANGE, Y_RANGE
from lab.perception import MIN_HEIGHT_MM, mean_colour

NY = int(np.ceil((Y_RANGE[1] - Y_RANGE[0]) / CELL))
STAGING = (np.arange(0.35, 0.475, 0.03), np.arange(0.13, 0.26, 0.03))  # x, y grid of spots to set things down (m)


def footprint(points: np.ndarray) -> np.ndarray:
    """Flat grid-cell indices (1 mm) under the parts of the points that stand above the table."""
    p = points[(points[:, 2] - SURFACE_Z) * 1000 > MIN_HEIGHT_MM]
    ix = ((p[:, 0] - X_RANGE[0]) / CELL).astype(int)
    iy = ((p[:, 1] - Y_RANGE[0]) / CELL).astype(int)
    return np.unique(ix * NY + iy)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


def colour_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two things' average colours (0..441); 0 when either has no colour."""
    ca, cb = mean_colour(a), mean_colour(b)
    return 0.0 if ca is None or cb is None else float(np.linalg.norm(ca - cb))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Share of a covered by b."""
    return len(np.intersect1d(a, b, assume_unique=True)) / max(len(a), 1)


@dataclass
class Thing:
    id: int
    points: np.ndarray
    status: str = "blob"            # "blob" untested, "one" moved as one rigid piece, "stuck" can't grip/split/reach
    parent: int | None = None
    history: list[str] = field(default_factory=list)
    model: object = None            # lab.objects.Model once it's confirmed as one object
    tries: int = 0                  # grips that slipped
    rakes: int = 0                  # rakes that didn't split it

    def __post_init__(self):
        self.fp = footprint(self.points)

    @property
    def centroid(self) -> np.ndarray:
        return self.points[:, :3].mean(0)

    @property
    def area(self) -> int:
        return len(self.fp)

    @property
    def height_mm(self) -> float:
        return float((self.points[:, 2].max() - SURFACE_Z) * 1000)

    @property
    def radius(self) -> float:
        return float(np.max(np.linalg.norm(self.points[:, :2] - self.centroid[:2], axis=1)))


class World:
    def __init__(self):
        self.things: dict[int, Thing] = {}
        self.next_id = 1
        self.reserved: list[np.ndarray] = []  # staging spots promised to things in flight

    def _new(self, points, status="blob", parent=None, note="") -> Thing:
        t = Thing(self.next_id, points, status, parent, [note] if note else [])
        self.things[t.id] = t
        self.next_id += 1
        return t

    def first_scan(self, segs) -> list[str]:
        for s in segs:
            self._new(s.points, note="found in the first scan")
        n = len(segs)
        return [f"I see {n} separate thing{'s' if n != 1 else ''}. Each one is a hypothesis: \"this is one object\". "
                f"None tested yet."]

    def blobs(self) -> list[Thing]:
        return [t for t in self.things.values() if t.status == "blob"]

    def mark_out_of_reach(self, t: Thing) -> str:
        t.status = "stuck"
        t.history.append("out of reach")
        return f"#{t.id} is out of the arm's reach ({t.centroid[0] * 1000:.0f}, {t.centroid[1] * 1000:.0f} mm). Leaving it."

    # ------------------------------------------------------------ comparing a fresh scan with what we expected

    def track(self, points: np.ndarray, skip: int | None = None) -> tuple[np.ndarray, list[str], set[int]]:
        """Work off the models: fit every modelled object (except `skip`, the one just acted on) to the new points where
        it should be. A good fit updates its pose and claims those points. Returns (leftover points, events, tracked ids)."""
        from lab import objects
        events, tracked = [], set()
        free = np.ones(len(points), bool)
        for t in self.things.values():
            if t.model is None or t.id == skip:
                continue
            pose, fitness, rms = objects.fit(t.model, points[free])
            if fitness < 0.5:
                events.append(f"#{t.id} isn't where my model of it says (only {fitness:.0%} of it fits). Checking from scratch.")
                continue
            moved = np.linalg.norm(pose[:3, 3] - t.model.pose[:3, 3]) * 1000
            t.model.pose = pose
            mine = objects.claim(t.model, points) & free
            t.points, t.fp = points[mine], footprint(points[mine])
            free &= ~mine
            tracked.add(t.id)
            if moved > 2:
                events.append(f"#{t.id} moved {moved:.0f} mm; my model followed it (fit {fitness:.0%}, {rms:.1f} mm off).")
        return points[free], events, tracked

    def update(self, segs, action: dict | None, tracked: set[int] = frozenset()) -> list[str]:
        new = [(s.points, footprint(s.points)) for s in segs]
        used: set[int] = set()
        events: list[str] = []
        target = self.things.get(action["target"]) if action else None
        old = {i: t for i, t in self.things.items() if t is not target and i not in tracked}

        # 1) untouched things: same footprint, same height -> same thing
        for t in list(old.values()):
            best, bi = 0.0, None
            for k, (pts, fp) in enumerate(new):
                if k in used:
                    continue
                v = iou(t.fp, fp)
                if v > best:
                    best, bi = v, k
            if bi is not None and best > 0.5 and colour_gap(t.points, new[bi][0]) < 60:
                t.points, t.fp = new[bi][0], new[bi][1]
                used.add(bi)
                old.pop(t.id)

        # 2) the thing we acted on
        if target is not None:
            events += self._judge(target, action, new, used)

        # 2b) things that got nudged: close by, similar size, same colour -> same thing, moved a little
        for t in list(old.values()):
            cands = [k for k, (pts, fp) in enumerate(new) if k not in used
                     and np.hypot(*(pts[:, :2].mean(0) - t.centroid[:2])) < 0.03 and 0.6 < len(fp) / max(t.area, 1) < 1.6
                     and colour_gap(t.points, pts) < 45]
            if cands:
                k = min(cands, key=lambda k: np.hypot(*(new[k][0][:, :2].mean(0) - t.centroid[:2])))
                moved = np.hypot(*(new[k][0][:, :2].mean(0) - t.centroid[:2])) * 1000
                t.points, t.fp = new[k][0], new[k][1]
                t.model = None
                used.add(k)
                old.pop(t.id)
                events.append(f"#{t.id} got nudged {moved:.0f} mm (same size, same colour): still #{t.id}.")

        # 3) leftovers
        for k, (pts, fp) in enumerate(new):
            if k not in used:
                t = self._new(pts, note="appeared (knocked loose or uncovered)")
                events.append(f"#{t.id} appeared: knocked loose or uncovered. New blob.")
        for t in old.values():
            self.things.pop(t.id, None)
            events.append(f"lost track of #{t.id} (it moved, or something covers it now)")
        self.reserved = []
        return events

    def _judge(self, T: Thing, action: dict, new, used) -> list[str]:
        src = [k for k, (pts, fp) in enumerate(new) if k not in used and overlap(fp, T.fp) > 0.2]
        if action["kind"] == "carry":
            gx, gy, gyaw = action["grip"]
            px, py, pyaw = action["place"]
            dest = [k for k, (pts, fp) in enumerate(new) if k not in used and k not in src
                    and np.hypot(*(pts[:, :2].mean(0) - [px, py])) < 0.03]
            fell = ""
            if not dest:  # not at the drop spot: did it land somewhere else, on its own? That's good enough.
                loose = [k for k, (pts, fp) in enumerate(new) if k not in used and k not in src
                         and self._clear_of_others(pts, new, k)]
                if loose:
                    k = min(loose, key=lambda k: _dist_to_segment(new[k][0][:, :2].mean(0), np.array([gx, gy]), np.array([px, py])))
                    dest = [k]
                    c = new[k][0][:, :2].mean(0) * 1000
                    fell = (f"It slipped out on the way and landed at ({c[0]:.0f}, {c[1]:.0f}) mm, clear of everything: "
                            f"good enough, it only needed to be on its own. ")
            for k in src + dest:
                used.add(k)
            if not dest:
                if len(src) == 1 and iou(new[src[0]][1], T.fp) > 0.5:
                    T.points, T.fp = new[src[0]][0], new[src[0]][1]
                    T.tries += 1
                    T.history.append("grip slipped: nothing arrived")
                    return [f"Nothing arrived at the drop spot and #{T.id} looks the same: the grip slipped. "
                            f"Still a blob (try {T.tries})."]
                self.things.pop(T.id)
                kids = [self._new(new[k][0], parent=T.id, note=f"disturbed when gripping #{T.id}") for k in src]
                return [f"Nothing arrived, but #{T.id} changed: now {self._ids(kids)} (blobs)."]
            if len(dest) == 1 and not src:
                T.points, T.fp = new[dest[0]][0], new[dest[0]][1]
                T.model = None
                T.status = "one"
                T.history.append("lifted and moved as one piece, nothing left behind")
                return [fell + f"#{T.id} came along whole and nothing stayed behind: it moved as one piece. "
                        f"Hypothesis confirmed: #{T.id} is one object."]
            self.things.pop(T.id)
            if len(dest) == 1:
                piece = self._new(new[dest[0]][0], "blob", T.id, f"came out of #{T.id} in one piece; not yet carried as a whole")
                rest = [self._new(new[k][0], parent=T.id, note=f"left behind when part of #{T.id} was carried off") for k in src]
                return [fell + f"Only part of #{T.id} came along. Hypothesis rejected: #{T.id} was several things. "
                        f"The piece I carried is #{piece.id}; it stayed together, but that could still be two touching "
                        f"things, so it gets tested on its own. What stayed behind: {self._ids(rest)} "
                        f"(new blob{'s' if len(rest) != 1 else ''})."]
            landed = [self._new(new[k][0], parent=T.id, note=f"carried out of #{T.id} together, landed apart") for k in dest]
            rest = [self._new(new[k][0], parent=T.id, note=f"left behind by #{T.id}") for k in src]
            return [f"What I carried from #{T.id} landed as {len(landed)} pieces ({self._ids(landed)}): several things "
                    f"that moved together. Left behind: {self._ids(rest) or 'nothing'}."]
        # rake: everything that's now where the target was, or along the rake line, is a new blob
        s, e = np.asarray(action["line"][0]), np.asarray(action["line"][1])
        near = [k for k, (pts, fp) in enumerate(new) if k not in used and k not in src
                and _dist_to_segment(pts[:, :2].mean(0), s, e) < T.radius + 0.03]
        kids_k = src + near
        for k in kids_k:
            used.add(k)
        self.things.pop(T.id)
        kids = [self._new(new[k][0], parent=T.id, note=f"after raking through #{T.id}") for k in kids_k]
        if len(kids) == 1 and action.get("spread"):
            kids[0].tries = T.tries
            return [f"Raked through #{T.id}: still one clump (#{kids[0].id}). I'll pick pieces off it instead."]
        if len(kids) == 1:  # it didn't split: carry the history over, and give up after the second time
            k = kids[0]
            k.tries, k.rakes = T.tries, T.rakes + 1
            if k.rakes >= 2:
                k.status = "stuck"
                k.history.append("raked twice without splitting, and no grip holds it")
                return [f"Raked #{T.id} again and it's still one piece (#{k.id}), and nothing I try can grip it. "
                        f"Probably one object I can't pick up with these fingers. Leaving it there."]
            return [f"Raked through #{T.id}: still one piece (#{k.id}). One more try later."]
        return [f"Raked through #{T.id}: it's now {len(kids)} separate things ({self._ids(kids)}). Each is a new hypothesis."]

    def _clear_of_others(self, pts, new, k, gap: float = 0.012) -> bool:
        """Is this new segment on its own: nothing else (new or known) within gap of it?"""
        xy = pts[:, :2]
        others = [p for j, (p, _) in enumerate(new) if j != k]
        for p in others:
            if np.min(np.linalg.norm(p[::7, None, :2] - xy[None, ::7], axis=2)) < gap:
                return False
        return True

    @staticmethod
    def _ids(things) -> str:
        return ", ".join(f"#{t.id}" for t in things)

    # ------------------------------------------------------------ where to put things

    def staging_spot(self, radius: float) -> np.ndarray | None:
        """A free spot in the staging area: nothing within radius + 15 mm, not promised to anything else."""
        for x in STAGING[0]:
            for y in STAGING[1]:
                p = np.array([x, y])
                clear = all(np.hypot(*(t.centroid[:2] - p)) > radius + t.radius + 0.015 for t in self.things.values())
                clear &= all(np.hypot(*(q - p)) > 0.03 for q in self.reserved)
                if clear:
                    self.reserved.append(p)
                    return p
        return None


def _dist_to_segment(p, a, b) -> float:
    d = b - a
    f = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + f * d)))
