"""Sort: physical intelligence and an expert VLM working together, from the first look to the last box.

    1. glance at the whole table (the table camera's depth: coarse, but enough to spot the pile and any boxes),
       then look closely with the wrist camera; find things, and boxes (big, tall, hollow)
    2. from then on the VLM runs alongside, in the background:
         - every thing gets a close-up and a question: what is it, how many objects in that outline, how sure,
           does anything look attached (lab/labels.py)
         - the director looks at the whole table with every thing numbered and decides: what this is, the sorting
           rule (the user's, if given), which box each group goes into, which things it is sure of, which to check
           (lab/director.py); asked again whenever new things appear
       the arm never waits for it while there is useful physical work to do
    3. each turn, the most useful action:
         - a thing the VLM is sure is one familiar object, alone on the table, with a box assigned: carry it straight
           into its box. Gone from the table, nothing left behind, nothing dropped: it was one object AND it's sorted.
           Part left behind: it was several (the rest are new things to look at)
         - a big clump: rake through it to spread it out (like tipping out a jigsaw)
         - a thing the VLM isn't sure of: test it physically (grip and carry it next to the box it will probably go
           in, so the last carry is short; or rake); once it has moved as one piece, it's sortable
         - nothing to do until the VLM answers: hold still for a few seconds (the clock keeps running, as on the
           real cell); if the answer is slow, test something meanwhile rather than stand there
       every touch counts: rakes push things toward the boxes and never out of the arm's reach
    4. anything that vanishes without a trace: look around the table for it
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from calibration.geometry import depth_to_base
from lab import grasp, perception, primitives as P, viz
from lab.director import Director, _name
from lab.fusion import X_RANGE, Y_RANGE, carve_merge, height_map
from lab.policies.explore import PUSH_M, REACH_MARGIN_MM, ExplorePolicy
from lab.policies.scan import camera_offset_mm
from lab.world import footprint

TRUST = 0.85        # VLM confidence at which an untested thing is carried straight to its box
WAIT_S = 2.0        # hold still this long at a time while waiting for the VLM
BOX_GAP_M = 0.035   # rakes leave things at least this far from a box: room for the fingers to get at them
PATIENCE_S = 15.0   # stand still at most this long in a row while there's something to test meanwhile (a close-up
                    # usually lands in 3-15 s, and a test costs ~12 s of arm time plus a touch: waiting is cheaper)
MAX_WAIT_S = 240.0  # after this much waiting in all, stop waiting: test things physically, without names


class SortPolicy(ExplorePolicy):
    def __init__(self, checkpoint: str | None = None, budget: int = 40, label_model: str | None = None,
                 instruction: str = "", trust: float = TRUST):
        self.trust = trust
        super().__init__(checkpoint, budget, label_model=label_model, name_things=True, instruction=instruction)

    def reset(self) -> None:
        super().reset()
        self.director = Director(self.labeler, self.cams, self.instruction)
        self.waited = 0.0  # seconds stood still for the VLM, in all
        self.streak = 0.0  # ... since the last move
        self.forced = 0

    # ------------------------------------------------------------ the loop

    def _loop(self, obs):
        off = camera_offset_mm(obs["joints"])
        viz.narrate(f"Instruction: \"{self.instruction or 'Sort these.'}\". First a glance at the whole table, then a close look.")
        spots = self._glance(obs)
        obs, pts = yield from self._sweep(obs, off, spots)
        segs, _ = perception.segment(pts)
        obs, pts = yield from self._look_more(obs, pts, [footprint(sg.points) for sg in segs], self._pile_center_of(segs),
                                              max_looks=3, what="the pile")
        segs, _ = perception.segment(pts)
        self._say(self.world.first_scan(segs) + self.world.find_bins())
        if not self.world.bins():
            viz.narrate("No boxes on the table: groups will be laid out on the table instead.", level="WARN")
        self._think(obs)
        self._show(obs)
        obs, pts = yield from self._spread(obs, off, pts)
        n, turns = 0, 0
        while n < self.budget and turns < 10 * self.budget:  # n counts moves; turns also count looks and waits
            turns += 1
            self._think(obs)
            if self.world.missing:
                obs, pts = yield from self._search(obs, off, pts)
                continue
            todo = [t for t in self.world.things.values() if t.status in ("blob", "one")]
            if not todo:
                viz.narrate(f"Done: nothing left on the table to sort. {len(self.world.sorted)} things went into boxes.")
                break
            for t in todo:
                if not P.reachable(*t.centroid[:2]):
                    self._say([self.world.mark_out_of_reach(t)])
            todo = [t for t in todo if t.status in ("blob", "one")]
            self._decide(todo)
            self._show(obs)
            if all(t.plan.startswith("done") for t in todo):
                viz.narrate(f"Done: {len(self.world.sorted)} things in boxes; the rest stay on the table "
                            f"({', '.join(f'#{t.id}' for t in todo)}).")
                break
            act = None
            for t in sorted([t for t in todo if t.plan.startswith("sort") and t.tries < 4], key=lambda t: (t.status != "one", t.tries)):
                act = self._sort_action(obs, pts, t)
                if act:
                    break
            if act is None:
                big = [t for t in todo if t.status == "blob" and t.area > 350 and t.plan.startswith("test")]
                tests = [t for t in todo if t.plan.startswith("test") and t.tries < 2]
                if big or tests:
                    T = (big or sorted(tests, key=lambda t: (t.tries, t.rakes, t.area)))[0]
                    act = self._test_action(obs, pts, T)
            if act is None:
                unseen = [t for t in todo if self.labeler.needs_look(t)]
                if unseen:  # no picture shows it well enough to ask about it: go and look
                    rows, self.note = P.look(obs["tcp_pose"], unseen[0].centroid[:2], off)
                    viz.narrate(f"No good picture of #{unseen[0].id} yet: one look straight at it.", level="DEBUG")
                    obs = yield rows
                    continue
                if (self.director.busy() or self.labeler.pending) and self.waited < MAX_WAIT_S:
                    idle = [t for t in todo if t.plan.startswith("waiting") and t.status == "blob" and t.tries < 2]
                    if self.streak < PATIENCE_S or not idle:
                        obs = yield from self._wait(obs)
                        continue
                    T = max(idle, key=lambda t: t.area)  # the answer is slow: do something useful meanwhile
                    T.plan = "test: no answer yet, so test it meanwhile"
                    act = self._test_action(obs, pts, T)
                if self.waited >= MAX_WAIT_S:  # the VLM is too slow today: fall back on physics
                    slow = [t for t in todo if t.plan.startswith("waiting") and t.status == "blob" and t.tries < 2]
                    if slow:
                        slow[0].plan = "test: no answer from the VLM"
                        act = self._test_action(obs, pts, slow[0])
            if act is None and self.forced < 3 and any(t.plan.startswith("waiting for the director") for t in todo):
                self.forced += 1  # nothing else to do and the plan doesn't settle it: ask again
                self.director.again()
                viz.narrate("The plan leaves some things undecided; asking the director again.", level="DEBUG")
                continue
            if act is None:
                left = [t for t in todo if t.status in ("blob", "one")]
                viz.narrate(f"Nothing more I can do: {', '.join(f'#{t.id} ({t.plan})' for t in left)}.", level="WARN")
                break
            rows, action, spots, msg = act
            n += 1
            self.streak = 0.0
            viz.narrate(f"Move {n}. {msg}")
            self._moving()
            obs = yield rows
            viz.narrate("Look where it was" + (" and into the box." if action["kind"] == "sort" else " and where it went."))
            obs, fresh = yield from self._sweep(obs, off, spots)
            pts = carve_merge(pts, fresh)
            rest, events, tracked = self.world.track(pts, skip=action["target"])
            segs, _ = perception.segment(rest)
            self._say(events + self.world.update(segs, action, tracked) + self.world.find_bins())
            viz.clear_plan()
            obs, pts = yield from self._model_new(obs, pts)
            self._show(obs)
        if n >= self.budget:
            viz.narrate(f"Used my budget of {self.budget} moves.", level="WARN")
        yield from self._finish(obs)

    # ------------------------------------------------------------ deciding

    def _decide(self, todo) -> None:
        """For each thing: sort it now, test it first, or wait for the VLM. Stored on the thing (shown in the roster)."""
        for t in todo:
            lab = getattr(t, "label", None)
            g = self.director.group_of(t.id)
            box = g.get("box") if g else None
            box = box if box in self.world.things and self.world.things[box].status == "bin" else None
            t.group, t.dest = (g.get("name") if g else None), box
            why_check = self.director.check(t.id)
            clash = self.director.conflict(t.id, self.world)
            if self.labeler.stale(t):
                t.plan = "waiting for its close-up" + (" (it changed)" if lab else "")
                continue
            if clash:
                t.plan = f"waiting for the director: {clash}"
                continue
            if lab and t.id in self.director.named_then and self.director.named_then[t.id] != _name(t):
                t.plan = "waiting for the director: renamed since its plan"
                continue
            if t.status == "one":  # physically certain it's one object
                t.plan = (f"sort → box #{box}" if box else f"done: stays on the table ({g.get('name')} has no box)" if g
                          else "waiting for the director")
                continue
            if lab is None:
                t.plan = "waiting for its close-up" if self.labeler.on else "test (no VLM)"
                continue
            conf = float(lab.get("confidence") or 0)
            count = int(lab.get("count") or 1)
            if why_check:
                t.plan = f"test: director says {why_check}"
            elif count > 1:
                t.plan = f"test: looks like {count} things"
            elif lab.get("attached"):
                t.plan = "test: looks attached"
            elif not self.world.isolated(t):
                t.plan = "test: touching something"
            elif conf >= self.trust or (self.director.sure(t.id) and conf >= 0.7):
                t.plan = (f"sort → box #{box} (sure: {conf:.0%})" if box else
                          "test: sure what it is, but its group has no box: confirm it in place" if g else "waiting for the director")
            else:
                t.plan = f"test: VLM only {conf:.0%} sure"

    # ------------------------------------------------------------ actions

    def _sort_action(self, obs, pts, T):
        """Carry T straight into its box (or its group's spot on the table)."""
        good, bad = grasp.find(self._scene(pts), self._target(T))
        if not good:
            T.plan = "test: no grip from here"
            return None
        g = good[0]
        lab = getattr(T, "label", None) or {}
        what = f"{lab.get('color', '')} {lab.get('name', 'thing')}".strip()
        B = self.world.things[T.dest]
        place = B.centroid[:2]
        drop = B.height_mm + 5
        spots = [T.centroid[:2], place]
        if not P.reachable(*place):
            return None
        safe = max([P.SAFE_MM] + [b.height_mm + 30 for b in self.world.bins()])
        viz.plan_grip(good[1:], bad, g, place)
        rows, note = P.pick_and_place(obs["tcp_pose"], (g.x, g.y), g.height_mm, g.yaw, place, drop_mm=drop, safe_mm=safe)
        self.note = note
        action = {"kind": "sort", "target": T.id, "grip": (g.x, g.y, g.yaw), "place": (place[0], place[1]), "bin": T.dest}
        how = ("it has already moved as one piece" if T.status == "one" else
               f"the VLM is sure it's one {what} ({float(lab.get('confidence') or 0):.0%}) and it's alone, so no test: "
               f"if it arrives whole, that's the test")
        return rows, action, spots, f"#{T.id} ({what}) goes into box #{T.dest} ({T.group}): {how}."

    def _test_action(self, obs, pts, T):
        """The physical test from the explore policy: grip and carry to a clear spot, or rake if nothing grips."""
        good, bad = grasp.find(self._scene(pts), self._target(T)) if T.tries < 2 else ([], [])
        if good and T.area <= 350:
            g, why = self._test_grip(T, good)
            near, where = self._stage_near(T)
            spot = self.world.staging_spot(T.radius if T.area < 400 else 0.02, near=near)
            if spot is None:
                return None
            viz.plan_grip(good[1:], bad, g, spot)
            rows, self.note = P.pick_and_place(obs["tcp_pose"], (g.x, g.y), g.height_mm, g.yaw, spot, safe_mm=self._safe_mm())
            action = {"kind": "carry", "target": T.id, "grip": (g.x, g.y, g.yaw), "place": (spot[0], spot[1], g.yaw)}
            return rows, action, [np.array(spot), T.centroid[:2]], f"Test #{T.id} ({T.plan}): {why}; carry it {where}."
        line = self._safe_rake(T)
        if line is None:
            self._say([f"#{T.id}: can't grip it and can't rake it without hitting a box. Leaving it."])
            T.status = "stuck"
            return None
        s, e = line
        viz.plan_rake(s, e, 3.0)
        rows, self.note = P.rake(obs["tcp_pose"], s, e)
        action = {"kind": "rake", "target": T.id, "line": (s, e)}
        return rows, action, [(s + e) / 2], f"#{T.id} ({T.plan}): rake through it to spread it out."

    def _stage_near(self, T):
        """Where to set T down after testing it: by its box if the director has given it one, else by the boxes."""
        bins = self.world.bins()
        if T.dest in self.world.things:
            return self.world.things[T.dest].centroid[:2], f"next to box #{T.dest}, where it will probably go"
        if bins:
            return np.mean([b.centroid[:2] for b in bins], axis=0), "over by the boxes"
        return None, "to a clear spot"

    def _safe_mm(self) -> float:
        return max([P.SAFE_MM] + [b.height_mm + 30 for b in self.world.bins()])

    def _safe_rake(self, T):
        """A rake line through T (across it or along it, either way) that stays clear of every box, and leaves what it
        pushes well inside the arm's reach. Of those, the one that pushes things toward the boxes (less to carry
        later); across beats along (it spreads a clump better)."""
        bins = self.world.bins()
        best, best_score = None, np.inf
        for along in (False, True):
            s, e = self._rake_line(T, along=along)
            for a, b in ((s, e), (e, s)):
                u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                push = b + u * PUSH_M  # where the pushed parts end up
                if not P.reachable(*push, margin_mm=REACH_MARGIN_MM):
                    continue
                if any(_dist_to_segment(x.centroid[:2], a, push) < x.radius + BOX_GAP_M for x in bins):
                    continue
                score = (min(np.hypot(*(x.centroid[:2] - push)) for x in bins) if bins else np.hypot(*push)) + 0.03 * along
                if score < best_score:
                    best, best_score = (a, b), score
        if best is None and bins:  # hard against a box: start between it and the box, push straight away from the box
            c = T.points[:, :2].mean(0)
            B = min(bins, key=lambda x: np.hypot(*(x.centroid[:2] - c)))
            u = (c - B.centroid[:2]) / max(np.hypot(*(c - B.centroid[:2])), 1e-9)
            a, b = c - u * (T.radius + 0.006), c + u * (T.radius + 0.04)
            if (np.hypot(*(a - B.centroid[:2])) > 0.72 * B.radius + 0.006  # fingers land outside the box's wall
                    and P.reachable(*(b + u * PUSH_M), margin_mm=REACH_MARGIN_MM)):
                best = (a, b)
        return best

    def _spread(self, obs, off, pts, max_rakes: int = 4, big_mm2: int = 350):
        """As in explore, but never rake into a box."""
        for k in range(max_rakes):
            self._think(obs)
            big = [t for t in self.world.blobs() if t.area > big_mm2]
            if not big:
                if k:
                    viz.narrate("The pile is spread out: no big clumps left.")
                break
            T = max(big, key=lambda t: t.area)
            line = self._safe_rake(T)
            if line is None:
                break
            s, e = line
            viz.plan_rake(s, e, 3.0)
            viz.narrate(f"Spread {k + 1}: #{T.id} is a big clump ({T.area} mm²). Rake through it to spread the pile out.")
            rows, self.note = P.rake(obs["tcp_pose"], s, e)
            self._moving()
            obs = yield rows
            obs, fresh = yield from self._sweep(obs, off, [(s + e) / 2])
            pts = carve_merge(pts, fresh)
            rest, events, tracked = self.world.track(pts, skip=T.id)
            segs, _ = perception.segment(rest)
            self._say(events + self.world.update(segs, {"kind": "rake", "target": T.id, "line": (s, e), "spread": True}, tracked))
            viz.clear_plan()
            self._show(obs)
        return obs, pts

    def _search(self, obs, off, pts):
        """Something vanished: look over the whole work area for it (two wide wrist looks cover the table)."""
        gone = ", ".join(f"#{t.id}" for t in self.world.missing)
        viz.narrate(f"Lost {gone}. Looking where it was last seen, then around the table.")
        last = [t.centroid[:2].copy() for t in self.world.missing if P.reachable(*t.centroid[:2])]
        self.world.missing = []
        spots = []
        for q in last:  # one look covers ~480 x 270 mm: no need to look twice at nearly the same place
            if all(np.hypot(*(q - r)) > 0.06 for r in spots):
                spots.append(q)
        obs, fresh = yield from self._sweep(obs, off, spots[:3] + [np.array([0.42, -0.14]), np.array([0.42, 0.14])])
        pts = carve_merge(pts, fresh)
        rest, events, tracked = self.world.track(pts)
        segs, _ = perception.segment(rest)
        self._say(events + self.world.update(segs, None, tracked) + self.world.find_bins())
        self._show(obs)
        return obs, pts

    def _wait(self, obs):
        """Hold still while the VLM thinks (the clock runs, as it would on the real cell)."""
        self.waited += WAIT_S
        self.streak += WAIT_S
        self.note = "waiting for the VLM"
        if self.waited % 10 < WAIT_S:
            viz.narrate(f"Waiting for the VLM ({'director' if self.director.busy() else 'close-ups'}): {self.waited:.0f} s so far.", level="DEBUG")
        self._think(obs, wait=WAIT_S)
        rows = P.hold(np.asarray(obs["tcp_pose"][:3], float), float(obs["tcp_pose"][5]), 0, int(WAIT_S / P.DT))
        return (yield rows)

    def _finish(self, obs):
        self._name(wait=60.0)
        self._show(obs)
        counts = {}
        for s in self.world.sorted:
            counts.setdefault(s["bin"], []).append(f"#{s['id']} {s.get('color') or ''} {s.get('name') or '?'}".strip())
        viz.narrate("Sorted: " + ("; ".join(f"box #{b}: {', '.join(v)}" for b, v in counts.items()) or "nothing")
                    + f". Waited {self.waited:.0f} s for the VLM in all.")
        return
        yield

    # ------------------------------------------------------------ the VLM side, in the background

    def _think(self, obs, wait: float = 0.0) -> None:
        """Ask about new things, pick up answers, re-plan when the table has changed."""
        self._name(wait=0.0)
        r = self.director.poll(wait=wait if self.director.busy() else 0.0)
        if r is not None:
            viz.label_crop({"id": "plan", "picture": r["image"], "answer": None})
            plan = r["plan"]
            if plan:
                viz.vlm(f"Director ({(r['model'] or '').split('/')[-1]}, {r['seconds']:.0f} s): {plan.get('scene')} State: {plan.get('state')}. Rule: {plan.get('rule')}")
                viz.vlm("Groups: " + "; ".join(f"{g.get('name')} → box {g.get('box')}: " + ", ".join(f"#{i}" for i in g.get("ids") or [])
                                               for g in plan.get("groups", [])))
                if plan.get("sure"):
                    viz.vlm("Sure about: " + ", ".join(f"#{i}" for i in plan["sure"]))
                for c in plan.get("check") or []:
                    if isinstance(c, dict):
                        viz.vlm(f"Check #{c.get('id')}: {c.get('why')}", level="WARN")
                if plan.get("instruction_check"):
                    viz.vlm(f"Instruction vs what I see: {plan['instruction_check']}")
                viz.narrate(f"Director: {plan.get('scene')} Rule: {plan.get('rule')}")
            else:
                viz.vlm(f"Director gave no usable plan ({'; '.join(r['tries'])})", level="WARN")
        if not self.director.busy() and self.director.stale(self.world):
            # ask once most close-ups are back (or nothing is pending), so the plan isn't made blind
            things = [t for t in self.world.things.values() if t.status in ("blob", "one")]
            named = sum(bool(getattr(t, "label", None)) for t in things)
            if things and (named >= 0.7 * len(things) or not self.labeler.pending or self.director.forced):
                msg = self.director.request(self.world, obs)
                if msg:
                    viz.vlm(msg, level="DEBUG")
        self.version += 1

    def _name(self, wait: float = 0.0) -> None:
        """Every thing gets a close-up question (blobs too), as soon as a picture shows it."""
        if self.labeler is None:
            return
        for t in self.world.things.values():
            msg = self.labeler.ask(t)
            if msg:
                viz.vlm(msg, level="DEBUG")
        for r in self.labeler.poll(wait):
            t = self.world.things.get(r["id"])
            a = r["answer"]
            viz.label_crop(r)
            if a is None:
                viz.vlm(f"#{r['id']}: no usable answer ({r['error'] or r['text'][:200]})", level="WARN")
                continue
            if t is not None:
                t.label = a
                if t.status == "blob" and "box" in str(a.get("name", "")).lower() and t.area > 1200:
                    t.status = "bin"
            viz.vlm(f"#{r['id']}: {a.get('color', '')} {a.get('name')} ({a.get('category')}), count {a.get('count', 1)}, "
                    f"{float(a.get('confidence') or 0):.0%} sure" + (", looks attached" if a.get("attached") else "")
                    + f" ({r['seconds']:.0f} s). {a.get('why', '')}")

    @staticmethod
    def _pile_center_of(segs) -> np.ndarray:
        if not segs:
            return np.array([0.43, 0.0])
        big = max(segs, key=lambda s: len(s.points))
        return big.points[:, :2].mean(0)

    def _glance(self, obs) -> list[np.ndarray]:
        """Where to look closely first. The table camera's depth is coarse (a point every 2-3 mm), so: a 5 mm grid,
        anything 5 mm above the table, connected clumps; then wrist looks to cover them all (a look spans ~480 x 270 mm)."""
        d = obs.get("depth", {}).get("table")
        if d is None:
            return [self.pile_center]
        p = perception.drop_self(depth_to_base(self.cams["table_depth"], d, stride=1), obs["tcp_pose"])
        hm = height_map(p, cell=0.005)
        lab, n = ndimage.label(np.nan_to_num(hm, nan=0.0) > 5.0)
        found = []
        for k in range(1, n + 1):
            ij = np.argwhere(lab == k)
            if len(ij) < 3:
                continue
            c = np.array([X_RANGE[0], Y_RANGE[0]]) + (ij.mean(0) + 0.5) * 0.005
            if P.reachable(*c):
                found.append((len(ij), c))
        if not found:
            return [self.pile_center]
        spots = []
        for _, c in sorted(found, key=lambda f: -f[0]):
            if all(np.hypot(*(c - q)) > 0.12 for q in spots):
                spots.append(c)
        viz.narrate(f"The table camera shows {len(found)} things or clumps; {len(spots)} close look{'s' if len(spots) != 1 else ''} will cover them.")
        return spots[:4]

def _dist_to_segment(p, a, b) -> float:
    d = b - a
    f = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + f * d)))
