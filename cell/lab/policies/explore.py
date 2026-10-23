"""Explore: lay the pieces out like a jigsaw puzzle, testing one hypothesis at a time.

    1. sweep: look at the pile from a few close wrist-camera spots, fuse the depth into a map, find the blobs
    2. pick the next hypothesis to test: "blob #k is one object"
    3. test it: grip part of it and carry it to a clear staging spot; or, if nothing can be gripped, rake through it
    4. look again, only where something should have changed (the source and the drop spot, or the raked area), fold
       that into the map (lab/fusion.carve_merge), compare with what was expected (lab/world.py), update the hypotheses
    5. repeat until every thing has moved as one piece, or the action budget runs out

Written as one readable loop: `obs = yield rows` means "play these targets, then hand me the next observation".
The policy wrapper below turns that into the cell's act(obs) -> chunk protocol (100 targets per chunk at most), so the
same code runs on the sim cell and on the real one.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import depth_to_base, load
from lab import grasp, nbv, objects, perception, primitives as P, viz
from lab.fusion import carve_merge, height_map
from lab.world import footprint
from lab.policies.scan import camera_offset_mm
from lab.world import STAGING, World

STAGING_CENTER = (float(STAGING[0].mean()), float(STAGING[1].mean()))


class ExplorePolicy:
    CAMERAS = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = ["table", "wrist"]
    DT = 0.1

    def __init__(self, checkpoint: str | None = None, budget: int = 24, pile_center=(0.43, 0.0)):
        self.budget, self.pile_center = budget, np.asarray(pile_center, float)
        self.cams = load()
        self.reset()

    def reset(self) -> None:
        self.world = World()
        self.gen = None
        self.pending: list = []
        self.done = False
        self.note = ""
        self.version = 0  # bumps whenever the world model changes

    @property
    def wants_obs(self) -> bool:
        return not self.pending

    def act(self, obs: dict):
        if not self.pending:
            try:
                if self.gen is None:
                    self.gen = self._loop(obs)
                    self.pending = list(next(self.gen))
                else:
                    self.pending = list(self.gen.send(obs))
            except StopIteration:
                self.done = True
                self.note = "finished"
                return [list(obs["tcp_pose"][:6]) + [0.0]] * 10
        chunk, self.pending = self.pending[:100], self.pending[100:]
        return chunk

    # ------------------------------------------------------------ the loop

    def _loop(self, obs):
        off = camera_offset_mm(obs["joints"])
        viz.narrate("Before touching anything: one look at the pile from above, then look wherever I can't see yet.")
        c = self._pile_center()
        obs, pts = yield from self._sweep(obs, off, [c])
        segs, _ = perception.segment(pts)
        obs, pts = yield from self._look_more(obs, pts, [footprint(sg.points) for sg in segs], c, max_looks=4,
                                              what="the pile")
        segs, _ = perception.segment(pts)
        self._say(self.world.first_scan(segs))
        self._show(obs)
        for n in range(1, self.budget + 1):
            for t in self.world.blobs():
                if not P.reachable(*t.centroid[:2]):
                    self._say([self.world.mark_out_of_reach(t)])
            blobs = sorted(self.world.blobs(), key=lambda t: (t.tries, t.rakes, t.area))
            if not blobs:
                n1 = sum(t.status == "one" for t in self.world.things.values())
                stuck = len(self.world.things) - n1
                viz.narrate(f"Done exploring: {n1} things moved as one piece each, laid out where I can see them"
                            + (f"; {stuck} I couldn't test (stuck or out of reach)." if stuck else "."))
                return
            T, good, bad = None, [], []
            for t in blobs:
                if t.tries >= 2:
                    continue
                good, bad = grasp.find(self._scene(pts), self._target(t))
                if good:
                    T = t
                    break
            if T is not None:  # before committing: are the sides (and where the fingers go) really clear?
                tid = T.id
                obs, pts = yield from self._look_more(obs, pts, [T.fp], T.centroid[:2], max_looks=2, what=f"#{tid}")
                segs, _ = perception.segment(pts)
                self._say(self.world.update(segs, None))
                T = self.world.things.get(tid)
                good, bad = grasp.find(self._scene(pts), self._target(T)) if T is not None else ([], [])
                if T is None or not good:
                    viz.narrate(f"After a closer look, #{tid} isn't what I thought; choosing again.")
                    continue
            if T is not None:
                g, why = self._test_grip(T, good)
                spot = self.world.staging_spot(T.radius if T.area < 400 else 0.02)
                if spot is None:
                    viz.narrate("The staging area is full. Stopping here.", level="WARN")
                    return
                viz.plan_grip(good[1:], bad, g, spot)
                rows, what = P.pick_and_place(obs["tcp_pose"], (g.x, g.y), g.height_mm, g.yaw, spot)
                action = {"kind": "carry", "target": T.id, "grip": (g.x, g.y, g.yaw), "place": (spot[0], spot[1], g.yaw)}
                viz.narrate(f"Test {n}. Hypothesis: #{T.id} ({T.area} mm², {T.height_mm:.0f} mm tall) is one object. "
                            f"{len(good)} ways to grip it; {why} ({g.material} mm² between the jaws, {g.height_mm:.0f} mm up). "
                            f"Carry it to a clear spot and see what comes along.")
            else:
                T = blobs[0]
                s, e = self._rake_line(T)
                viz.plan_rake(s, e, 3.0)
                rows, what = P.rake(obs["tcp_pose"], s, e)
                action = {"kind": "rake", "target": T.id, "line": (s, e)}
                viz.narrate(f"Test {n}. Nothing I can grip in #{T.id} ({T.area} mm²): the fingers have nowhere to go. "
                            f"Rake through it to shuffle it apart.")
            self.note = what
            obs = yield rows
            if action["kind"] == "carry":
                spots = [np.array(action["place"][:2]), T.centroid[:2]]
                viz.narrate("Look where it went, then where it was.")
            else:
                s_, e_ = action["line"]
                spots = [(s_ + e_) / 2]
                viz.narrate("Look at the raked area.")
            obs, fresh = yield from self._sweep(obs, off, spots)
            pts = carve_merge(pts, fresh)
            rest, events, tracked = self.world.track(pts, skip=action["target"])  # known objects first
            segs, _ = perception.segment(rest)                                      # then whatever's left
            self._say(events + self.world.update(segs, action, tracked))
            viz.clear_plan()
            obs, pts = yield from self._model_new(obs, pts)
            self._show(obs)
        viz.narrate(f"Used my budget of {self.budget} tests; {len(self.world.blobs())} blobs still untested.", level="WARN")

    @staticmethod
    def _test_grip(T, good):
        """A test has to be able to fail. Gripping the middle of two touching parts lifts both and proves nothing, so for
        anything bigger than one grip's worth, grab near one end: if it's really several things, only one comes along."""
        if T.area <= 160:  # about one 10 mm cube plus noise
            return good[0], "gripping it in the middle"
        c = T.centroid[:2]
        strong = [g for g in good if g.material >= 0.8 * good[0].material]
        g = max(strong, key=lambda g: np.hypot(g.x - c[0], g.y - c[1]))
        return g, f"gripping near one end ({np.hypot(g.x - c[0], g.y - c[1]) * 1000:.0f} mm from its middle) so that, if it's several things, only one comes along"

    def _scene(self, pts):
        """What the fingers must avoid: everything seen, plus the full surface of every modelled object."""
        extra = [t.model.samples(800) for t in self.world.things.values() if t.model is not None]
        return np.concatenate([pts[:, :3], *extra]) if extra else pts[:, :3]

    @staticmethod
    def _target(t):
        return np.concatenate([t.points[:, :3], t.model.samples(800)]) if t.model is not None else t.points[:, :3]

    def _model_new(self, obs, pts):
        """Every newly confirmed thing gets a solid model: one more look around it if needed, then build."""
        for t in [t for t in self.world.things.values() if t.status == "one" and t.model is None]:
            tid = t.id
            obs, pts = yield from self._look_more(obs, pts, [t.fp], t.centroid[:2], max_looks=1, what=f"#{tid}")
            segs, _ = perception.segment(pts)
            self.world.update(segs, None)
            t = self.world.things.get(tid)
            if t is None:
                continue
            t.model = objects.build(t.points)
            if t.model is not None:
                e = t.model.world_mesh().extents * 1000
                viz.narrate(f"#{tid} is one object, so I built a solid model of it: {t.model.kind}, about "
                            f"{e[0]:.0f} × {e[1]:.0f} × {e[2]:.0f} mm, {t.model.world_mesh().volume * 1e9:.0f} mm³.")
        self.version += 1
        return obs, pts

    def _look_more(self, obs, pts, footprints, target_xy, max_looks: int, what: str):
        """Next best view: keep looking where the most still-unseen table near these footprints would be revealed."""
        skip, before = [], None
        for _ in range(max_looks):
            v, n_unseen = nbv.best_view(pts, footprints, np.asarray(target_xy), obs["tcp_pose"], skip)
            hm = height_map(pts)
            viz.unseen(nbv.unseen_near(hm, footprints), hm.shape)
            if v is None:
                viz.narrate(f"Around {what}: {n_unseen} mm² still unseen, but no view would show enough of it. Moving on."
                            if n_unseen >= nbv.MIN_GAIN else f"I can see all around {what}.")
                break
            viz.next_view(v)
            viz.narrate(f"I can't see {n_unseen} mm² around {what} (purple). Best next look: {v.note}; "
                        f"it should reveal about {v.gain} mm² for {v.seconds:.0f} s of moving.")
            rows, self.note = P.view(obs["tcp_pose"], v.tip_xy, v.yaw)
            obs = yield rows
            pts = np.concatenate([pts, self._points(obs)])  # nothing moved: just add what's new
            revealed = n_unseen - len(nbv.unseen_near(height_map(pts), footprints))
            if revealed < 0.3 * v.gain:
                skip.append((v.tip_xy, v.yaw))
                viz.narrate(f"That look showed only {revealed} mm² of the {v.gain} I expected; I won't try it again.", level="DEBUG")
        viz.unseen(np.zeros(0, int), (1, 1))
        return obs, pts

    def _sweep(self, obs, off, spots):
        """Hold the wrist camera still over each spot in turn; returns (last obs, the points those looks saw)."""
        pts = []
        for spot in spots:
            rows, what = P.look(obs["tcp_pose"], spot, off)
            self.note = what
            obs = yield rows
            pts.append(self._points(obs))
        return obs, np.concatenate(pts)

    def _points(self, obs) -> np.ndarray:
        """What the wrist camera sees now: depth points without the robot itself, coloured from the wrist image."""
        d = depth_to_base(self.cams["wrist_depth"], obs["depth"]["wrist"], stride=2, joints=obs["joints"])
        return perception.colour(perception.drop_self(d, obs["tcp_pose"]), obs, self.cams)

    def _pile_center(self) -> np.ndarray:
        blobs = self.world.blobs()
        if not blobs:
            return self.pile_center
        return np.mean([t.centroid[:2] for t in blobs], axis=0)

    def _rake_line(self, T):
        """Through the middle of the blob, across its long axis, heading away from the staging area; 25 mm past each edge."""
        xy = T.points[:, :2]
        c = xy.mean(0)
        w, v = np.linalg.eigh(np.cov((xy - c).T))
        d = v[:, 0]  # across the long axis
        if np.dot(d, np.array(STAGING_CENTER) - c) > 0:
            d = -d
        reach = T.radius + 0.025
        return c - d * reach, c + d * reach

    def _say(self, events) -> None:
        self.version += 1
        for e in events:
            viz.narrate(e)

    def _show(self, obs) -> None:
        viz.things(self.world, self.cams, obs)
