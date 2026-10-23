"""Explore: lay the pieces out like a jigsaw puzzle, testing one hypothesis at a time.

    1. sweep: look at the pile from a few close wrist-camera spots, fuse the depth, find the blobs
    2. pick the next hypothesis to test: "blob #k is one object"
    3. test it: grip part of it and carry it to a clear staging spot; or, if nothing can be gripped, rake through it
    4. look again, compare with what was expected (lab/world.py), and update the hypotheses
    5. repeat until every thing has moved as one piece, or the action budget runs out

Written as one readable loop: `obs = yield rows` means "play these targets, then hand me the next observation".
The policy wrapper below turns that into the cell's act(obs) -> chunk protocol (100 targets per chunk at most), so the
same code runs on the sim cell and on the real one.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import depth_to_base, load
from lab import grasp, perception, primitives as P, viz
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
        viz.narrate("Before touching anything: a close look at the pile from a few angles.")
        obs, pts = yield from self._sweep(obs, off)
        segs, _ = perception.segment(pts)
        self._say(self.world.first_scan(segs))
        self._show(obs)
        for n in range(1, self.budget + 1):
            blobs = sorted(self.world.blobs(), key=lambda t: (t.tries, t.area))
            if not blobs:
                viz.narrate(f"Every thing on the table has moved as one piece: {len(self.world.things)} objects, "
                            f"laid out where I can see each one. Done exploring.")
                return
            T, good, bad = None, [], []
            for t in blobs:
                if t.tries >= 2:
                    continue
                good, bad = grasp.find(pts, t.points)
                if good:
                    T = t
                    break
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
            obs, pts = yield from self._sweep(obs, off)
            segs, _ = perception.segment(pts)
            self._say(self.world.update(segs, action))
            viz.clear_plan()
            self._show(obs)
        viz.narrate(f"Used my budget of {self.budget} tests; {len(self.world.blobs())} blobs still untested.", level="WARN")

    @staticmethod
    def _test_grip(T, good):
        """A test has to be able to fail. Gripping the middle of two touching parts lifts both and proves nothing, so for
        anything bigger than one grip's worth, grab near one end: if it's really several things, only one comes along."""
        if T.area <= 1.2 * 144:
            return good[0], "gripping it in the middle"
        c = T.centroid[:2]
        strong = [g for g in good if g.material >= 0.6 * good[0].material]
        g = max(strong, key=lambda g: np.hypot(g.x - c[0], g.y - c[1]))
        return g, f"gripping near one end ({np.hypot(g.x - c[0], g.y - c[1]) * 1000:.0f} mm from its middle) so that, if it's several things, only one comes along"

    def _sweep(self, obs, off):
        """Close looks at the pile from three spots and one at the staging area; returns (last obs, fused points)."""
        c = self._pile_center()
        spots = [c, c + [0.035, 0.035], c - [0.035, 0.035], np.array(STAGING_CENTER)]
        pts = []
        for spot in spots:
            rows, what = P.look(obs["tcp_pose"], spot, off)
            self.note = what
            obs = yield rows
            d = depth_to_base(self.cams["wrist_depth"], obs["depth"]["wrist"], stride=2, joints=obs["joints"])
            pts.append(perception.drop_self(d, obs["tcp_pose"]))
        return obs, np.concatenate(pts)

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
