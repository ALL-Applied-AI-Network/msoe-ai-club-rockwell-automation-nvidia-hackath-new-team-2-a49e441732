"""Fill the fish tank: paint the space over the table with evidence as fast as possible, before anything is touched.

1. the table camera's depth, once: most of the table, from one fixed angle (its shadows behind things stay unknown)
2. then, over and over: the best view of the gaps (lab.views: tilted looks allowed), move there, fuse the wrist depth
3. stop when the gaps are gone, the views stop finding anything, or the look budget is spent

Shown live in Rerun (mind/evidence/...): solid voxels coloured by height, the frontier (the gaps) in purple, the next
view's camera, and how full the tank is after each look.
"""
from __future__ import annotations

import time

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
from lab import primitives as P, viz
from lab.evidence import EvidenceMap
from lab.views import best_view

import rerun as rr


class MapperMind:
    IMAGE_SIZE = (640, 360)  # colour is only for showing; the map is built from depth

    def __init__(self, instruction: str = "", budget: int = 24, voxel: float = 0.002, min_frontier: int = 150):
        self.instruction, self.budget, self.voxel, self.min_frontier = instruction, budget, voxel, min_frontier

    def reset(self) -> None:
        self.map = EvidenceMap(self.voxel)
        self.cams = load()
        self.done = False
        self.looks = 0
        self.history: list[dict] = []
        self.gen = None
        self.version = 0
        self.t_compute = 0.0

    def act(self, obs):
        try:
            if self.gen is None:
                self.gen = self._loop(obs)
                return next(self.gen)
            return self.gen.send(obs)
        except StopIteration:  # done (possibly before the first move: nothing left to look at)
            self.done = True
            return P.hold(np.asarray(obs["tcp_pose"][:3], float), float(obs["tcp_pose"][5]), 0, 2)

    # ------------------------------------------------------------ the loop

    def _loop(self, obs):
        t0 = time.time()
        r = self.map.integrate(obs["depth"]["table"], self.cams["table_depth"])
        self._show(obs, f"The table camera's depth: {r['free']:,} voxels seen empty, {r['hit']:,} solid.", t0)
        self._fuse_wrist(obs)
        stall = 0
        while self.looks < self.budget:
            st = self.map.stats()
            if st["frontier_voxels"] < self.min_frontier:
                viz.narrate(f"The gaps are filled: {st['frontier_voxels']} frontier voxels left. Done looking.")
                break
            t0 = time.time()
            q_now = np.asarray(obs["joints"][:6], float)
            tip_now = np.array([obs["tcp_pose"][0], obs["tcp_pose"][1], obs["tcp_pose"][2] - P.UF0_Z_MM]) / 1000
            v = best_view(self.map, self.cams, q_now, tip_now)
            self.t_compute += time.time() - t0
            if v is None:
                viz.narrate("No reachable view sees any of the remaining gaps. Done looking.", level="WARN")
                break
            viz.camera_frustum("mind/evidence/next_view", self.cams["wrist_depth"], v["cam"])
            rows, what = P.move_to_pose(obs["tcp_pose"], v["tip"], v["wpr"])
            viz.narrate(f"Look {self.looks + 1}: {what}; should see {v['sees']} gap voxels "
                        f"({'straight down' if v['tilt'] == 0 else 'tilted %.0f deg' % v['tilt']}), "
                        f"~{v['seconds']:.1f} s of moving (chosen in {time.time() - t0:.1f} s).")
            obs = yield rows
            gained = self._fuse_wrist(obs)
            missed = self.map.missed(v["expected"])
            if missed > 0.5 * len(v["expected"]):
                viz.narrate(f"That look should have seen {len(v['expected'])} gap voxels and {missed} are still unknown: "
                            f"marking them (twice missed = not reachable from here).", level="DEBUG")
            stall = stall + 1 if gained < 200 else 0
            if stall >= 3:
                viz.narrate("Three looks in a row added almost nothing. Done looking.")
                break
        self.version += 1
        self._mesh()

    def _fuse_wrist(self, obs) -> int:
        """Fuse the wrist depth; returns how many voxels got their first evidence."""
        t0 = time.time()
        before = int((self.map.looks > 0).sum())
        r = self.map.integrate(obs["depth"]["wrist"], self.cams["wrist_depth"], joints=obs["joints"])
        self.looks += 1
        gained = int((self.map.looks > 0).sum()) - before
        self._show(obs, f"Look {self.looks} fused: {gained:,} voxels seen for the first time ({r['free']:,} empty, {r['hit']:,} solid).", t0)
        return gained

    # ------------------------------------------------------------ what it shows and reports

    def _show(self, obs, line: str, t0: float) -> None:
        self.t_compute += time.time() - t0
        st = self.map.stats()
        self.history.append({"look": self.looks, "robot_s": None, **st})
        pts = self.map.solid_points()
        if len(pts):
            h = (pts[:, 2] - SURFACE_Z) / 0.06
            cols = (np.stack([0.95 * np.clip(h, 0, 1), 0.55 + 0.3 * (1 - np.clip(h, 0, 1)), 0.9 - 0.6 * np.clip(h, 0, 1)], 1) * 255).astype(np.uint8)
            rr.log("mind/evidence/solid", rr.Points3D(pts, colors=cols, radii=self.map.v * 0.45))
        lean_t = self.map.table_points(sure=False)
        rr.log("mind/evidence/table_leaning", rr.Points3D(lean_t, colors=[[88, 94, 104]], radii=self.map.v * 0.6))
        rr.log("mind/evidence/table", rr.Points3D(self.map.table_points(), colors=[[160, 168, 180]], radii=self.map.v * 0.8))
        lean = self.map.leaning_points()
        rr.log("mind/evidence/leaning", rr.Points3D(lean, colors=[[200, 140, 60, 140]], radii=self.map.v * 0.35))
        fr = self.map.frontier_points()
        if len(fr) > 6000:
            fr = fr[np.random.default_rng(0).choice(len(fr), 6000, replace=False)]
        rr.log("mind/evidence/gaps", rr.Points3D(fr, colors=[[190, 70, 255]], radii=self.map.v * 0.45))
        un = self.map.unseen_points()
        rr.log("mind/evidence/unseen", rr.Points3D(un, colors=[[110, 60, 150, 90]], radii=self.map.v * 0.6))
        rr.log("evidence_top", rr.Image(self.map.top_view()))
        viz.narrate(f"{line} The tank is {st['known_pct']}% known above the table; {st['frontier_voxels']:,} gap voxels left.")
        self.version += 1

    def _mesh(self) -> None:
        m = self.map.surface_mesh()
        if m is not None:
            v, f = m
            rr.log("mind/evidence/surface", rr.Mesh3D(vertex_positions=v, triangle_indices=f, albedo_factor=[200, 205, 215, 255]))

    def status(self) -> str:
        st = self.map.stats()
        return (f"Mapping the table: {self.looks} looks so far, {st['known_pct']}% of the space above the table has "
                f"evidence, {st['solid_voxels']:,} voxels believed solid, {st['frontier_voxels']:,} gap voxels left.")

    def score(self, truth: list[dict]) -> dict:
        """Against the simulator's truth: of the solid evidence above the table, how much is on a real object
        (precision), and how much of the objects' visible surface got solid evidence (coverage)."""
        import trimesh
        objs = [o["mesh"] for o in truth]
        out = {"looks": self.looks, "compute_s": round(self.t_compute, 1), **self.map.stats()}
        if not objs:
            return out
        from scipy.spatial import cKDTree
        allm = trimesh.util.concatenate(objs)
        dense, face = trimesh.sample.sample_surface(allm, 60000, seed=0)  # the true surfaces, finely sampled
        pts = self.map.solid_points()
        if len(pts):
            d, _ = cKDTree(dense).query(pts)
            out["solid_on_objects_pct"] = round(100 * float(np.mean(d < 0.004)), 1)  # precision: solid evidence on real things
        up = allm.face_normals[face][:, 2] > -0.5  # bottoms can't be seen
        samples = dense[up & (dense[:, 2] > SURFACE_Z + 0.001)][::10]
        if len(samples) and len(pts):
            dist, _ = cKDTree(pts).query(samples)
            out["surface_covered_pct"] = round(100 * float(np.mean(dist < 0.004)), 1)  # coverage: real surface with evidence
        return out
