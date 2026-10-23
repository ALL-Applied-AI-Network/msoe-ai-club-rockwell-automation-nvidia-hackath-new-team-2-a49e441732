"""Everything that happens at one step, shown the same way whether the step is live (sim cell) or replayed (any run).

    s = Session(cams, live_truth=lambda: cell.object_meshes())   # or truth=[...] from a saved run, or none (real cell)
    s.step(obs, note="look 2/5", actions=chunk)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import rerun as rr

from calibration.geometry import depth_to_base, flange_pose
from lab import perception, viz


class Session:
    def __init__(self, cams: dict, truth: list[dict] | None = None, live_truth: Callable[[], list[dict]] | None = None):
        self.cams, self.truth, self.live_truth = cams, truth, live_truth
        self.k = 0
        self.fused: list[np.ndarray] = []  # assumes nothing moved yet (scanning only); nudges will need per-object fusion
        self.segs: list = []
        self.last_score: dict = {}

    def step(self, obs: dict, note: str = "", actions=None) -> dict:
        viz.set_step(self.k)
        viz.images(obs)
        if obs.get("joints") is not None:
            viz.arm("mind", obs["joints"])
            viz.arm("truth", obs["joints"])
        for cam, d in (obs.get("depth") or {}).items():
            c = self.cams[f"{cam}_depth"]
            if c["frame"] == "flange" and obs.get("joints") is None:
                continue
            T = flange_pose(obs["joints"]) @ c["T"] if c["frame"] == "flange" else c["T"]
            viz.camera_frustum(f"mind/cameras/{cam}", c, T)
            pts = depth_to_base(c, d, stride=2, joints=obs["joints"] if c["frame"] == "flange" else None)
            pts = perception.drop_self(pts, obs.get("tcp_pose"))
            self.fused.append(pts)
            rr.log(f"mind/latest_view/{cam}", rr.Points3D(viz.thin(pts, 0.003), colors=viz.height_colors(viz.thin(pts, 0.003)),
                                                          radii=0.0006))
        if self.fused:
            allpts = np.concatenate(self.fused)
            self.segs, hm = perception.segment(allpts)
            shown = viz.thin(allpts, 0.002)
            rr.log("mind/fused", rr.Points3D(shown, colors=viz.height_colors(shown), radii=0.0005))
            viz.heightmap(hm)
            viz.segments(self.segs, self.cams, obs)
        if actions is not None:
            viz.path("mind", actions, name="sent")
        truth = self.live_truth() if self.live_truth else self.truth
        if truth:
            viz.truth_objects(truth)
            self.last_score = perception.score(self.segs, truth)
        viz.roster(self._roster())
        viz.narrate(self._line(note))
        self.k += 1
        return self.last_score

    def _line(self, note: str) -> str:
        s = self.last_score
        seen = f"I see {len(self.segs)} separate thing{'s' if len(self.segs) != 1 else ''}"
        if s:
            seen += (f" (truth: {s['objects']} objects; {s['separated']} cleanly separated, {s['merged']} clumps, "
                     f"{s['objects'] - s['found']} not found" + (f", position error {s['error_mm']:.1f} mm" if s.get('error_mm') else "") + ")")
        return f"step {self.k}: {note}. {seen}" if note else f"step {self.k}: {seen}"

    def _roster(self) -> str:
        rows = ["| id | footprint mm² | height mm | where (x, y mm) |", "|---|---|---|---|"]
        for sg in self.segs:
            rows.append(f"| #{sg.id} | {sg.cells} | {sg.height_mm:.0f} | {sg.centroid[0] * 1000:.0f}, {sg.centroid[1] * 1000:.0f} |")
        head = "**What the robot has found** (blobs from depth; not yet labelled)\n\n"
        if self.last_score:
            s = self.last_score
            head += f"Truth check: {s['found']}/{s['objects']} objects found, {s['separated']} separated, {s['merged']} clumps\n\n"
        return head + "\n".join(rows)
