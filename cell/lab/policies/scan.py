"""Look before touching: an overview from home, then the wrist camera held still over a few spots around the pile.

Works on the real cell, the mock, and the sim cell: it only reads obs["tcp_pose"] and returns fingertip targets.
Each act() call moves to one spot and holds still for HOLD steps, so the next observation is a sharp, settled frame.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose, load

SPEED = 150.0          # mm/s
HOLD = 12              # 1.2 s still before each look (the real arm trails its targets by ~0.8 s)
REACH = 540.0
TCP_BOX = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]


def camera_offset_mm(joints) -> np.ndarray:
    """Where the wrist depth camera sits relative to the fingertip, in x/y (mm), for this arm pose."""
    cam = load()["wrist_depth"]
    T = flange_pose(joints)
    tip = T[:3, 3] + T[:3, 0] * 0.293
    return ((T @ cam["T"])[:2, 3] - tip[:2]) * 1000


class ScanPolicy:
    CAMERAS = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = ["table", "wrist"]
    DT = 0.1

    def __init__(self, checkpoint: str | None = None, center_mm=(430.0, 0.0), spread_mm: float = 45.0,
                 tip_height_mm: float = 70.0):
        self.center = np.asarray(center_mm, float)
        self.spread = spread_mm
        self.z = SURFACE_Z * 1000 + UF0_Z_MM + tip_height_mm  # UF0
        self.reset()

    def reset(self) -> None:
        self.stops: list[np.ndarray] | None = None
        self.k = 0
        self.done = False

    def _plan(self, obs) -> list[np.ndarray]:
        off = camera_offset_mm(obs["joints"]) if obs.get("joints") is not None else np.array([102.0, 30.0])
        looks = [self.center + np.array(d) * self.spread for d in [(0, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]]
        stops = []
        for p in looks:
            tip = p - off  # put the camera, not the fingertip, over the spot
            tip[0] = np.clip(tip[0], TCP_BOX[0][0] + 5, TCP_BOX[0][1] - 5)
            if np.hypot(*tip) > REACH:
                tip *= REACH / np.hypot(*tip)
            stops.append(np.array([tip[0], tip[1], self.z]))
        return stops

    def act(self, obs: dict):
        cur = np.asarray(obs["tcp_pose"][:3], float) if obs.get("tcp_pose") is not None else None
        if self.stops is None:
            self.stops = self._plan(obs)
            self.note = f"overview from where I am; then {len(self.stops)} close looks"
            return [list(cur) + [-180.0, 0.0, 0.0, 0.0]] * HOLD
        if self.k >= len(self.stops):
            self.done = True
            self.note = "done looking"
            return [list(cur) + [-180.0, 0.0, 0.0, 0.0]] * 10
        goal = self.stops[self.k]
        n = max(1, int(np.ceil(np.linalg.norm(goal - cur) / (SPEED * self.DT))))
        path = [cur + (goal - cur) * s for s in np.linspace(0, 1, n + 1)[1:]]
        self.k += 1
        self.note = f"look {self.k}/{len(self.stops)}: camera over ({goal[0]:.0f}, {goal[1]:.0f}) mm, fingertip {goal[2] - UF0_Z_MM - SURFACE_Z * 1000:.0f} mm up"
        return [list(p) + [-180.0, 0.0, 0.0, 0.0] for p in path] + [list(goal) + [-180.0, 0.0, 0.0, 0.0]] * HOLD
