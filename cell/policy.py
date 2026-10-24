"""Baseline v0: blind scripted pick of one screw at a known position, dropped into the yellow bin.

No vision. It reads the screw and bin positions from submission/placement.baseline.yaml (the operator places them there) and
plays a fixed approach -> descend -> close -> lift -> carry -> open sequence in tcp mode. This is the baseline that a
vision policy has to beat once the screw is moved.

act(obs) returns the arm's next target, or a list of targets to play in order. When they have been played, act is
called again with a new observation. act has no time limit: the arm holds still while it runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from calibration.geometry import UF0_Z_MM, flange_pose
from lab.recorder import Recorder

HERE = Path(__file__).resolve().parent

TABLE_Z = -335.0      # work surface, mm in UF0
GRASP_Z = -331.0      # fingertips 4 mm above the table: the screw shank (6 mm) sits ~0-6 mm up. Tune in the twin.
SAFE_Z = -240.0       # travel height: clears the 55 mm bin with margin
DROP_Z = -250.0       # fingertips ~30 mm above the bin rim
GRASP_ALONG = 15.0    # grasp this far from the head toward the tip (M6x30: middle of the shank)
JAW_OFFSET = 90.0     # R that puts the jaws across the screw = screw yaw + this. UNVERIFIED: check in the twin.
SPEED = 100.0         # mm/s of fingertip travel
REACH = 540.0         # stay inside the ~550 mm reach limit (controller faults beyond it)


def _load_scene(path: Path = HERE / "submission" / "placement.baseline.yaml") -> tuple[np.ndarray, float, np.ndarray]:
    """(screw head xy, screw yaw deg, bin xy) from the first rollout in placement.yaml."""
    rollouts = yaml.safe_load(path.read_text())["rollouts"]
    objs = rollouts[0]["objects"]
    screw = objs["screw"][0] if isinstance(objs["screw"][0], list) else objs["screw"]
    return np.array(screw[:2], float), float(screw[2]), np.array(objs["bin"][:2], float)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Policy:
    CAMERAS = ["table", "side", "wrist"]  # recorded either way; v0 ignores them
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = ["table", "wrist"]            # live depth; the cell's own recordings leave it out, so we record it
    DT = 0.1
    RECORD = True                         # save every observation to runs/ (replay: uv run python -m lab.replay)

    def __init__(self, checkpoint: str | None = None):
        head, yaw, self.bin_xy = _load_scene()
        tip_dir = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])  # yaw 0 = +x, 90 = +y
        self.grasp_xy = head + GRASP_ALONG * tip_dir
        self.grasp_r = _wrap(yaw + JAW_OFFSET)
        for name, xy in (("grasp", self.grasp_xy), ("bin", self.bin_xy)):
            if np.hypot(*xy) > REACH:
                raise ValueError(f"{name} point {xy} is {np.hypot(*xy):.0f} mm from the base, beyond {REACH} mm")
        self.rec = Recorder("baseline_v0", meta={"policy": "baseline_v0", "grasp_xy": self.grasp_xy,
                                                  "bin_xy": self.bin_xy}) if self.RECORD else None

    def reset(self) -> None:
        """Called at the start of every rollout."""
        self.plan: list[list[float]] | None = None
        if self.rec:
            self.rec.new_rollout()

    def _build_plan(self, start: np.ndarray) -> list[list[float]]:
        gx, gy = self.grasp_xy
        bx, by = self.bin_xy
        r = self.grasp_r
        waypoints = [  # (x, y, z, R, gripper, dwell steps after arriving)
            (gx, gy, SAFE_Z, r, 0, 5),
            (gx, gy, GRASP_Z, r, 0, 12),  # the real arm trails targets by ~0.8 s: arrive first
            (gx, gy, GRASP_Z, r, 1, 10),   # close and wait for the pneumatics
            (gx, gy, SAFE_Z, r, 1, 3),
            (bx, by, SAFE_Z, r, 1, 3),
            (bx, by, DROP_Z, r, 1, 12),
            (bx, by, DROP_Z, r, 0, 10),    # open and let it fall
            (bx, by, SAFE_Z, r, 0, 0),
        ]
        plan: list[list[float]] = []
        cur = np.array([*start[:3], start[5]], float)
        for x, y, z, rr, g, dwell in waypoints:
            goal = np.array([x, y, z, rr], float)
            dr = _wrap(goal[3] - cur[3])
            n = max(1, int(np.ceil(np.linalg.norm(goal[:3] - cur[:3]) / (SPEED * self.DT))), int(abs(dr) / 9))
            for s in np.linspace(0, 1, n + 1)[1:]:
                p = cur[:3] + s * (goal[:3] - cur[:3])
                plan.append([*p, -180.0, 0.0, _wrap(cur[3] + s * dr), g])
            plan += [plan[-1]] * dwell
            cur = goal
        return plan

    @staticmethod
    def _tcp(obs: dict) -> np.ndarray:
        """The fingertip [x, y, z, w, p, r] in UF0; from forward kinematics when the cell has not reported it yet."""
        if obs.get("tcp_pose") is not None:
            return np.asarray(obs["tcp_pose"], float)
        T = flange_pose(obs["joints"])
        tip = T[:3, 3] + T[:3, 0] * 0.293  # UTOOL1 is 293 mm along the tool axis from the J6 origin
        return np.array([tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, -180.0, 0.0, 0.0])

    def act(self, obs: dict):
        t0 = time.perf_counter()
        chunk, note = self._act(obs)
        if self.rec:
            self.rec.log(obs, chunk, act_s=time.perf_counter() - t0, note=note)
        return chunk

    def _act(self, obs: dict) -> tuple[list, str]:
        if self.plan is None:
            self.plan = self._build_plan(self._tcp(obs))
            note = f"planned {len(self.plan)} targets: screw grasp at {np.round(self.grasp_xy).tolist()}, bin at {self.bin_xy.tolist()}"
        else:
            note = "continuing plan"
        if not self.plan:  # done: hold where we are for 1 s (the policy cannot end the rollout itself)
            return [[*self._tcp(obs), 0]] * 10, "done, holding"
        chunk, self.plan = self.plan[:100], self.plan[100:]
        return chunk, note
