"""The real cell, behind the same interface lab/robot.py drives the Isaac cell through (observe, play), plus the trial flow.

    cell = RealCell(url, token)          # a run token (given directly, or from an approved `run.py --submit`)
    cell.check()                         # connection, latency, a picture, the trials: is everything ready?
    cell.start_trial()                   # start the trial the operator has marked ready
    obs = cell.observe(); cell.play(rows)

There's no truth here (no object poses, nothing to pour or take): the person's real hands do that. play() waits until the
arm has played everything it was sent, as the cell reports it (queued == 0 and not running), so the next observation is
taken standing still. When the trial ends (time up, or the operator stops it), play() and observe() raise TrialOver.
"""
from __future__ import annotations

import time

import numpy as np

from calibration.geometry import UF0_Z_MM, load
from calibration.kinematics import HOME, hand_matrix
from cell_client import OPEN, CellClient
from lab import gripper as G

DEFAULT_URL = "https://hackathon.sidclab.org"


class TrialOver(RuntimeError):
    pass


class RealCell:
    def __init__(self, url: str = DEFAULT_URL, token: str = "", ca: str | None = None):
        self.client = CellClient(url or DEFAULT_URL, token, ca)
        self.url = url or DEFAULT_URL
        self.instances, self.local, self.poses = [], {}, {}  # no truth on the real cell
        self.cams, self.instruction, self.image_size = load(), "", (1280, 720)
        self.q, self.grip, self.t = np.array(HOME, float), 0.0, 0.0
        self.wpr = np.array([-180.0, 0.0, 0.0])
        self.tip_xyz = None
        self.faults: list[str] = []
        self.trial: dict | None = None
        self.t0: float | None = None

    # ------------------------------------------------------------ the trial flow

    def trials(self) -> list[dict]:
        return self.client.trials()

    def check(self) -> dict:
        """Everything a person needs to know before pressing Run: can we reach it, how slow is it, does it see."""
        out = {"ok": False, "url": self.url}
        offset, rtt = self.client.clock()
        out["round_trip_ms"] = round(rtt * 1000)
        trials = self.trials()
        out["trials"] = [{"id": t["trial_id"], "status": t["status"], "instruction": t.get("instruction")} for t in trials]
        t0 = time.time()
        obs = self.client.observe(["table"], (640, 360), "jpeg", depth=["table"])
        out["observe_s"] = round(time.time() - t0, 2)
        out["joints"] = np.round(obs["joints"], 1).tolist()
        out["sees"] = bool(obs["images"].get("table") is not None and (obs.get("depth") or {}).get("table") is not None)
        ready = [t for t in trials if t["status"] == "ready"]
        out["ready_trial"] = ready[0]["trial_id"] if ready else None
        out["ok"] = out["sees"]
        return out

    def start_trial(self, trial_id: str | None = None) -> dict:
        trials = self.trials()
        t = next((x for x in trials if x["trial_id"] == trial_id), None) if trial_id else \
            next((x for x in trials if x["status"] == "ready"), None)
        if t is None:
            raise RuntimeError("no trial is ready: the operator marks it ready when the table is set up")
        self.client.start(t["trial_id"])
        self.trial, self.t0 = t, None
        for _ in range(50):  # until it's running (the cell homes the arm first)
            obs = self.client.observe(["table"], (64, 36), "jpeg")
            if obs.get("trial") and obs["trial"]["status"] == "running" and obs["status"] != "running":
                break
            time.sleep(0.2)
        return t

    def running(self) -> bool:
        return self.trial is not None and self.trial.get("status") != "over"

    # ------------------------------------------------------------ what the robot uses

    def observe(self, rng_noise: bool = True) -> dict:
        obs = self.client.observe(["table", "side", "wrist"], self.image_size, "jpeg", depth=["table", "wrist"])
        self._take(obs)
        obs["trial"] = {"status": "running", "instruction": self.instruction or (obs.get("trial") or {}).get("instruction", "")}
        return obs

    def _take(self, obs: dict) -> None:
        tr = obs.get("trial")
        if tr is None or tr.get("status") not in OPEN:
            if self.trial is not None:
                self.trial["status"] = "over"
            raise TrialOver("the trial is over")
        if self.t0 is None:
            self.t0 = float(obs["t"])
        self.t = float(obs["t"]) - self.t0
        self.q, self.grip = np.asarray(obs["joints"][:6], float), float(obs["joints"][6])
        tp = obs.get("tcp_pose")
        if tp is None:  # (the mock knows it only after a first move; the real cell always sends it)
            from calibration.kinematics import tcp_uf0
            tp = obs["tcp_pose"] = tcp_uf0(self.q)
        self.tip_xyz = np.array([tp[0], tp[1], tp[2] - UF0_Z_MM]) / 1000
        self.wpr = np.array(tp[3:6], float)

    def play(self, chunk, dt: float = 0.1, on_frame=None, kind: str = "tcp") -> None:
        rows = np.atleast_2d(np.asarray(chunk, float))
        for k in range(0, len(rows), 100):
            part = rows[k:k + 100]
            (self.client.act if kind == "joints" else self.client.act_tcp)(part, dt)
            time.sleep(len(part) * dt * 0.8)  # most of it plays out before we start asking
            while True:  # then until the queue is empty and the arm has stopped
                obs = self.client.observe(["table"], (64, 36), "jpeg")
                self._take(obs)
                if obs["queued"] == 0 and obs["status"] != "running":
                    break
                time.sleep(0.15)
            if on_frame is not None:
                on_frame(self)

    # ------------------------------------------------------------ for the viewer (the gripper only: no truth)

    def body_positions(self) -> dict:
        return {}

    def body_pose(self, name: str) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = hand_matrix(self.wpr)
        T[:3, 3] = self.tip()
        if name in ("finger_l", "finger_r"):
            T[:3, 3] = T[:3, 3] + T[:3, 0] * (1 - self.grip) * G.STROKE * (-1 if name == "finger_l" else 1)
        return T

    def tip(self) -> np.ndarray:
        if self.tip_xyz is None:
            from calibration.kinematics import tcp_of
            self.tip_xyz = tcp_of(self.q)[0]
        return self.tip_xyz.copy()

    def actual_wpr(self) -> np.ndarray:
        return self.wpr.copy()

    def actual_yaw(self) -> float:
        return float(np.radians(self.wpr[2]))

    def object_meshes(self) -> list:
        return []

    def settle(self, *a, **k) -> float:
        return 0.0

    def dump(self) -> None:
        pass
