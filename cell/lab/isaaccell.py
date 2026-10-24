"""The Isaac cell: a SimCell look-alike whose physics and cameras run in Isaac Sim on the 4090 box (isaac/server.py).

    uv run python -m lab.trial run sort_mixed --policy sort --cell isaac

MuJoCo still lays out the trial (the dump and the settle: a seed gives the same pile in both sims). The poses and meshes
go to Isaac, and from then on Isaac moves everything and renders everything the policy sees, through the real cell's
HTTP API (cell_client.py). Isaac's window on the box's monitor is the truth; Rerun on the Mac is the robot's mind.

The server listens on the box's localhost; this opens an SSH tunnel to it if one isn't up (ISAAC_SSH, default
joel@shepherd-traffic-base-server, over Tailscale).
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from calibration.geometry import UF0_Z_MM
from cell_client import CellClient
from lab import gripper as G
from lab.catalog import COLORS, KINDS
from lab.simcell import SimCell, _quat
from lab.synth import HOME

PORT = int(os.getenv("ISAAC_PORT", 8765))
SSH = os.getenv("ISAAC_SSH", "joel@shepherd-traffic-base-server")


def tunnel() -> None:
    """Make localhost:PORT reach the server's localhost:PORT (an ssh -L in the background, if it isn't up already)."""
    def up() -> bool:
        with socket.socket() as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", PORT)) == 0
    if up():
        return
    subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                      "-L", f"{PORT}:127.0.0.1:{PORT}", SSH], stdin=subprocess.DEVNULL)
    for _ in range(50):
        if up():
            return
        time.sleep(0.2)
    raise SystemExit(f"no tunnel to the Isaac server: is it running on {SSH}? (isaac/run_server.sh there)")


class IsaacCell:
    def __init__(self, sim: SimCell):
        tunnel()
        self.url = f"http://127.0.0.1:{PORT}/api/v1"
        self.client = CellClient(f"http://127.0.0.1:{PORT}", token="sim")
        self.instances, self.local, self.cams = sim.instances, sim.local, sim.cams
        self.instruction, self.image_size = sim.instruction, sim.image_size
        self.q, self.grip, self.t, self.yaw = np.array(HOME, float), 0.0, 0.0, 0.0
        self.tip_xyz = None
        objs = []
        for i in self.instances:
            k = KINDS[i.kind]
            T = sim.body_pose(f"obj{i.id}")
            o = {"id": i.id, "name": i.name, "rgb": list(COLORS[i.color]), "mass": float(sim.model.body(f"obj{i.id}").mass[0]),
                 "pos": T[:3, 3].tolist(), "quat": _quat(T[:3, :3]).tolist()}
            if "container" in k.tags and all(p.kind == "box" and not any(p.rot) for p in k.parts):
                o["boxes"] = [{"center": [c / 1000 for c in p.pos], "size": [s / 1000 for s in p.size]} for p in k.parts]
            else:
                m = self.local[i.id]
                o["vertices"], o["faces"] = m.vertices.tolist(), m.faces.tolist()
            objs.append(o)
        r = self._post("sim/reset", {"objects": objs, "instruction": self.instruction}, timeout=120)
        self.poses = {int(k): np.array(v) for k, v in r["truth"].items()}
        self.trial = r["trial"]

    def _post(self, path: str, body: dict, timeout: float = 30) -> dict:
        r = requests.post(f"{self.url}/{path}", json=body, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"isaac {path}: {r.json().get('error', r.status_code)}")
        return r.json()

    def _get(self, path: str, timeout: float = 600) -> dict:
        r = requests.get(f"{self.url}/{path}", timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"isaac {path}: {r.json().get('error', r.status_code)}")
        return r.json()

    # ------------------------------------------------------------ what the policy uses

    def observe(self, rng_noise: bool = True) -> dict:
        obs = self.client.observe(["table", "side", "wrist"], self.image_size, "png", depth=["table", "wrist"])
        self.q, self.grip, self.t = np.asarray(obs["joints"][:6], float), float(obs["joints"][6]), float(obs["t"])
        tp = obs["tcp_pose"]
        self.tip_xyz = np.array([tp[0], tp[1], tp[2] - UF0_Z_MM]) / 1000
        self.yaw = np.radians(tp[5])
        obs["trial"] = {"status": "running", "instruction": self.instruction}
        return obs

    def play(self, chunk, dt: float = 0.1, on_frame=None) -> None:
        """Send the rows, wait until Isaac has played them, then replay its per-row truth for the viewer and checks."""
        try:
            self.client.act_tcp(np.atleast_2d(np.asarray(chunk, float)), dt)
        except RuntimeError as e:
            raise ValueError(str(e)) from e  # like SimCell: a target outside the cell's limits
        for f in self._get("sim/wait")["frames"]:
            self.poses = {int(k): np.array(v) for k, v in f["poses"].items()}
            self.t, self.grip, self.yaw = f["t"], f["grip"], np.radians(f["yaw"])
            self.tip_xyz, self.q = np.array(f["tip"]), np.array(f["joints"])
            if on_frame is not None:
                on_frame(self)

    def settle(self, max_s: float = 6.0, still: float = 2e-3) -> float:
        return 0.0  # Isaac settles the layout on reset

    def dump(self) -> None:
        pass

    # ------------------------------------------------------------ truth, in SimCell's terms

    def body_positions(self) -> dict[int, np.ndarray]:
        return {i: p[:3].copy() for i, p in self.poses.items()}

    def body_pose(self, name: str) -> np.ndarray:
        T = np.eye(4)
        if name.startswith("obj"):
            p = self.poses[int(name[3:])]
            w, x, y, z = p[3:]
            T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
            T[:3, 3] = p[:3]
            return T
        c, s = np.cos(self.yaw), np.sin(self.yaw)  # the hand: at the fingertip, turned to the yaw (as in SimCell)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = self.tip()
        if name in ("finger_l", "finger_r"):
            off = (1 - self.grip) * G.STROKE * (-1 if name == "finger_l" else 1)
            T[:3, 3] = T[:3, 3] + T[:3, 0] * off
        return T

    def tip(self) -> np.ndarray:
        if self.tip_xyz is None:
            from lab.synth import TCP_OFFSET
            from calibration.geometry import flange_pose
            F = flange_pose(self.q)
            self.tip_xyz = F[:3, 3] + F[:3, 0] * TCP_OFFSET
        return self.tip_xyz.copy()

    def actual_yaw(self) -> float:
        return float(self.yaw)

    def object_meshes(self) -> list[dict]:
        out = []
        for i in self.instances:
            m = self.local[i.id].copy()
            m.apply_transform(self.body_pose(f"obj{i.id}"))
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color, "rgb": COLORS[i.color], "mesh": m})
        return out

    def save_truth(self, folder: Path) -> None:
        SimCell.save_truth(self, folder)
