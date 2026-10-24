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
import trimesh

from calibration.geometry import SURFACE_Z, UF0_Z_MM
from calibration.kinematics import hand_matrix
from cell_client import CellClient
from lab import gripper as G
from lab.catalog import COLORS, DENSITY, KINDS, kind_mesh
from lab.simcell import Instance, SimCell, _quat, random_quat
from lab.synth import HOME

PORT = int(os.getenv("ISAAC_PORT", 8765))
POUR_AT = (0.42, -0.02)  # where a poured cup lands (base frame, m): the usual pile spot
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
        self.q, self.grip, self.t = np.array(HOME, float), 0.0, 0.0
        self.wpr = np.array([-180.0, 0.0, 0.0])
        self.tip_xyz = None
        self.faults: list[str] = []  # collisions the real cell would have stopped the trial for
        objs = []
        for i in self.instances:
            T = sim.body_pose(f"obj{i.id}")
            objs.append(self._spec(i, T, float(sim.model.body(f"obj{i.id}").mass[0])))
        r = self._post("sim/reset", {"objects": objs, "instruction": self.instruction}, timeout=120)
        self.poses = {int(k): np.array(v) for k, v in r["truth"].items()}
        self.trial = r["trial"]

    def _spec(self, i: Instance, T: np.ndarray, mass: float | None = None) -> dict:
        """What the server needs to build an object: its mesh (or a box's walls), colour, mass and pose."""
        k = KINDS[i.kind]
        m = self.local[i.id]
        if mass is None:
            vol = m.volume if m.is_volume else m.convex_hull.volume
            mass = vol * DENSITY[k.material]
        o = {"id": i.id, "name": i.name, "rgb": list(COLORS[i.color]), "mass": float(mass),
             "pos": T[:3, 3].tolist(), "quat": _quat(T[:3, :3]).tolist()}
        if "container" in k.tags and all(p.kind == "box" and not any(p.rot) for p in k.parts):
            o["boxes"] = [{"center": [c / 1000 for c in p.pos], "size": [s / 1000 for s in p.size]} for p in k.parts]
        else:
            o["vertices"], o["faces"] = m.vertices.tolist(), m.faces.tolist()
        return o

    # ------------------------------------------------------------ Joel's hands (the robot isn't told: it has to notice)

    def pour(self, order: list[tuple[str, str]], at=POUR_AT, seed: int | None = None) -> list[Instance]:
        """Pour a cup of these (kind, colour) onto the table at `at` (base frame, m): they drop from a little stack
        above it and land in a pile."""
        rng = np.random.default_rng(seed)
        new, specs = [], []
        nid = self._next_id(len(order))
        for k, (kind, color) in enumerate(order):
            if kind not in KINDS:
                raise ValueError(f"no such kind {kind!r}")
            i = Instance(nid + k, kind, color, np.zeros(3), np.array([1.0, 0, 0, 0]))
            self.local[i.id] = kind_mesh(KINDS[kind])
            T = np.eye(4)
            T[:3, :3] = trimesh.transformations.quaternion_matrix(random_quat(rng))[:3, :3]
            T[:3, 3] = [at[0] + rng.normal(0, 0.012), at[1] + rng.normal(0, 0.012), SURFACE_Z + 0.04 + 0.025 * k]
            new.append(i)
            specs.append(self._spec(i, T))
        r = self._post("sim/add", {"objects": specs}, timeout=120)
        self.instances = self.instances + new
        self._set(r["truth"])
        return new

    def take(self, ids) -> None:
        """Take things off the table (someone picks them up)."""
        ids = [int(i) for i in ids]
        r = self._post("sim/remove", {"ids": ids})
        self.instances = [i for i in self.instances if i.id not in ids]
        self._set(r["truth"])

    def add_box(self, color: str = "white", at=(0.36, -0.24), kind: str = "box_60") -> Instance:
        i = Instance(self._next_id(1), kind, color, np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.local[i.id] = kind_mesh(KINDS[kind])
        T = np.eye(4)
        T[:3, 3] = [at[0], at[1], SURFACE_Z - self.local[i.id].bounds[0][2] + 0.0005]
        r = self._post("sim/add", {"objects": [self._spec(i, T)]}, timeout=120)
        self.instances = self.instances + [i]
        self._set(r["truth"])
        return i

    def tip_box(self, box_id: int, at=POUR_AT, seed: int | None = None) -> list[int]:
        """Tip a box out onto the table at `at`, by hand: everything inside it lands in a pile there. Returns their ids."""
        B = self.body_pose(f"obj{box_id}")
        half = self.local[box_id].extents[:2] / 2
        inside = [i.id for i in self.instances if i.id != box_id
                  and np.all(np.abs((np.linalg.inv(B) @ np.append(self.poses[i.id][:3], 1))[:2]) < half)]
        if not inside:
            return []
        rng = np.random.default_rng(seed)
        poses = {}
        for k, i in enumerate(inside):
            q = random_quat(rng)
            poses[str(i)] = [at[0] + rng.normal(0, 0.012), at[1] + rng.normal(0, 0.012), SURFACE_Z + 0.04 + 0.025 * k, *q.tolist()]
        self._set(self._post("sim/move", {"poses": poses}, timeout=120)["truth"])
        return inside

    def _next_id(self, n: int) -> int:
        """Fresh ids, never reused: a thing taken off the table still exists in Isaac (in Joel's pocket)."""
        first = max([i.id for i in self.instances] + [getattr(self, "last_id", 0)]) + 1
        self.last_id = first + n - 1
        return first

    def _set(self, truth: dict) -> None:
        self.poses = {int(k): np.array(v) for k, v in truth.items()}

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
        obs = self.client.observe(["table", "side", "wrist"], self.image_size, "jpeg", depth=["table", "wrist"])  # jpeg: ~10x smaller, as the real cell allows
        self.q, self.grip, self.t = np.asarray(obs["joints"][:6], float), float(obs["joints"][6]), float(obs["t"])
        tp = obs["tcp_pose"]
        self.tip_xyz = np.array([tp[0], tp[1], tp[2] - UF0_Z_MM]) / 1000
        self.wpr = np.array(tp[3:6], float)
        obs["trial"] = {"status": "running", "instruction": self.instruction}
        return obs

    def play(self, chunk, dt: float = 0.1, on_frame=None, kind: str = "tcp") -> None:
        """Send the rows (tcp, or joints: [J1..J6 deg, grip]), wait until Isaac has played them, then replay its
        per-row truth for the viewer and checks."""
        rows = np.atleast_2d(np.asarray(chunk, float))
        if len(rows) > 100:  # the cell takes at most 100 rows a chunk
            for k in range(0, len(rows), 100):
                self.play(rows[k:k + 100], dt, on_frame, kind)
            return
        try:
            (self.client.act if kind == "joints" else self.client.act_tcp)(rows, dt)
        except RuntimeError as e:
            raise ValueError(str(e)) from e  # like SimCell: a target outside the cell's limits
        for f in self._get("sim/wait")["frames"]:
            if f.get("fault"):
                self.faults.append(f["fault"])
                print(f"[isaac] FAULT: {f['fault']}", flush=True)
            self.poses = {int(k): np.array(v) for k, v in f["poses"].items()}
            self.t, self.grip, self.wpr = f["t"], f["grip"], np.array(f["wpr"], float)
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
        T[:3, :3] = hand_matrix(self.wpr)  # the hand: at the fingertip, turned like the tool (as in SimCell)
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

    def actual_wpr(self) -> np.ndarray:
        return self.wpr.copy()

    def actual_yaw(self) -> float:
        return float(np.radians(self.wpr[2]))

    def object_meshes(self) -> list[dict]:
        out = []
        for i in self.instances:
            m = self.local[i.id].copy()
            m.apply_transform(self.body_pose(f"obj{i.id}"))
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color, "rgb": COLORS[i.color], "mesh": m})
        return out

    def save_truth(self, folder: Path) -> None:
        SimCell.save_truth(self, folder)
