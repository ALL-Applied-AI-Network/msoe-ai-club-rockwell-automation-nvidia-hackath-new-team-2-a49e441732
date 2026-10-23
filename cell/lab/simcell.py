"""A simulated cell: MuJoCo physics for the objects and a floating gripper, cameras rendered through the real calibration.

It speaks the same language as the real cell, so a policy can't tell the difference:
    obs = cell.observe()        # {"images", "depth", "joints", "tcp_pose", "trial", ...} like cell_client.observe()
    cell.play(chunk)            # [[x, y, z mm, w, p, r deg, gripper], ...] in UF0, one row per DT

The arm itself isn't simulated: the FANUC controller does that on the real cell. Only the gripper (two 76 x 12 x 6 mm
fingers, 12 mm apart when open) moves, straight to each fingertip target. The real arm trails its targets by about
0.8 s (lag_s); the gripper's open/close is not delayed, so a close sent the moment a target is sent happens early.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose, load
from lab import gripper as G
from lab.catalog import COLORS, DENSITY, KINDS, kind_mesh, rot_matrix
from lab.synth import HOME, IMAGE_SIZE, TCP_OFFSET, Renderer, wrist_ik

TCP_BOX = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]  # the cell's limits, mm in UF0
OPEN_GAP = 2 * G.STROKE          # flat faces 12 mm apart when open (the hooked tips: 16 mm)
SUBSTEPS = 50                     # physics steps per 0.1 s target (2 ms each)
GRIP_N = float(os.getenv("SIM_GRIP_N", 20))  # finger force limit, N: the real MHZL2-16D grips with ~14 N; at 180 N parts squirted out at 1.7 m/s


@dataclass
class Instance:
    id: int
    kind: str
    color: str
    pos: np.ndarray               # base frame, m
    quat: np.ndarray              # w, x, y, z

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind}"


def _quat(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return trimesh.transformations.quaternion_from_matrix(T)  # w, x, y, z


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def build_mjcf(instances: list[Instance], chute: tuple | None = None) -> str:
    """chute: (x, y, radius, height) of a tube to fill before a dump; it's removed by SimCell.lift_chute()."""
    assets, bodies = [], []
    for inst in instances:
        k = KINDS[inst.kind]
        rgba = np.array(COLORS[inst.color] + (255,)) / 255
        geoms = []
        for j, p in enumerate(k.parts):
            s = np.asarray(p.size, float) / 1000
            pos, quat = np.asarray(p.pos) / 1000, _quat(rot_matrix(p.rot))
            common = f'pos="{_fmt(pos)}" quat="{_fmt(quat)}" rgba="{_fmt(rgba)}" density="{DENSITY[k.material]}"'
            if p.kind == "box":
                geoms.append(f'<geom type="box" size="{_fmt(s / 2)}" {common}/>')
            elif p.kind == "cylinder":
                geoms.append(f'<geom type="cylinder" size="{_fmt([s[0], s[1] / 2])}" {common}/>')
            elif p.kind == "capsule":
                geoms.append(f'<geom type="capsule" size="{_fmt([s[0], s[1] / 2])}" {common}/>')
            elif p.kind == "sphere":
                geoms.append(f'<geom type="sphere" size="{_fmt(s[:1])}" {common}/>')
            elif p.kind == "convex":
                mname = f"m{inst.id}_{j}"
                assets.append(f'<mesh name="{mname}" vertex="{_fmt(np.asarray(p.vertices) / 1000)}"/>')
                geoms.append(f'<geom type="mesh" mesh="{mname}" {common}/>')
        bodies.append(f'<body name="obj{inst.id}" pos="{_fmt(inst.pos)}" quat="{_fmt(inst.quat)}"><freejoint/>{"".join(geoms)}</body>')
    home_tip = _home_tip()
    finger_assets, finger_geoms = [], {}
    for side, tag in ((-1, "l"), (1, "r")):
        gs = []
        for k, (v, fr) in enumerate(G.piece_vertices_hand(side)):
            finger_assets.append(f'<mesh name="f{tag}{k}" vertex="{_fmt(v)}"/>')
            gs.append(f'<geom type="mesh" mesh="f{tag}{k}" friction="{fr} 0.02 0.002" rgba=".68 .68 .7 1"/>')
        finger_geoms[tag] = "".join(gs)
    bw, bd, bh = G.BODY
    walls = ""
    if chute is not None:
        cx, cy, cr, ch = chute
        n = 16
        seg = 2 * np.pi * cr / n * 1.1
        for k, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
            walls += (f'<geom name="chute{k}" type="box" size="0.002 {seg / 2} {ch / 2}" '
                      f'pos="{cx + cr * np.cos(a)} {cy + cr * np.sin(a)} {SURFACE_Z + ch / 2}" euler="0 0 {np.degrees(a)}" rgba="1 1 1 0.1"/>')
    return f"""
<mujoco model="cell">
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <default><geom friction="0.8 0.01 0.0003" condim="6" solref="0.004 1"/></default>
  <asset>{"".join(assets)}{"".join(finger_assets)}</asset>
  <worldbody>
    <geom name="table" type="box" size="0.54 0.735 0.01" pos="0.17 0.01 {SURFACE_Z - 0.01}" rgba=".8 .78 .74 1"/>
    {walls}
    {"".join(bodies)}
    <body name="hand_target" mocap="true" pos="{_fmt(home_tip)}"/>
    <body name="hand" pos="{_fmt(home_tip)}" gravcomp="1">
      <freejoint name="hand_free"/>
      <inertial pos="0 0 0.09" mass="0.6" diaginertia="2e-3 2e-3 1e-3"/>
      <geom name="palm" type="box" size="{bw / 2} {bd / 2} {bh / 2}" pos="0 0 {G.LENGTH + bh / 2}" rgba=".27 .27 .29 1"/>
      <geom name="spacer" type="cylinder" size="{G.SPACER_R} {G.SPACER_LEN / 2}" pos="0 0 {G.LENGTH + bh + G.SPACER_LEN / 2}" rgba=".18 .18 .2 1"/>
      <body name="finger_l" pos="{-G.STROKE} 0 0" gravcomp="1">
        <joint name="jl" type="slide" axis="1 0 0" range="0 {G.STROKE}" damping="20"/>
        <inertial pos="-0.006 0 0.038" mass="0.032" diaginertia="2e-5 2e-5 1e-6"/>
        {finger_geoms["l"]}
      </body>
      <body name="finger_r" pos="{G.STROKE} 0 0" gravcomp="1">
        <joint name="jr" type="slide" axis="-1 0 0" range="0 {G.STROKE}" damping="20"/>
        <inertial pos="0.006 0 0.038" mass="0.032" diaginertia="2e-5 2e-5 1e-6"/>
        {finger_geoms["r"]}
      </body>
    </body>
  </worldbody>
  <equality><weld body1="hand_target" body2="hand" solref="0.004 1"/></equality>
  <actuator>
    <position joint="jl" kp="10000" ctrlrange="0 {G.STROKE}" forcerange="-{GRIP_N} {GRIP_N}"/>
    <position joint="jr" kp="10000" ctrlrange="0 {G.STROKE}" forcerange="-{GRIP_N} {GRIP_N}"/>
  </actuator>
</mujoco>"""


def _home_tip() -> np.ndarray:
    T = flange_pose(HOME)
    return T[:3, 3] + T[:3, 0] * TCP_OFFSET


class SimCell:
    def __init__(self, instances: list[Instance], instruction: str = "", seed: int = 0, chute: tuple | None = None,
                 lag_s: float = 0.8):
        self.instances = instances
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(instances, chute))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.cams = load()
        self.local = {i.id: kind_mesh(KINDS[i.kind]) for i in instances}
        self.instruction = instruction
        self.yaw = 0.0
        self.grip = 0.0
        self.q = np.array(HOME, float)
        self.t = 0.0
        self.image_size = IMAGE_SIZE  # what observe() renders; a policy may ask for another size
        mujoco.mj_forward(self.model, self.data)
        self.cmd_pos = self.data.mocap_pos[0].copy()   # where the controller has been told to go
        n_lag = max(1, int(round(lag_s / self.model.opt.timestep)))
        self.lagged = deque([(self.cmd_pos.copy(), 0.0)] * n_lag, maxlen=n_lag)  # what the arm actually follows

    # ------------------------------------------------------------ state

    def body_positions(self) -> dict[int, np.ndarray]:
        """Every object's position now (base frame, m), by id."""
        return {i.id: self.data.body(f"obj{i.id}").xpos.copy() for i in self.instances}

    def body_pose(self, name: str) -> np.ndarray:
        b = self.data.body(name)
        T = np.eye(4)
        T[:3, :3] = b.xmat.reshape(3, 3)
        T[:3, 3] = b.xpos
        return T

    def object_meshes(self) -> list[dict]:
        """Every object's mesh where it is now (base frame), with id, name and colour: the ground truth."""
        out = []
        for i in self.instances:
            m = self.local[i.id].copy()
            m.apply_transform(self.body_pose(f"obj{i.id}"))
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color, "rgb": COLORS[i.color], "mesh": m})
        return out

    def finger_meshes(self) -> list[dict]:
        """The gripper where it is now (base frame): both fingers, the SMC body and the spacer."""
        out = []
        for name, side in (("finger_l", -1), ("finger_r", 1)):
            m = G.finger_body_mesh(side)
            m.apply_transform(self.body_pose(name))
            out.append({"mesh": m, "rgb": (175, 175, 180)})
        for m, col in G.body_meshes():
            m = m.copy()
            m.apply_transform(self.body_pose("hand"))
            out.append({"mesh": m, "rgb": col})
        return out

    def tip(self) -> np.ndarray:
        return self.data.body("hand").xpos.copy()

    # ------------------------------------------------------------ physics

    def settle(self, max_s: float = 6.0, still: float = 2e-3) -> float:
        """Run physics until nothing moves (or max_s). Returns the simulated seconds it took."""
        t0 = self.data.time
        while self.data.time - t0 < max_s:
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            v = max((np.abs(self.data.body(f"obj{i.id}").cvel).max() for i in self.instances), default=0.0)
            if self.data.time - t0 > 0.3 and v < still:
                break
        return self.data.time - t0

    def lift_chute(self) -> None:
        """Take the tube away: the objects inside spill into a pile."""
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("chute"):
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

    def play(self, chunk, dt: float = 0.1, on_frame=None) -> None:
        """Move the gripper through fingertip targets (UF0 mm, deg), like the cell plays a chunk.
        on_frame(cell) is called after every target (every dt of simulated time): for animation."""
        a = np.atleast_2d(np.asarray(chunk, float))
        for k, (lo, hi) in enumerate(TCP_BOX):
            if a[:, k].min() < lo or a[:, k].max() > hi:
                raise ValueError(f"tcp {'xyz'[k]} outside [{lo}, {hi}] mm (UF0)")
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for row in a:
            start = self.cmd_pos.copy()
            goal = np.array([row[0] / 1000, row[1] / 1000, (row[2] - UF0_Z_MM) / 1000])
            yaw0, yaw1 = self.yaw, np.radians(row[5])
            self.grip = 1.0 if row[6] >= 0.5 else 0.0
            self.data.ctrl[:] = OPEN_GAP / 2 * self.grip  # the gripper valve isn't delayed
            for s in range(1, n + 1):
                f = s / n
                self.lagged.append((start + f * (goal - start), yaw0 + f * (yaw1 - yaw0)))
                pos, yaw = self.lagged[0]  # the arm is where it was told to be lag_s ago
                self.data.mocap_pos[0] = pos
                self.data.mocap_quat[0] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_step(self.model, self.data)
            self.cmd_pos, self.yaw = goal, yaw1
            self.t += dt
            if on_frame is not None:
                on_frame(self)

    def actual_yaw(self) -> float:
        return float(self.lagged[0][1])

    # ------------------------------------------------------------ cameras

    def observe(self, rng_noise: bool = True) -> dict:
        """What the real cell would send: RGB from table/side/wrist, table + wrist depth, joints, fingertip pose."""
        tip = self.tip()
        try:
            self.q = wrist_ik(tip, np.degrees(self.actual_yaw()), q0=self.q)
        except ValueError:
            pass  # keep the last joints: the wrist camera pose will be slightly off
        scene = [*self.object_meshes(), *self.finger_meshes()]
        r = Renderer(scene, self.cams)
        rng = self.rng if rng_noise else None
        return {
            "t": self.t, "status": "running", "queued": 0,
            "trial": {"status": "running", "instruction": self.instruction},
            "joints": np.append(self.q, self.grip),
            "tcp_pose": [tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, -180.0, 0.0, np.degrees(self.actual_yaw())],
            "images": {c: r.rgb(c, joints=self.q, size=self.image_size) for c in ("table", "side", "wrist")},
            "depth": {c: r.depth(c, joints=self.q, rng=rng) for c in ("table", "wrist")},
        }

    # ------------------------------------------------------------ truth

    def dump(self) -> None:
        """Fill the tube, lift it, let the pile come to rest."""
        self.settle(max_s=3.0)
        self.lift_chute()
        self.settle(max_s=5.0)

    def save_truth(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        objs = self.object_meshes()
        for o in objs:
            o["mesh"].export(folder / f"{o['id']:03d}.ply")
        (folder / "objects.json").write_text(json.dumps(
            [{"id": o["id"], "name": o["name"], "kind": o["kind"], "color": o["color"], "rgb": o["rgb"],
              "centroid": o["mesh"].centroid.tolist(), "mesh": f"{o['id']:03d}.ply"} for o in objs], indent=2))


# ---------------------------------------------------------------- layouts

def random_quat(rng) -> np.ndarray:
    u = rng.random(3)
    return np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]), np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                     np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]), np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])


CHUTE_RADIUS = 0.045


def dump_layout(order: list[tuple[str, str]], rng, center=(0.43, 0.0)) -> tuple[list[Instance], tuple]:
    """Objects poured into a tube at random tilts; SimCell.dump() lets them settle, lifts the tube, and they spill.
    Long objects (wider than the tube) are dropped across the top. Returns (instances, chute)."""
    out, z = [], SURFACE_Z + 0.01
    for i, (kind, color) in enumerate(rng.permutation(np.array(order, dtype=object)).tolist(), start=1):
        r = np.linalg.norm(kind_mesh(KINDS[kind]).extents) / 2
        d = max(0.0, CHUTE_RADIUS - min(r, CHUTE_RADIUS) - 0.004) * np.sqrt(rng.random())
        a = rng.uniform(0, 2 * np.pi)
        z += r
        out.append(Instance(i, kind, color, np.array([center[0] + d * np.cos(a), center[1] + d * np.sin(a), z]), random_quat(rng)))
        z += r + 0.002
    return out, (center[0], center[1], CHUTE_RADIUS, max(0.15, z - SURFACE_Z + 0.02))


def spread_layout(order: list[tuple[str, str]], rng, center=(0.43, -0.03), half=(0.07, 0.10), gap=0.012) -> list[Instance]:
    """Objects set down flat, apart from each other, at random spots and yaws."""
    out, placed = [], []
    for i, (kind, color) in enumerate(order, start=1):
        m = kind_mesh(KINDS[kind])
        r = np.linalg.norm(m.extents[:2]) / 2
        for _ in range(500):
            xy = np.array(center) + rng.uniform(-1, 1, 2) * half
            if all(np.linalg.norm(xy - q) > r + rq + gap for q, rq in placed):
                break
        placed.append((xy, r))
        yaw = rng.uniform(0, 2 * np.pi)
        out.append(Instance(i, kind, color, np.array([xy[0], xy[1], SURFACE_Z - m.bounds[0][2] + 0.002]),
                            np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
    return out


def tile_layout(order: list[tuple[str, str]], rng, cols: int, center=(0.44, -0.04), yaw_deg: float | None = None) -> list[Instance]:
    """Objects (same-size boxes, e.g. cubes) packed edge to edge in rows of `cols`, the whole tile turned by yaw_deg."""
    m = kind_mesh(KINDS[order[0][0]])
    pitch = float(m.extents[0]) + 0.0002
    yaw = np.radians(rng.uniform(0, 90) if yaw_deg is None else yaw_deg)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rows = int(np.ceil(len(order) / cols))
    out = []
    for i, (kind, color) in enumerate(rng.permutation(np.array(order, dtype=object)).tolist(), start=1):
        r, c = divmod(i - 1, cols)
        off = R @ np.array([(r - (rows - 1) / 2) * pitch, (c - (cols - 1) / 2) * pitch])
        z = SURFACE_Z - kind_mesh(KINDS[kind]).bounds[0][2] + 0.0005
        out.append(Instance(i, kind, color, np.array([center[0] + off[0], center[1] + off[1], z]),
                            np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
    return out
