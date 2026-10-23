"""Synthetic runs: a known scene rendered through the cell's real camera calibration, saved in the recorder's format.

Because the scene is known, every perception step can be scored against the truth (saved in truth/).

    uv run python -m lab.synth            # writes runs/<time>_synthetic/ and replays it into the viewer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import least_squares

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose, load
from lab.recorder import Recorder

JOINT_LIMITS = [(-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360)]
HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286]
TCP_OFFSET = 0.293  # fingertip, metres along the flange's +x (tool) axis
NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}
IMAGE_SIZE = (640, 360)
TABLE_RGB = (205, 200, 190)


# ---------------------------------------------------------------- scene

def _on_table(m: trimesh.Trimesh, x: float, y: float, yaw_deg: float = 0.0) -> trimesh.Trimesh:
    """Rotate about z, then drop the mesh so its lowest point rests on the table at (x, y)."""
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1]))
    c = m.bounds.mean(0)
    m.apply_translation([x - c[0], y - c[1], SURFACE_Z - m.bounds[0][2]])
    return m


def build_scene() -> list[dict]:
    """[{"id", "name", "rgb", "mesh"}]: a flush 3x3 tile of 16 mm cubes, a screw, a cup with a brim, a jack, a tongue depressor."""
    mm = 1e-3
    objs: list[dict] = []
    tile_colors = [(200, 40, 40), (40, 90, 200), (40, 160, 70)]
    for i in range(3):
        for j in range(3):
            objs.append({"name": "16 mm cube", "rgb": tile_colors[(i + j) % 3],
                         "mesh": _on_table(trimesh.creation.box([16 * mm] * 3), 0.45 + i * 16 * mm, -0.12 + j * 16 * mm)})
    shank = trimesh.creation.cylinder(3 * mm, 30 * mm)
    head = trimesh.creation.cylinder(6 * mm, 4 * mm)
    head.apply_translation([0, 0, 17 * mm])
    screw = trimesh.util.concatenate([shank, head])
    screw.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    objs.append({"name": "M6 screw", "rgb": (150, 150, 155), "mesh": _on_table(screw, 0.36, -0.02, yaw_deg=30)})
    wall = trimesh.creation.annulus(18 * mm, 20 * mm, 40 * mm)
    brim = trimesh.creation.annulus(18 * mm, 26 * mm, 2 * mm)
    brim.apply_translation([0, 0, 20 * mm])
    bottom = trimesh.creation.cylinder(20 * mm, 2 * mm)
    bottom.apply_translation([0, 0, -19 * mm])
    objs.append({"name": "cup with brim", "rgb": (240, 240, 235),
                 "mesh": _on_table(trimesh.util.concatenate([wall, brim, bottom]), 0.38, 0.14)})
    legs = []
    for axis in np.eye(3):
        leg = trimesh.creation.cylinder(2 * mm, 26 * mm)
        leg.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis))
        legs.append(leg)
    jack = trimesh.util.concatenate(legs)
    jack.apply_transform(trimesh.transformations.rotation_matrix(np.radians(35), [1, 1, 0]))
    objs.append({"name": "jack", "rgb": (230, 180, 30), "mesh": _on_table(jack, 0.46, 0.06, yaw_deg=20)})
    objs.append({"name": "tongue depressor", "rgb": (215, 185, 140),
                 "mesh": _on_table(trimesh.creation.box([150 * mm, 18 * mm, 2 * mm]), 0.30, 0.25, yaw_deg=-20)})
    for i, o in enumerate(objs, start=1):
        o["id"] = i
    return objs


def table_mesh() -> trimesh.Trimesh:
    t = trimesh.creation.box([1.08, 1.47, 0.02])
    t.apply_translation([0.17, 0.01, SURFACE_Z - 0.01])
    return t


# ---------------------------------------------------------------- arm

def wrist_ik(tip_xyz, yaw_deg: float = 0.0, q0=HOME) -> np.ndarray:
    """Joints (6, deg) that put the fingertip at tip_xyz (base frame, m) with the fingers pointing straight down."""
    yaw = np.radians(yaw_deg)

    def err(q):
        T = flange_pose(q)
        tip = T[:3, 3] + T[:3, 0] * TCP_OFFSET
        return np.concatenate([(tip - tip_xyz) * 10, T[:3, 0] - [0, 0, -1],
                               0.3 * (T[:3, 1] - [np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2), 0])])

    lo, hi = np.array(JOINT_LIMITS, float).T
    sol = least_squares(err, np.clip(q0, lo, hi), bounds=(lo, hi))
    T = flange_pose(sol.x)
    miss = np.linalg.norm(T[:3, 3] + T[:3, 0] * TCP_OFFSET - tip_xyz)
    if miss > 2e-3:
        raise ValueError(f"no IK solution within 2 mm of {tip_xyz} (missed by {miss * 1000:.1f} mm)")
    return sol.x


def tcp_pose_uf0(joints) -> list[float]:
    T = flange_pose(joints)
    tip = T[:3, 3] + T[:3, 0] * TCP_OFFSET
    return [tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, -180.0, 0.0, 0.0]


# ---------------------------------------------------------------- rendering

class Renderer:
    """Ray-casts the scene (table + objects) for depth and colour cameras."""

    def __init__(self, objs: list[dict], cams: dict):
        self.cams, self.scene = cams, o3d.t.geometry.RaycastingScene()
        self.colors = [np.array(TABLE_RGB, float)]
        for m, rgb in [(table_mesh(), TABLE_RGB), *[(o["mesh"], o["rgb"]) for o in objs]]:
            self.scene.add_triangles(o3d.core.Tensor(m.vertices, o3d.core.float32), o3d.core.Tensor(m.faces, o3d.core.uint32))
        self.colors += [np.array(o["rgb"], float) for o in objs]

    def _cast(self, origin, dirs):
        rays = np.concatenate([np.broadcast_to(origin, dirs.shape), dirs], -1).astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(rays))

    def depth(self, name: str, joints=None, rng=None) -> np.ndarray:
        """uint16 mm along the optical axis, like the real stream: quantised, noisier with distance, 0 = no return."""
        cam = self.cams[f"{name}_depth"]
        T = flange_pose(joints) @ cam["T"] if cam["frame"] == "flange" else cam["T"]
        w, h = cam["size"]
        v, u = np.mgrid[0:h, 0:w]
        d = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones((h, w))], -1) @ T[:3, :3].T
        t = self._cast(T[:3, 3], d)["t_hit"].numpy()  # unnormalised rays: t is the depth along the optical axis
        t[~np.isfinite(t)] = 0
        if rng is not None:  # roughly an Orbbec's error: ~1.5 mm at 0.8 m, growing with distance squared
            t = np.where(t > 0, t + rng.normal(0, 1.5e-3 * (t / 0.8) ** 2), 0)
        return np.round(t * 1000).astype(np.uint16)

    def rgb(self, name: str, joints=None, size=IMAGE_SIZE) -> np.ndarray:
        """A shaded colour image as served at `size`, with the colour camera's pinhole + radial model."""
        cam = self.cams[name]
        T = flange_pose(joints) @ cam["T"] if cam["frame"] == "flange" else cam["T"]
        W, H = NATIVE[name]
        v, u = np.mgrid[0:size[1], 0:size[0]] + 0.5
        xd, yd = (u * W / size[0] - W / 2) / cam["fx"], (v * H / size[1] - H / 2) / cam["fx"]
        x, y = xd, yd
        for _ in range(10):
            dist = 1 + cam["k1"] * (x * x + y * y)
            x, y = xd / dist, yd / dist
        d = np.stack([x, y, np.ones_like(x)], -1) @ T[:3, :3].T
        hit = self._cast(T[:3, 3], d)
        gid = hit["geometry_ids"].numpy().astype(np.int64)
        n = hit["primitive_normals"].numpy()
        light = np.array([0.3, 0.2, 0.93])
        shade = 0.35 + 0.65 * np.abs(n @ (light / np.linalg.norm(light)))
        img = np.zeros((*gid.shape, 3))
        ok = gid < len(self.colors)
        img[ok] = np.array(self.colors)[gid[ok]] * shade[ok, None]
        img[~ok] = (60, 60, 65)
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- a synthetic run

SWEEP = [(0.47, -0.10), (0.37, 0.00), (0.46, 0.06), (0.38, 0.14), (0.32, 0.24)]


def make_run(seed: int = 0) -> Path:
    """Overhead view from home, then a wrist sweep: written exactly like a recorded cell run, plus truth/."""
    cams, rng, objs = load(), np.random.default_rng(seed), build_scene()
    r = Renderer(objs, cams)
    rec = Recorder("synthetic", meta={"source": "synthetic", "seed": seed, "image_size": IMAGE_SIZE})
    truth = rec.dir / "truth"
    truth.mkdir()
    for o in objs:
        o["mesh"].export(truth / f"{o['id']:03d}.ply")
    (truth / "objects.json").write_text(json.dumps([{"id": o["id"], "name": o["name"], "rgb": o["rgb"],
                                                     "centroid": o["mesh"].centroid.tolist(), "mesh": f"{o['id']:03d}.ply"}
                                                    for o in objs], indent=2))
    rec.new_rollout()
    poses = [("overhead look from home", np.array(HOME))]
    poses += [(f"wrist over ({x * 1000:.0f}, {y * 1000:.0f}) mm", wrist_ik(np.array([x, y, SURFACE_Z + 0.12]))) for x, y in SWEEP]
    for note, q in poses:
        obs = {"joints": np.append(q, 0.0), "tcp_pose": tcp_pose_uf0(q), "t": None, "trial": None,
               "images": {c: r.rgb(c, joints=q) for c in ("table", "side", "wrist")},
               "depth": {c: r.depth(c, joints=q, rng=rng) for c in ("table", "wrist")}}
        rec.log(obs, note=note)
    return rec.dir


if __name__ == "__main__":
    from lab.replay import replay
    d = make_run()
    print(f"wrote {d}")
    replay(d, spawn=True)
