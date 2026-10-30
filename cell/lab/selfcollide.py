"""Does the arm hit itself at these joints? Checked on the ER-4iA's real collision meshes (Isaac's robot asset, the
twin's: isaac/export_arm.py -> calibration/arm/er4ia_arm.npz), plus the gripper and the wrist camera.

    gap_mm(q)          the closest any two parts that can touch come (mm; <= 0: they overlap)
    hits(q)            None, or which two parts come closer than CLEAR_MM

The links are placed by our own chain (calibration/geometry.CHAIN, which the sim feeds straight into the asset, so
they agree): each mesh rides its link, moved from where the asset has it at all-zero joints. Parts joined at a joint
(base-J1, J1-J2, ..., J5-J6, J6-gripper) aren't compared. Distances are between mesh vertices (a few mm apart), so
the answer is good to about their spacing: CLEAR_MM covers it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from calibration.geometry import CHAIN, _rot

CLEAR_MM = 12.0
ARM = Path(__file__).resolve().parents[1] / "calibration" / "arm" / "er4ia_arm.npz"
LINKS = ["base", "j1", "j2", "j3", "j4", "j5", "j6"]
_M: dict = {}


def _frames(q) -> list[np.ndarray]:
    """4x4 world poses of the base and links J1..J6 at joints q (deg): the same chain as geometry.flange_pose."""
    T = np.eye(4)
    out = [T.copy()]
    for (off, axis, sign), a in zip(CHAIN, np.radians(np.asarray(q, float)[:6])):
        step = np.eye(4)
        step[:3, 3] = off
        rot = np.eye(4)
        rot[:3, :3] = _rot(axis, sign * a)
        T = T @ step @ rot
        out.append(T.copy())
    return out


def _load():
    if _M:
        return _M
    z = np.load(ARM, allow_pickle=True)
    rest = _frames(np.zeros(6))
    for i, name in enumerate(z["names"]):
        parts = str(name).split("/")
        if "collisions" not in parts:
            continue
        link = parts[parts.index("collisions") + 1]
        k = LINKS.index(link)
        pts = np.asarray(z[f"mesh{i}_pts"], float)
        W = np.asarray(z[f"mesh{i}_world"], float)
        world_rest = pts @ W[:3, :3].T + W[:3, 3]
        Li = np.linalg.inv(rest[k])  # into the link's own frame (where it is at all-zero joints)
        local = world_rest @ Li[:3, :3].T + Li[:3, 3]
        keep = np.unique(np.round(local / 0.004).astype(int), axis=0, return_index=True)[1]  # ~4 mm apart
        _M[link] = (k, local[keep])
    return _M


def _tool_parts(q, grip: float):
    """The gripper body, fingers and the wrist camera as points (from lab/armcheck's shapes), in the world."""
    from lab import armcheck
    out = {}
    for name, pts, r, _ in armcheck.shapes(np.asarray(q, float)[:6], grip):
        if name in ("gripper body", "wrist camera", "left finger", "right finger"):
            out[name] = (np.atleast_2d(pts), r)
    return out


# what can touch what (neighbours across a joint can't: they're built to move against each other)
PAIRS = [("base", "j3"), ("base", "j4"), ("base", "j5"), ("base", "j6"),
         ("j1", "j3"), ("j1", "j4"), ("j1", "j5"), ("j1", "j6"),
         ("j2", "j4"), ("j2", "j5"), ("j2", "j6"),
         ("j3", "j5"), ("j3", "j6")]
TOOL_VS = ["base", "j1", "j2", "j3", "j4"]


def parts_at(q, grip: float = 0.0):
    M = _load()
    F = _frames(q)
    world = {}
    for link, (k, local) in M.items():
        world[link] = local @ F[k][:3, :3].T + F[k][:3, 3]
    return world, _tool_parts(q, grip)


def gaps(q, grip: float = 0.0) -> dict:
    """Closest distance (mm) between every pair that can touch."""
    world, tool = parts_at(q, grip)
    out = {}
    trees = {k: cKDTree(v) for k, v in world.items()}
    for a, b in PAIRS:
        if a in world and b in world:
            d, _ = trees[a].query(world[b], k=1)
            out[f"{a}-{b}"] = float(d.min() * 1000)
    for name, (pts, r) in tool.items():
        for b in TOOL_VS:
            if b in world:
                d, _ = trees[b].query(pts, k=1)
                out[f"{name}-{b}"] = float((d - r).min() * 1000)
    return out


def gap_mm(q, grip: float = 0.0) -> tuple[float, str]:
    g = gaps(q, grip)
    k = min(g, key=g.get)
    return g[k], k


def hits(q, grip: float = 0.0) -> str | None:
    d, pair = gap_mm(q, grip)
    if d < CLEAR_MM:
        a, b = pair.split("-")
        return f"the arm would hit itself ({a} against {b}: {max(d, 0):.0f} mm apart)"
    return None
