"""Does the arm hit itself at these joints? Padded, on the ER-4iA's real collision meshes (Isaac's robot asset, the
twin's: isaac/export_arm.py -> calibration/arm/er4ia_arm.npz), with what we've bolted on: the flange plate, the
gripper body and fingers, and the wrist camera. Used by every planner (lab/scan.unsafe) and by the Isaac cell, which
faults on it the way the real controller does.

    problem(q, grip=0)   None, or what would hit what (and how close)
    gaps(q, grip=0)      every pair's closest distance (mm), for checking

Why padded: the real arm faulted twice on Oct 30 with 13 mm (upper arm to forearm, folded back on an orbit) and
22 mm (reaching 509 mm out, 71 mm up) of air between the bare meshes: cables, the camera's mount and the gripper's
real bulk aren't in the CAD. PAD_MM is the space each pair must keep, set so both faults are flagged and none of the
4123 poses the arm ran without a fault are; REACH is the second rule (fault 1 isn't a closeness of any pair).

Parts joined at a joint aren't compared. Distances are between vertices about 4 mm apart, so they're good to ~2 mm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.geometry import CHAIN, _rot, load

ARM = Path(__file__).resolve().parent / "arm" / "er4ia_arm.npz"
LINKS = ["base", "j1", "j2", "j3", "j4", "j5", "j6"]
PAD_MM = {"j2-j4": 17.0}  # upper arm to forearm: real faults at 13.2 and 15.5 mm (Oct 30), fault-free poses never under 17.2
DEFAULT_PAD_MM = 10.0
TOOL_PAD_MM = 10.0         # the gripper and camera against the arm (they're modelled as fat as they are)
REACH_MM, FAR_MM, FAR_TIP_MM = 495.0, 470.0, 90.0  # fingertips: never further out (fault 1: 509 mm); beyond FAR_MM, at least FAR_TIP_MM up
TCP_OFFSET = 0.293         # wrist centre to fingertips along the tool (calibration/kinematics.TCP_OFFSET)
FLANGE = 0.070             # wrist centre to the flange face
GRIP_BODY = (0.070, 0.217, 0.030)  # the gripper body: from FLANGE to TCP_OFFSET - finger length, radius (m)
_M: dict = {}


def _frames(q) -> list[np.ndarray]:
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
    from scipy.spatial import cKDTree  # noqa: F401  (checked here so a missing scipy fails early)
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
        Li = np.linalg.inv(rest[k])
        local = world_rest @ Li[:3, :3].T + Li[:3, 3]
        keep = np.unique(np.round(local / 0.004).astype(int), axis=0, return_index=True)[1]
        _M[link] = (k, local[keep])
    cam = load().get("wrist_depth") or load().get("wrist")
    _M["_cam_T"] = np.asarray(cam["T"], float)
    return _M


def _tool(F6, grip: float) -> dict:
    """The bolted-on parts as (points, radius): the gripper body along the tool axis, the two fingers, the camera."""
    wc, tool = F6[:3, 3], F6[:3, 0]  # the wrist centre; the tool along the flange's +x
    a, b, r = GRIP_BODY
    body = wc + np.linspace(a, b, 8)[:, None] * tool
    jaw = F6[:3, 2]  # the jaws close along the flange's z (calibration/kinematics: JAW_AXIS_DEG 90 about the tool)
    half = 0.010 * (1 - float(grip >= 0.5)) + 0.005
    fingers = np.concatenate([wc + np.linspace(b, TCP_OFFSET, 6)[:, None] * tool + s * half * jaw for s in (-1, 1)])
    T = F6 @ _load()["_cam_T"]
    cam = np.array([T[:3, 3] + T[:3, 0] * dx + T[:3, 1] * dy - T[:3, 2] * dz
                    for dx in (-0.045, 0.0, 0.045) for dy in (-0.012, 0.012) for dz in (0.0, 0.02)])
    return {"gripper": (body, r), "fingers": (fingers, 0.006), "camera": (cam, 0.008)}


PAIRS = [("base", "j3"), ("base", "j4"), ("base", "j5"), ("base", "j6"),
         ("j1", "j3"), ("j1", "j4"), ("j1", "j5"), ("j1", "j6"),
         ("j2", "j4"), ("j2", "j5"), ("j2", "j6"),
         ("j3", "j5"), ("j3", "j6")]
TOOL_VS = ["base", "j1", "j2", "j3"]  # (the forearm J4 holds the wrist: what folds onto it is the links above)


def gaps(q, grip: float = 0.0) -> dict:
    from scipy.spatial import cKDTree
    M = _load()
    F = _frames(q)
    world = {link: M[link][1] @ F[M[link][0]][:3, :3].T + F[M[link][0]][:3, 3] for link in LINKS if link in M}
    trees = {k: cKDTree(v) for k, v in world.items()}
    out = {}
    for a, b in PAIRS:
        d, _ = trees[a].query(world[b], k=1)
        out[f"{a}-{b}"] = float(d.min() * 1000)
    for name, (pts, r) in _tool(F[6], grip).items():
        for b in TOOL_VS:
            d, _ = trees[b].query(pts, k=1)
            out[f"{name}-{b}"] = float((d - r).min() * 1000)
    return out


def reach_problem(q) -> str | None:
    F6 = _frames(q)[6]
    tip = F6[:3, 3] + F6[:3, 0] * TCP_OFFSET
    from calibration.geometry import SURFACE_Z
    reach, up = float(np.hypot(tip[0], tip[1]) * 1000), float((tip[2] - SURFACE_Z) * 1000)
    if reach > REACH_MM + 15:
        return f"the fingertips {reach:.0f} mm out from the base (the arm hit itself reaching 509 mm out)"
    if reach > FAR_MM + 15 and up < FAR_TIP_MM - 10:
        return f"the fingertips {reach:.0f} mm out and only {up:.0f} mm up (the arm hit itself reaching far and low)"
    return None


def problem(q, grip: float = 0.0) -> str | None:
    """None if the arm keeps every padded space at joints q, else which parts would come too close."""
    g = gaps(q, grip)
    worst, why = None, None
    for k, d in g.items():
        pad = PAD_MM.get(k, TOOL_PAD_MM if k.split("-")[0] in ("gripper", "fingers", "camera") else DEFAULT_PAD_MM)
        if d < pad and (worst is None or d - pad < worst):
            worst, why = d - pad, f"the {k.replace('-', ' against ')} ({max(d, 0):.0f} mm, needs {pad:.0f})"
    return f"self-collision: {why}" if why else reach_problem(q)
