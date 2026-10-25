"""Check a planned motion against the robot's own model of its arm and the table, before sending it.

    report = check(rows, q0, obstacles_from(world))     # rows: the tcp rows about to be sent
    if not report.ok: ...                               # report.row, report.problem: the first thing that goes wrong

For every row, in order:
    1. joints by IK, starting only from the previous row's joints: no row may need the wrist to flip (a joint turning
       more than JUMP_DEG in one row). An IK that only works from somewhere else would make the arm whip round, as it
       did once in Isaac and threw a box 4 m; the real controller refuses such a move.
    2. the arm as simple shapes, placed by the forward kinematics: capsules along the upper arm, the forearm and the
       wrist, the gripper body, and the two fingers (where the jaws are at that row's opening).
    3. those shapes against what the robot believes is on the table: the table itself, each box as five slabs (its
       floor and four walls, WALL_MM thick), and the other things as boxes around their depth points. The fingers are
       allowed to touch things (gripping and raking is what they're for); everything above them is not.

It's a model, so it's deliberately a little generous (capsules a few mm fatter than the parts). The Isaac server's
fault check (and on the real cell, the controller) is the second line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import CHAIN, SURFACE_Z, UF0_Z_MM, _rot
from calibration.kinematics import FLANGE_FROM_TOOL, JAW_AXIS_DEG, JOINT_LIMITS, TCP_OFFSET, tool_ik, tool_pose, wpr_matrix
from lab import gripper as G

JUMP_DEG = 20.0       # most any joint may turn in one 0.1 s row
WALL_MM = 10.0        # the sorting boxes' walls and floor (Joel: "about 10 mm"; measure the real ones)
TABLE_MM = 4.0        # everything but the fingertips stays this far above the table
TIP_MM = 0.5          # the fingertips may come this close (a pick goes down to ~2.5 mm)
FLANGE_FACE = 0.070   # wrist centre to the flange face, along the tool
FINGER_ROOT = TCP_OFFSET - G.LENGTH  # where the fingers start, from the wrist centre


@dataclass
class Box:
    """An oriented box obstacle: centre (base, m), yaw (rad), half sizes (m)."""
    center: np.ndarray
    yaw: float
    half: np.ndarray
    what: str

    def distance(self, pts: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = pts - self.center
        local = np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, d[:, 2]], -1)
        out = np.maximum(np.abs(local) - self.half, 0)
        return np.linalg.norm(out, axis=1)


@dataclass
class Report:
    ok: bool
    row: int | None = None
    problem: str = ""
    joints: list = field(default_factory=list)


def obstacles_from(world, ignore=()) -> list[Box]:
    """The robot's beliefs as obstacles: each box (bin) as its floor and four walls, everything else as one box."""
    out = []
    for t in world.things.values():
        if t.id in ignore or t.status not in ("blob", "one", "bin", "stuck"):
            continue
        p = t.points[:, :3]
        (cx, cy), (w, h), ang = cv2.minAreaRect((p[:, :2] * 1000).astype(np.float32))
        yaw, hw, hh = np.radians(ang), w / 2000, h / 2000
        top = float(p[:, 2].max())
        if t.status != "bin":
            out.append(Box(np.array([cx / 1000, cy / 1000, (SURFACE_Z + top) / 2]), yaw,
                           np.array([hw, hh, (top - SURFACE_Z) / 2]), f"#{t.id}"))
            continue
        wall = WALL_MM / 2000
        c, s = np.cos(yaw), np.sin(yaw)
        ctr = np.array([cx / 1000, cy / 1000])
        zc = (SURFACE_Z + top) / 2
        H = (top - SURFACE_Z) / 2
        for sx, sy, half in ((1, 0, (wall, hh, H)), (-1, 0, (wall, hh, H)), (0, 1, (hw, wall, H)), (0, -1, (hw, wall, H))):
            off = np.array([sx * (hw - wall), sy * (hh - wall)])
            xy = ctr + [off[0] * c - off[1] * s, off[0] * s + off[1] * c]
            out.append(Box(np.array([*xy, zc]), yaw, np.array(half), f"box #{t.id}'s wall"))
        out.append(Box(np.array([*ctr, SURFACE_Z + wall]), yaw, np.array([hw, hh, wall]), f"box #{t.id}'s floor"))
    return out


def shapes(q, grip: float) -> list[tuple[str, np.ndarray, float, bool]]:
    """(name, sample points (N, 3), radius, is_finger) along the arm at joints q (deg)."""
    R, p = np.eye(3), np.zeros(3)
    joints = []
    for (off, axis, sign), a in zip(CHAIN, np.radians(np.asarray(q, float)[:6])):
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * a)
        joints.append(p.copy())
    tool = R[:, 0]  # the flange's x: along the gripper
    wc = joints[5]  # the wrist centre

    def seg(a, b, n=6):
        return np.linspace(a, b, n)
    out = [("base", seg(np.zeros(3), joints[1], 4), 0.07, False),
           ("upper arm", seg(joints[1], joints[2]), 0.055, False),
           ("forearm", seg(joints[3], joints[4], 8), 0.045, False),
           ("wrist", seg(wc, wc + tool * FLANGE_FACE, 3), 0.045, False),
           ("gripper body", seg(wc + tool * FLANGE_FACE, wc + tool * FINGER_ROOT, 6), 0.026, False)]
    tool_R = R @ FLANGE_FROM_TOOL.T
    jaw = tool_R @ [np.cos(np.radians(JAW_AXIS_DEG)), np.sin(np.radians(JAW_AXIS_DEG)), 0.0]
    half = G.STROKE * (1 - float(grip >= 0.5)) + G.DEPTH / 2
    for side in (-1, 1):
        root, tip = wc + tool * FINGER_ROOT + side * half * jaw, wc + tool * TCP_OFFSET + side * half * jaw
        out.append((f"{'left' if side < 0 else 'right'} finger", seg(root, tip, 8), 0.004, True))
    cam = _wrist_cam_points(q)
    out.append(("wrist camera", cam, 0.02, False))
    return out


_CAM = []


def _wrist_cam_points(q) -> np.ndarray:
    """The wrist depth camera's body: a short bar along its lens axis, bolted to the flange."""
    if not _CAM:
        from calibration.geometry import load
        _CAM.append(load()["wrist_depth"]["T"])
    from calibration.geometry import flange_pose
    T = flange_pose(q) @ _CAM[0]
    return np.array([T[:3, 3] - T[:3, 2] * 0.015, T[:3, 3], T[:3, 3] + T[:3, 0] * 0.03, T[:3, 3] - T[:3, 0] * 0.03])


# the arm against itself: what rides on the wrist against the links it could fold back into (neighbours excluded)
SELF = {"wrist camera": ("base", "upper arm", "forearm"), "gripper body": ("base", "upper arm", "forearm"),
        "left finger": ("base", "upper arm", "forearm"), "right finger": ("base", "upper arm", "forearm"),
        "wrist": ("base", "upper arm")}


def self_problem(parts) -> str | None:
    """The first part that would run into another part of the arm."""
    by = {name: (pts, rad) for name, pts, rad, _ in parts}
    wc = by["wrist"][0][0]
    for a, others in SELF.items():
        pa, ra = by[a]
        for b in others:
            pb, rb = by[b]
            if b == "forearm":  # the forearm's end is joined to the wrist: only the part that can fold onto it counts
                pb = pb[np.linalg.norm(pb - wc, axis=1) > 0.12]
                if not len(pb):
                    continue
            d = np.linalg.norm(pa[:, None] - pb[None], axis=2).min()
            if d < ra + rb:
                return f"the {a} would hit the arm's own {b}"
    return None


def pose_problem(q, grip: float, obstacles: list[Box] = (), held=None) -> str | None:
    """What the arm at joints q would hit (None: nothing). held: (local points (N, 3), T_tool_object, name): an object
    in the gripper, whose points are checked against the table and the obstacles too."""
    parts = shapes(q, grip)
    problem = self_problem(parts)
    if problem:
        return problem
    for name, pts, rad, finger in parts:
        if name == "base":  # bolted down below the table top: only the arm folding back into it matters
            continue
        low = pts[:, 2] - rad - SURFACE_Z
        limit = (TIP_MM if finger else TABLE_MM) / 1000
        if low.min() < limit:
            return f"the {name} would hit the table ({low.min() * 1000:.0f} mm above it)"
        for b in obstacles:
            if finger and not b.what.startswith("box"):
                continue  # fingers may touch things: that's gripping and raking
            if (b.distance(pts) - rad).min() < 0:
                return f"the {name} would hit {b.what}"
    if held is not None:
        local, X, what = held
        P = (tool_pose(q) @ X @ np.c_[local, np.ones(len(local))].T).T[:, :3]
        if (P[:, 2] - SURFACE_Z).min() < TABLE_MM / 1000:
            return f"the {what} it's holding would hit the table"
        for b in obstacles:
            if b.distance(P).min() < 0.002:
                return f"the {what} it's holding would hit {b.what}"
    return None


def check(rows, q0, obstacles: list[Box] = (), held=None) -> Report:
    """Walk the tcp rows in order (see the module docstring). Returns the first problem, or ok with the joints."""
    q = np.asarray(q0, float)[:6]
    qs = []
    for k, row in enumerate(np.atleast_2d(np.asarray(rows, float))):
        tip = np.array([row[0], row[1], row[2] - UF0_Z_MM]) / 1000
        try:
            q1 = tool_ik(tip, row[3:6], q, near=True)
        except ValueError as e:
            return Report(False, k, f"can't reach row {k}: {e}", qs)
        jump = np.abs(q1 - q)
        if jump.max() > JUMP_DEG:
            return Report(False, k, f"row {k} needs J{int(jump.argmax()) + 1} to turn {jump.max():.0f} deg at once "
                                    f"(a wrist flip): the arm would whip round", qs)
        problem = pose_problem(q1, row[6] if len(row) > 6 else 0.0, obstacles, held)
        if problem:
            return Report(False, k, f"row {k}: {problem}", qs)
        qs.append(q1)
        q = q1
    return Report(True, None, "", qs)


def check_joints(qs, grip: float, obstacles: list[Box] = (), held=None) -> Report:
    """The same for joint rows (deg): limits, jumps, and what the arm (and what it holds) would hit."""
    lo, hi = np.array(JOINT_LIMITS, float).T
    for k, q in enumerate(qs):
        q = np.asarray(q, float)[:6]
        if np.any(q < lo) or np.any(q > hi):
            j = int(np.argmax((q < lo) | (q > hi)))
            return Report(False, k, f"row {k}: J{j + 1} at {q[j]:.0f} deg is past its limit")
        if k and np.abs(q - np.asarray(qs[k - 1])[:6]).max() > JUMP_DEG:
            return Report(False, k, f"row {k}: a joint turns more than {JUMP_DEG:.0f} deg at once")
        problem = pose_problem(q, grip, obstacles, held)
        if problem:
            return Report(False, k, f"row {k}: {problem}")
    return Report(True, None, "", [np.asarray(q, float)[:6] for q in qs])


def tool_axis_down(wpr) -> bool:
    return wpr_matrix(*wpr)[2, 2] < -0.99
