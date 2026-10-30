"""The opening scan: how high it's safe to fly, then sweep the whole table, fast.

    tallest_mm, info = tallest(client)            # one table-camera depth picture: the tallest thing on the table
    tip = safe_tip_mm(tallest_mm)                 # fingertips this high: CLEAR_MM over it, never under MIN_TIP_MM
    rows = sweep(q, tip, speed)                   # joint rows: lines both ways over REGION, hops between them
    lowest_tip_mm(rows)                           # the check: the lowest the fingertips go on the way

The table camera looks straight down and measures depth, so the tallest thing is a height read off it directly (its
~8 mm noise is covered by the clearance); nothing has to be picked out of a picture. Lines go both ways (an edge along
the motion has no parallax, so both directions are needed), LINE_GAP apart: the wrist camera sees about +-240 x +-135 mm
of table from ~280 mm up, so neighbouring lines overlap.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, depth_to_base, load
from calibration.kinematics import DOWN, tool_ik
from lab import primitives as P

REGION = ((0.26, 0.50), (-0.30, 0.08))  # x, y (m): where things are put (the sim's layouts)
CLEAR_MM = 30.0
MIN_TIP_MM = 60.0
LINE_GAP = 0.10
HOP = 150.0  # mm/s between lines


def tallest(client, region=REGION) -> tuple[float, dict]:
    """The tallest thing over the region (mm above the table), from one table-camera depth picture: the 99.7th
    percentile of the heights there (a few noisy pixels don't count), leaving out the arm itself."""
    o = client.observe(["table"], (64, 36), "jpeg", depth=["table"])
    pts = depth_to_base(load()["table_depth"], o["depth"]["table"], stride=2)
    pts = pts[~near_arm(pts, o["joints"])]  # the arm stands over the table in this picture: it isn't on the table
    (x0, x1), (y0, y1) = region
    m = (pts[:, 0] > x0 - 0.05) & (pts[:, 0] < x1 + 0.05) & (pts[:, 1] > y0 - 0.05) & (pts[:, 1] < y1 + 0.05)
    h = (pts[m, 2] - SURFACE_Z) * 1000
    top = float(np.percentile(h, 99.7)) if len(h) else 0.0
    return top, {"pixels": int(m.sum()), "above_5mm": int((h > 5).sum())}


def near_arm(pts: np.ndarray, joints, margin: float = 0.05) -> np.ndarray:
    """Which points lie within `margin` of the arm or gripper at these joints: its capsule model (lab/armcheck.py),
    sampled every centimetre along each link (the rendered arm is a little fatter than the capsules)."""
    from scipy.spatial import cKDTree
    from lab.armcheck import shapes
    near = np.zeros(len(pts), bool)
    tree = cKDTree(pts)
    for _, sp, r, _ in shapes(np.asarray(joints, float)[:6], float(np.asarray(joints)[6]) if len(joints) > 6 else 0.0):
        sp = np.atleast_2d(sp)
        dense = [sp[:1]] + [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.01) + 1))[1:]
                            for a, b in zip(sp[:-1], sp[1:])]
        for c in np.concatenate(dense):
            near[tree.query_ball_point(c, r + margin)] = True
    return near


def safe_tip_mm(tallest_mm: float) -> float:
    return max(MIN_TIP_MM, tallest_mm + CLEAR_MM)


def lines(region=REGION, gap: float = LINE_GAP) -> list:
    """Lines along x at every `gap` of y, then along y at every `gap` of x, each the other way from the last."""
    (x0, x1), (y0, y1) = region
    out = []
    for i, y in enumerate(np.linspace(y0, y1, max(2, int(round((y1 - y0) / gap)) + 1))):
        out.append(((x0, y), (x1, y)) if i % 2 == 0 else ((x1, y), (x0, y)))
    for i, x in enumerate(np.linspace(x0, x1, max(2, int(round((x1 - x0) / gap)) + 1))):
        out.append(((x, y1), (x, y0)) if i % 2 == 0 else ((x, y0), (x, y1)))
    return out


def _tip_m(p):
    return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000


def cam_offset(q, T_fc) -> np.ndarray:
    """Where the wrist camera is relative to the fingertips, in x and y (m), with the fingers straight down."""
    from calibration.geometry import flange_pose
    from calibration.kinematics import tcp_of
    return (flange_pose(q) @ T_fc)[:2, 3] - tcp_of(q)[0][:2]


def for_camera(segs, offset) -> list:
    """Lines for the camera turned into lines for the fingertips: the camera rides `offset` ahead of them."""
    o = np.asarray(offset)
    return [(tuple(np.asarray(a) - o), tuple(np.asarray(b) - o)) for a, b in segs]


def sweep(q, tip_mm: float, speed: float, segs=None, start=None, say=print) -> list:
    """Joint rows along `segs` (fingertip lines; default: lines()) at fingertip height tip_mm, from joints q at UF0
    point `start` (default: the first line's start): hop to each line's start at HOP mm/s, glide along it at `speed`.
    Each row is solved on the same wrist branch as the one before; a leg that can't be is skipped."""
    segs = lines() if segs is None else segs
    p = P.uf0(*segs[0][0], tip_mm) if start is None else start
    rows, r = [], float(DOWN[2])
    for a, b in segs:
        for goal, v in ((P.uf0(*a, tip_mm), HOP), (P.uf0(*b, tip_mm), speed)):
            seg = []
            try:
                for row in P.line(p, goal, r, r, 0, v):
                    q = tool_ik(_tip_m(row[:3]), row[3:6], q, near=True)
                    seg.append([*q.tolist(), 0.0])
            except ValueError:
                say(f"skipped a leg to ({goal[0]:.0f}, {goal[1]:.0f}) mm: out of reach on this wrist branch")
                break
            rows += seg
            p = goal
    return rows + [rows[-1]] * 10


def lowest_tip_mm(rows) -> float:
    """The lowest the fingertips go along joint rows (mm above the table)."""
    tcp = np.array(P.tcp_rows(np.asarray(rows)[:, :7]))
    return float((tcp[:, 2] - P.TABLE_UF0).min())
