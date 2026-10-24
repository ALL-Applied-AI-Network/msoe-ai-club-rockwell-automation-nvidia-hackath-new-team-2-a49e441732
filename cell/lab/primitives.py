"""Motion primitives: turn an intention ("grab here, put it there", "rake through this") into fingertip targets.

Every function returns rows [x, y, z mm, w, p, r deg, gripper] in UF0, one per DT (0.1 s), ready for the cell.
Fingers point straight down (w, p = -180, 0); r turns them, and the jaws close along the direction r points.
Positions given to these functions are in the robot base frame, metres (what perception works in).

The real arm trails its targets by ~0.8 s, so every place where timing matters (closing, opening, looking) holds still
for HOLD steps first: the arm has arrived before the gripper acts or the camera looks.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM

DT = 0.1
TRAVEL = 150.0     # mm/s in free space
APPROACH = 50.0    # mm/s near the table and objects
HOLD = 12          # 1.2 s: longer than the arm's lag
SAFE_MM = 60.0     # travel height above the table (fingertips); the tallest parts here are ~35 mm
TABLE_UF0 = SURFACE_Z * 1000 + UF0_Z_MM
MIN_Z_UF0 = -334.0  # the cell refuses anything lower
TCP_X, TCP_Y = (155.0, 645.0), (-445.0, 445.0)  # the cell's box (mm, UF0), with 5 mm to spare
REACH_MM = 540.0


def reachable(x_m: float, y_m: float, margin_mm: float = 0.0) -> bool:
    """Can the fingertip get there (with margin_mm to spare)? Inside the cell's box and the arm's ~550 mm reach."""
    x, y, m = x_m * 1000, y_m * 1000, margin_mm
    return (TCP_X[0] + m <= x <= TCP_X[1] - m and TCP_Y[0] + m <= y <= TCP_Y[1] - m
            and np.hypot(x, y) <= REACH_MM - m)


def clamp_xy(x_m: float, y_m: float) -> np.ndarray:
    """The nearest reachable fingertip position."""
    p = np.array([np.clip(x_m * 1000, *TCP_X), np.clip(y_m * 1000, *TCP_Y)])
    r = np.hypot(*p)
    return (p * min(1.0, REACH_MM / r) if r > 0 else p) / 1000


def uf0(x_m: float, y_m: float, above_table_mm: float) -> np.ndarray:
    return np.array([x_m * 1000, y_m * 1000, max(MIN_Z_UF0, TABLE_UF0 + above_table_mm)])


def _row(p, r_deg, g) -> list[float]:
    return [float(p[0]), float(p[1]), float(p[2]), -180.0, 0.0, float(r_deg), float(g)]


def line(a, b, r0, r1, g, speed) -> list[list[float]]:
    """Straight move a -> b (UF0 mm), turning r0 -> r1 on the way, at `speed` mm/s (and at most 9 deg per step)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dr = (r1 - r0 + 180) % 360 - 180
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / (speed * DT))), int(np.ceil(abs(dr) / 9)))
    return [_row(a + (b - a) * f, r0 + dr * f, g) for f in np.linspace(0, 1, n + 1)[1:]]


def hold(p, r, g, n=HOLD) -> list[list[float]]:
    return [_row(p, r, g)] * n


def travel(cur, goal_xy_m, r1, g, height_mm=SAFE_MM) -> list[list[float]]:
    """Up to the safe height, across, (no descent). cur: [x, y, z mm UF0, w, p, r]."""
    p0 = np.asarray(cur[:3], float)
    r0 = float(cur[5])
    up = np.array([p0[0], p0[1], max(p0[2], TABLE_UF0 + height_mm)])
    over = uf0(goal_xy_m[0], goal_xy_m[1], height_mm)
    return line(p0, up, r0, r0, g, TRAVEL) + line(up, over, r0, r1, g, TRAVEL)


def pick_and_place(cur, grip_xy_m, grip_mm, grip_yaw, place_xy_m, place_yaw=None, drop_mm: float | None = None,
                   safe_mm: float = SAFE_MM) -> tuple[list, str]:
    """Open, go over the grip, descend to grip_mm above the table, close, lift, carry, set down, open, lift.
    drop_mm: let go with the part's bottom this high above the table (over a box: just above its rim) instead of
    setting it down. safe_mm: travel height of the fingertips (the part hangs grip_mm below them)."""
    place_yaw = grip_yaw if place_yaw is None else place_yaw
    over_g, at_g = uf0(*grip_xy_m, safe_mm), uf0(*grip_xy_m, grip_mm)
    over_p = uf0(*place_xy_m, safe_mm)
    at_p = uf0(*place_xy_m, grip_mm + 1.5 if drop_mm is None else drop_mm + grip_mm)
    rows = travel(cur, grip_xy_m, grip_yaw, 0, safe_mm)
    rows += line(over_g, at_g, grip_yaw, grip_yaw, 0, APPROACH) + hold(at_g, grip_yaw, 0)          # arrive, then
    rows += hold(at_g, grip_yaw, 1, 8)                                                              # close
    rows += line(at_g, over_g, grip_yaw, grip_yaw, 1, APPROACH)                                     # lift
    rows += line(over_g, over_p, grip_yaw, place_yaw, 1, TRAVEL)                                    # carry
    rows += line(over_p, at_p, place_yaw, place_yaw, 1, APPROACH) + hold(at_p, place_yaw, 1)        # set down, then
    rows += hold(at_p, place_yaw, 0, 6) + line(at_p, over_p, place_yaw, place_yaw, 0, APPROACH)    # open, lift
    rows += hold(over_p, place_yaw, 0)
    what = (f"grip at ({grip_xy_m[0] * 1000:.0f}, {grip_xy_m[1] * 1000:.0f}) mm, {grip_mm:.0f} mm up, jaws at {grip_yaw:.0f}°; "
            f"{'set down' if drop_mm is None else 'drop'} at ({place_xy_m[0] * 1000:.0f}, {place_xy_m[1] * 1000:.0f})")
    return rows, what


def rake(cur, start_xy_m, end_xy_m, height_mm=3.0) -> tuple[list, str]:
    """Close the fingers, lower them just above the table at start, drag slowly to end, lift. A shuffle, not a push.
    Both ends are pulled inside what the arm can reach (a rake 25 mm past a part near the edge would leave the cell's box)."""
    start_xy_m, end_xy_m = clamp_xy(*start_xy_m), clamp_xy(*end_xy_m)
    d = np.asarray(end_xy_m) - np.asarray(start_xy_m)
    r = float(np.degrees(np.arctan2(d[1], d[0])))  # jaws along the drag: the closed fingers make a 12 mm wide blade
    over_s, at_s = uf0(*start_xy_m, SAFE_MM), uf0(*start_xy_m, height_mm)
    at_e, over_e = uf0(*end_xy_m, height_mm), uf0(*end_xy_m, SAFE_MM)
    rows = travel(cur, start_xy_m, r, 1)
    rows += line(over_s, at_s, r, r, 1, APPROACH) + hold(at_s, r, 1, 6)
    rows += line(at_s, at_e, r, r, 1, 30.0)
    rows += line(at_e, over_e, r, r, 1, APPROACH) + hold(over_e, r, 1)
    what = (f"rake from ({start_xy_m[0] * 1000:.0f}, {start_xy_m[1] * 1000:.0f}) to "
            f"({end_xy_m[0] * 1000:.0f}, {end_xy_m[1] * 1000:.0f}) mm, {height_mm:.0f} mm above the table")
    return rows, what


def look(cur, spot_xy_m, cam_offset_mm, height_mm=70.0) -> tuple[list, str]:
    """Hold the wrist camera over a spot (the camera sits cam_offset_mm from the fingertip) and keep still."""
    tip = clamp_xy(*(np.asarray(spot_xy_m, float) - np.asarray(cam_offset_mm) / 1000))  # the camera may land off-centre
    goal = uf0(tip[0], tip[1], height_mm)
    p0 = np.asarray(cur[:3], float)
    rows = line(p0, np.array([p0[0], p0[1], max(p0[2], goal[2])]), float(cur[5]), 0.0, 0, TRAVEL) if p0[2] < goal[2] else []
    rows += line(rows[-1][:3] if rows else p0, goal, 0.0 if rows else float(cur[5]), 0.0, 0, TRAVEL) + hold(goal, 0.0, 0)
    return rows, f"look at ({spot_xy_m[0] * 1000:.0f}, {spot_xy_m[1] * 1000:.0f}) mm"


def view(cur, tip_xy_m, yaw: float, height_mm: float = 70.0) -> tuple[list, str]:
    """Fingertip to tip_xy, wrist turned to yaw (which swings the camera round the fingertip), then keep still."""
    goal = uf0(*clamp_xy(*tip_xy_m), height_mm)
    p0 = np.asarray(cur[:3], float)
    mid = np.array([p0[0], p0[1], max(p0[2], goal[2])])
    rows = (line(p0, mid, float(cur[5]), float(cur[5]), 0, TRAVEL) if p0[2] < goal[2] else [])
    rows += line(mid, goal, float(cur[5]), yaw, 0, TRAVEL) + hold(goal, yaw, 0)
    return rows, f"look from ({goal[0]:.0f}, {goal[1]:.0f}) mm, wrist {yaw:.0f}°"
