"""Motion primitives: turn an intention ("grab here, put it there", "rake through this") into fingertip targets.

Every function returns rows [x, y, z mm, w, p, r deg, gripper] in UF0, one per DT (0.1 s), ready for the cell.
Fingers point straight down (w, p = -180, 0); r turns them, and the jaws close along the direction r points.
Positions given to these functions are in the robot base frame, metres (what perception works in).

The real arm trails its targets by ~0.8 s, so every place where timing matters (closing, opening, looking) holds still
for HOLD steps first: the arm has arrived before the gripper acts or the camera looks.
"""
from __future__ import annotations

import numpy as np

from calibration.kinematics import tool_r
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
    grip_yaw, place_yaw: the direction the jaws close (deg, base frame); turned into the tool's r here.
    drop_mm: let go with the part's bottom this high above the table (over a box: just above its rim) instead of
    setting it down. safe_mm: travel height of the fingertips (the part hangs grip_mm below them)."""
    jaw_yaw = grip_yaw
    place_yaw = tool_r(grip_yaw if place_yaw is None else place_yaw)
    grip_yaw = tool_r(grip_yaw)
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
    what = (f"grip at ({grip_xy_m[0] * 1000:.0f}, {grip_xy_m[1] * 1000:.0f}) mm, {grip_mm:.0f} mm up, jaws at {jaw_yaw:.0f}°; "
            f"{'set down' if drop_mm is None else 'drop'} at ({place_xy_m[0] * 1000:.0f}, {place_xy_m[1] * 1000:.0f})")
    return rows, what


def rake(cur, start_xy_m, end_xy_m, height_mm=3.0) -> tuple[list, str]:
    """Close the fingers, lower them just above the table at start, drag slowly to end, lift. A shuffle, not a push.
    Both ends are pulled inside what the arm can reach (a rake 25 mm past a part near the edge would leave the cell's box)."""
    start_xy_m, end_xy_m = clamp_xy(*start_xy_m), clamp_xy(*end_xy_m)
    d = np.asarray(end_xy_m) - np.asarray(start_xy_m)
    r = tool_r(float(np.degrees(np.arctan2(d[1], d[0]))))  # jaws along the drag: the closed fingers make a 12 mm wide blade
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


def _axis_rot(axis, angle: float) -> np.ndarray:
    """Rotation by angle (rad) about a unit axis (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def pour(cur, box_xy_m, toward_xy_m, outer_mm: float = 60.0, height_mm: float = 45.0, wall_mm: float = 10.0,
         lift_mm: float = 230.0, tilt_deg: float = 125.0, home_xy_m=None, flip: bool = False,
         pivot: str = "lip") -> tuple[list, str]:
    """Pick a box up by its wall and pour it out over toward_xy_m, the way a person would: grip the wall on the far
    side from the pour, lift high, carry it over, then swing it: the box turns about its far lip (the edge the parts
    pour over), so the wrist sweeps an arc rather than turning on the spot. Then back, and set it down where it was.

    The jaws close across the wall (their 12 mm opening over its 10 mm). flip turns the tool 180 deg: the same grip
    (the jaws are symmetric) but the wrist bends the other way as the box tips, which is what keeps J5 inside its
    +-120 deg in some places. pivot: what the box turns about: "lip" (the far top edge: the biggest swing), "centre"
    (a smaller swing) or "tip" (turning on the spot). Which of these the arm can do depends on where the box is and
    which way it pours (tipping it far moves the wrist 293 mm back from the fingertip): lab.skills.pour_plan tries
    them all through lab.armcheck and keeps one that works."""
    from calibration.kinematics import tool_r, wpr_matrix, wpr_of
    box = np.asarray(box_xy_m, float)
    home = box if home_xy_m is None else np.asarray(home_xy_m, float)
    u = box - np.asarray(toward_xy_m, float)  # from the pour spot to the box: grip the wall on this side
    u = u / max(np.linalg.norm(u), 1e-9)
    jaw = float(np.degrees(np.arctan2(u[1], u[0])))  # jaws close along u: across that wall
    r = tool_r(jaw + (180.0 if flip else 0.0))
    reach = (outer_mm - wall_mm) / 2000  # box centre to the wall's middle
    grip_z = max(wall_mm + 2.0, height_mm - 33.0)  # fingertips most of the way down the wall: a grip 20 mm deep let
                                                    # the box swing in the fingers like a pendulum (Isaac)
    wall_xy = box + u * reach
    rows = travel(cur, wall_xy, r, 0, max(SAFE_MM, height_mm + 30))
    over_w, at_w = uf0(*wall_xy, max(SAFE_MM, height_mm + 30)), uf0(*wall_xy, grip_z)
    rows += line(over_w, at_w, r, r, 0, APPROACH) + hold(at_w, r, 0)
    rows += hold(at_w, r, 1, 8)                                                       # close on the wall
    up = uf0(*wall_xy, lift_mm)
    rows += line(at_w, up, r, r, 1, APPROACH)                                         # lift
    # carry: the box's centre ends up over the pour spot (the wall, and the gripper, on the far side of it)
    spot = np.asarray(toward_xy_m, float) + u * reach
    over = uf0(*spot, lift_mm)
    rows += line(up, over, r, r, 1, TRAVEL)
    # swing about the far lip: the top edge on the pour side, across the box from the gripped wall
    R0 = wpr_matrix(-180.0, 0.0, r)
    tip0 = over / 1000 + [0, 0, -UF0_Z_MM / 1000]  # base frame, m
    lip = {"lip": tip0 + np.append(-u * (outer_mm - wall_mm / 2) / 1000, (height_mm - grip_z) / 1000),
           "centre": tip0 + np.append(-u * reach, (height_mm / 2 - grip_z) / 1000),
           "tip": tip0}[pivot]
    axis = np.cross([0.0, 0.0, 1.0], np.append(u, 0.0))  # horizontal, along the wall
    n = int(np.ceil(tilt_deg / 3))  # 3 deg a row (30 deg/s): faster, the wrist joints jump near the singularity
    swing = []
    for k in range(1, n + 1):
        Rt = _axis_rot(axis, -np.radians(tilt_deg) * k / n)  # the box's top turns away from the gripped wall
        tip = lip + Rt @ (tip0 - lip)
        swing.append([tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, *wpr_of(Rt @ R0), 1.0])
    rows += swing + [swing[-1]] * 15 + swing[::-1]                                    # pour, hold, swing back
    home_wall = home + u * reach
    rows += line(over, uf0(*home_wall, lift_mm), r, r, 1, TRAVEL)                    # carry it home
    rows += line(uf0(*home_wall, lift_mm), uf0(*home_wall, grip_z + 1), r, r, 1, APPROACH)
    rows += hold(uf0(*home_wall, grip_z + 1), r, 0, 6)                                # set down, open
    rows += line(uf0(*home_wall, grip_z + 1), uf0(*home_wall, height_mm + 30), r, r, 0, APPROACH)
    what = (f"pour the box at ({box[0] * 1000:.0f}, {box[1] * 1000:.0f}) mm out over ({toward_xy_m[0] * 1000:.0f}, "
            f"{toward_xy_m[1] * 1000:.0f}): grip its far wall, lift, swing it {tilt_deg:.0f} deg about its lip, put it back")
    return rows, what
