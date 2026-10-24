"""Skills that need a plan checked against the robot's own model before they're tried (lab.armcheck).

pour_plan: pick a box up by its wall and pour it out. There's no single right way: tipping a box far moves the wrist
293 mm back from the fingertip, so which direction it can pour, how much it can swing, and how high it can lift depend
on where the box is. This tries them (direction every 30 deg, pivot lip / centre / tip, two heights) and keeps the
one that pours furthest (tilt), then swings most, then lands nearest to where the parts should go.
"""
from __future__ import annotations

import numpy as np

from lab import armcheck, primitives as P

PIVOTS = ("lip", "centre", "tip")  # most swing first
TILTS = (125.0, 110.0)              # degrees: 110 still tips the parts out in Isaac (the floor slopes past the lip)


def pour_plan(cur, q0, box_xy_m, obstacles=(), near_xy_m=None, dist_m: float = 0.11, **box) -> dict | None:
    """{"rows", "what", "toward", "pivot", "lift"} for a pour the arm can do, or None. near_xy_m: where the parts should
    preferably land (default: toward the middle of the table)."""
    b = np.asarray(box_xy_m, float)
    near = np.asarray(near_xy_m if near_xy_m is not None else (0.40, 0.0), float)
    cands = []
    for a in range(0, 360, 30):
        toward = b + dist_m * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        if not P.reachable(*toward, margin_mm=40):
            continue
        for rank, pivot in enumerate(PIVOTS):
            for t_rank, tilt in enumerate(TILTS):
                for lift in (200.0, 240.0):
                    cands.append((t_rank, rank, float(np.linalg.norm(toward - near)), lift, toward, pivot, tilt))
    cands.sort(key=lambda c: c[:3])  # pour well first, then swing most, then land nearest
    for _, rank, _, lift, toward, pivot, tilt in cands:
        rows, what = P.pour(cur, b, toward, lift_mm=lift, pivot=pivot, tilt_deg=tilt, **box)
        rep = armcheck.check(rows, q0, obstacles)
        if rep.ok:
            return {"rows": rows, "what": what + f" (pivot: {pivot}, {lift:.0f} mm up)", "toward": toward, "pivot": pivot,
                    "lift": lift, "joints": rep.joints}
    return None


# ---------------------------------------------------------------- the swing pour (Joel's way)

def _wall_grip_rows(cur, box, u, outer_mm, height_mm, wall_mm, lift_mm):
    """Over the wall on side u, down, close across it, lift straight up. Returns (rows, index of the first closed row)."""
    from calibration.kinematics import tool_r
    r = tool_r(float(np.degrees(np.arctan2(u[1], u[0]))))  # jaws close along u: across the wall
    reach = (outer_mm - wall_mm) / 2000
    grip_z = max(wall_mm + 2.0, height_mm - 33.0)  # most of the finger on the wall (20 mm deep let it pivot)
    wall_xy = box + u * reach
    high = max(P.SAFE_MM, height_mm + 30)
    rows = P.travel(cur, wall_xy, r, 0, high)
    over, at = P.uf0(*wall_xy, high), P.uf0(*wall_xy, grip_z)
    rows += P.line(over, at, r, r, 0, P.APPROACH) + P.hold(at, r, 0)
    closed = len(rows)
    rows += P.hold(at, r, 1, 8)
    rows += P.line(at, P.uf0(*wall_xy, lift_mm), r, r, 1, P.APPROACH)
    return rows, closed, r, at


def swing_pour(cur, q0, box_xy_m, box_yaw: float = 0.0, obstacles=(), outer_mm: float = 60.0, height_mm: float = 45.0,
               wall_mm: float = 10.0) -> dict | None:
    """Pour a box out the way a person does: hold it by a wall, lift it to a comfortable spot, and swing the wrist.

    Solved from the end: the end of the swing is the lifted pose with only J5 (the wrist's bend) turned 100-135 deg,
    which leaves the box mostly upside down; the swing is that one joint turning, sent as joint rows, so the box sweeps
    an arc and the parts fall out along it. Which wall to hold, how high to lift, which way and how far to swing:
    the first that lab.armcheck passes, the held box included (it must clear the table and the other boxes all the
    way round). Returns {"segments": [("tcp" | "joints", rows), ...], "what"} or None.
    obstacles: everything else (not this box)."""
    from calibration.geometry import SURFACE_Z
    from calibration.kinematics import tool_pose
    b = np.asarray(box_xy_m, float)
    radial = b / np.linalg.norm(b)
    normals = [np.array([np.cos(box_yaw + k * np.pi / 2), np.sin(box_yaw + k * np.pi / 2)]) for k in range(4)]
    normals.sort(key=lambda n: -abs(n @ radial))  # walls facing the base or away from it: J5 tips the box over them
    h, w = height_mm / 1000, outer_mm / 2000
    local = np.array([[x, y, z] for x in (-w, 0, w) for y in (-w, 0, w) for z in (0.0, h)])  # the box's outline points
    Tb = np.eye(4)
    Tb[:3, :3] = [[np.cos(box_yaw), -np.sin(box_yaw), 0], [np.sin(box_yaw), np.cos(box_yaw), 0], [0, 0, 1]]
    Tb[:3, 3] = [b[0], b[1], SURFACE_Z]
    from calibration.kinematics import tool_ik, wpr_of
    tried, best = [], None
    for u in normals:
        axis = np.cross([0.0, 0.0, 1.0], np.append(u, 0.0))  # horizontal, along the held wall: the box tips about it
        starts = {}  # lift -> (rows, closed, r, at, report): the grip and lift, checked once each
        def start(lift):
            if lift not in starts:
                rows, closed, r, at = _wall_grip_rows(cur, b, u, outer_mm, height_mm, wall_mm, lift)
                starts[lift] = (rows, closed, r, at, armcheck.check(rows, q0, obstacles))
            return starts[lift]
        first = start(140.0)
        if not first[4].ok:
            tried.append(first[4].problem)
            continue
        X = np.linalg.inv(tool_pose(first[4].joints[first[1]])) @ Tb  # the box in the tool's frame, once held
        held = (local, X, "box")
        # 1. the end states, lowest pour first: tipped d about the wall, lowest corner 20 mm up, beside home
        ends = []
        # (a 120 deg tip points the tool 30 deg above level, the wrist 150 mm below the fingertip: a low pour has to
        # tip less. The trade is made by the sort below: lowest pour first, whatever the tilt.)
        for d in (130.0, 120.0, 110.0):  # at 90 the parts just lie on the lower wall (Isaac): it has to go past level
            for sgn in (1.0, -1.0):
                Rt = P._axis_rot(axis, sgn * np.radians(d))
                for a in range(0, 360, 45):
                  for corner in (0.020, 0.040, 0.070, 0.100, 0.130):
                    Tend = np.eye(4)
                    Tend[:3, :3] = Rt @ Tb[:3, :3]
                    c = b + 0.09 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
                    pts = (Tend[:3, :3] @ local.T).T
                    Tend[:3, 3] = [c[0], c[1], SURFACE_Z + corner - pts[:, 2].min()]
                    mouth = Tend[:3, :3] @ [0, 0, h] + Tend[:3, 3]
                    land = mouth[:2] + Tend[:2, 2] * 0.02
                    if np.linalg.norm(land - b) < 0.055 or any(
                            np.linalg.norm(land - o.center[:2]) < 0.06 for o in obstacles if o.what.startswith("box")):
                        continue  # it would pour where the box goes back down, or onto another box
                    if not P.reachable(*c, margin_mm=30):
                        continue
                    ends.append((float(mouth[2] - SURFACE_Z) - 0.0005 * d, d, Tend, land))  # ties: tip more
        ends.sort(key=lambda e: e[0])
        # 2. for each, the backswing: the lowest start (lift) from which a straight joint swing down gets there
        for drop, d, Tend, land in ends:
            T_tool = Tend @ np.linalg.inv(X)
            for lift in (140.0, 180.0, 220.0, 260.0):
                rows, closed, r, at, rep = start(lift)
                if not rep.ok:
                    continue
                q_up = np.asarray(rep.joints[-1], float)
                try:  # the swing's end in joints, from the start of the swing (the same branch)
                    q_end = tool_ik(T_tool[:3, 3], wpr_of(T_tool[:3, :3]), q_up)
                except ValueError:
                    break  # unreachable whatever the start
                n = int(np.ceil(np.abs(q_end - q_up).max() / 10))  # <= 10 deg a row: a swing, not a creep
                path = [q_up + (q_end - q_up) * k / n for k in range(1, n + 1)]
                jr = armcheck.check_joints(path, 1.0, obstacles, held)
                if jr.ok:
                    best = (drop, rows, path, q_up, r, at, lift, d, land, Tend[2, 2])
                    break
                tried.append(jr.problem)
            if best is not None:
                break
        if best is not None:
            break
    if best is not None:
        drop, rows, path, q_up, r, at, lift, swing_deg, land, up_axis = best
        swing = [list(q) + [1.0] for q in path]
        back = swing[::-1][1:] + [list(q_up) + [1.0]]
        here = np.array(at[:2]) / 1000
        down = P.line(P.uf0(*here, lift), at, r, r, 1, P.APPROACH)
        down += P.hold(at, r, 0, 6) + P.line(at, P.uf0(*here, height_mm + 30), r, r, 0, P.APPROACH)
        return {"segments": [("tcp", rows), ("joints", swing + [swing[-1]] * 15 + back), ("tcp", down)],
                "lands": land.tolist(), "drop_mm": drop * 1000,
                "what": (f"pour the box at ({b[0] * 1000:.0f}, {b[1] * 1000:.0f}) mm: hold its wall, lift {lift:.0f} mm, "
                         f"swing it down, tipped {swing_deg:.0f} deg (the opening ends {drop * 1000:.0f} mm over the table), "
                         f"swing back, set it down"),
                "tip_deg": float(np.degrees(np.arccos(np.clip(up_axis, -1, 1))))}
    return {"segments": None, "why": tried[-3:]}
