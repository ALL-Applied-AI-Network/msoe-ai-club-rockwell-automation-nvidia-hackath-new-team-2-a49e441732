"""The three moves, nothing else: each gives the fingertip rows for one job, and joints() turns them into joint rows that
all pass the padded self-collision check (lab/scan.unsafe) or says why not.

    pick_top(cur, at, jaw, place, travel_mm, drop_mm)        long and large things (pens, markers, batteries, cups by
                                                             the wall): over it, straight down to GRIP_MM, close, carry
    push_grasp(cur, at, jaw, length_mm, place, travel_mm, drop_mm, clear)
                                                             small things (blocks, bolts): down BESIDE it to GRIP_MM, the
                                                             jaws open, slide across so it ends between the fingers,
                                                             close (the closing jaws centre it and square it up: the
                                                             slide forgives ~10 mm along it), carry
    push(cur, at, radius_mm, goal, height_mm, travel_mm)     cups, bins, boxes: fingers closed, down behind it, push it
                                                             to the goal, up

`jaw` is the direction the jaws close (deg, base frame): across the thing's narrow side. Heights are above the table.
`travel_mm` is how high the fingertips fly between places: the caller makes it clear everything on the way, the target
included. `clear(xy)` says whether a spot is free for the fingers (push_grasp tries the other side when not).
"""
from __future__ import annotations

import numpy as np

from lab import primitives as P
from lab.primitives import APPROACH, hold, line, tool_r, travel, uf0

GRIP_MM = 4.0      # fingertips this far off the table to grip (low: the height reading doesn't have to be exact)
SLIDE_MM_S = 20.0  # across the table into a push-grasp, slow
PUSH_MM_S = 40.0
BESIDE_MM = 12.0   # a push-grasp starts this far past the thing's edge


def _jaw_dirs(jaw: float):
    a = np.radians(jaw)
    across = np.array([np.cos(a), np.sin(a)])     # the jaws close along this
    along = np.array([-np.sin(a), np.cos(a)])     # the gap between the fingers runs along this: the slide goes this way
    return across, along


def _carry(rows, at_g, over_g, place, r, travel_mm, drop_mm):
    over_p = uf0(*place, travel_mm)
    at_p = uf0(*place, GRIP_MM + 1.5 if drop_mm is None else drop_mm + GRIP_MM)
    rows += hold(at_g, r, 1, 8)                                                   # close
    rows += line(at_g, over_g, r, r, 1, APPROACH)                                 # lift
    rows += line(over_g, over_p, r, r, 1, P.TRAVEL)                               # carry
    rows += line(over_p, at_p, r, r, 1, APPROACH) + hold(at_p, r, 1)              # down
    rows += hold(at_p, r, 0, 6) + line(at_p, over_p, r, r, 0, APPROACH) + hold(over_p, r, 0)  # open, up
    return rows


def pick_top(cur, at, jaw: float, place, travel_mm: float, drop_mm: float | None = None) -> list:
    r = tool_r(jaw)
    over_g, at_g = uf0(*at, travel_mm), uf0(*at, GRIP_MM)
    rows = travel(cur, at, r, 0, travel_mm)
    rows += line(over_g, at_g, r, r, 0, APPROACH) + hold(at_g, r, 0)
    return _carry(rows, at_g, over_g, place, r, travel_mm, drop_mm)


def push_grasp(cur, at, jaw: float, length_mm: float, place, travel_mm: float, drop_mm: float | None = None,
               clear=lambda xy: True) -> tuple[list, str]:
    r = tool_r(jaw)
    _, along = _jaw_dirs(jaw)
    at = np.asarray(at, float)
    back = (length_mm / 2 + BESIDE_MM) / 1000
    for sgn in (1.0, -1.0):  # start on whichever side is free for the fingers
        start = at - sgn * along * back
        if clear(start):
            break
    else:
        return [], "no free spot beside it for the fingers to come down"
    over_s, at_s, at_g = uf0(*start, travel_mm), uf0(*start, GRIP_MM), uf0(*at, GRIP_MM)
    rows = travel(cur, start, r, 0, travel_mm)
    rows += line(over_s, at_s, r, r, 0, APPROACH) + hold(at_s, r, 0, 4)          # down beside it, jaws open
    rows += line(at_s, at_g, r, r, 0, SLIDE_MM_S) + hold(at_g, r, 0, 3)          # slide: it ends between the fingers
    return _carry(rows, at_g, uf0(*at, travel_mm), place, r, travel_mm, drop_mm), "ok"


def push(cur, at, radius_mm: float, goal, height_mm: float, travel_mm: float) -> list:
    at, goal = np.asarray(at, float), np.asarray(goal, float)
    u = goal - at
    u = u / max(np.linalg.norm(u), 1e-9)
    r = tool_r(float(np.degrees(np.arctan2(u[1], u[0]))) + 90.0)  # the closed fingers side-on: a wide paddle
    back = (radius_mm + 15.0) / 1000
    start, end = at - u * back, goal - u * (radius_mm + 5.0) / 1000
    over_s, at_s, at_e = uf0(*start, travel_mm), uf0(*start, height_mm), uf0(*end, height_mm)
    rows = travel(cur, start, r, 1, travel_mm)
    rows += line(over_s, at_s, r, r, 1, APPROACH) + hold(at_s, r, 1, 3)
    rows += line(at_s, at_e, r, r, 1, PUSH_MM_S) + hold(at_e, r, 1, 3)
    rows += line(at_e, uf0(*end, travel_mm), r, r, 1, APPROACH) + hold(uf0(*end, travel_mm), r, 1)
    return rows


def joints(rows, q) -> tuple[list | None, str]:
    """Fingertip rows -> joint rows, each solved on the branch of the one before, every one checked (lab/scan.unsafe).
    (None, why) if any can't be reached or isn't safe: the move then isn't sent at all."""
    from calibration.geometry import UF0_Z_MM
    from calibration.kinematics import tool_ik
    from lab.scan import unsafe
    out = []
    q = np.asarray(q, float)[:6]
    for k, row in enumerate(rows):
        try:
            q = tool_ik(np.array([row[0], row[1], row[2] - UF0_Z_MM]) / 1000, row[3:6], q, near=True)
        except ValueError:
            return None, "a pose on the way is out of reach"
        jr = [*q.tolist(), float(row[6])]
        if k % 2 == 0 and (w := unsafe(jr)):
            return None, w
        out.append(jr)
    return out, "ok"
