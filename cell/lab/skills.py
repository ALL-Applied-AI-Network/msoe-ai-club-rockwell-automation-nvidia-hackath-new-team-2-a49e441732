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
