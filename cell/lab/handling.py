"""Pick and place with every height solved from the measured shapes (lab/objects.py), none written in by hand.

    plan = put(o, dest, objs, cols, cur_tcp)     o: the object to move; dest: a container (or None: the table at
                                                 plan.place_xy); returns the moves and every number behind them

The motion is the simple one (lab/primitives.pick_and_place): up, across, down, close, up, across, down, open, up.
The numbers come from the shapes:
    grip height   fingertips below the object's middle (the hooked tips cage it from under its widest part) with at
                  least GRIP_ON_MM of finger on it, never under TIP_MIN_MM; the jaw angle and centre from the grasp
                  planner (lab/grasp.from_outline) on its top outline, everything taller than the grip height around
                  it an obstacle for the fingers
    hang          how far the object reaches below the fingertips: grip height - its bottom (the table)
    carry         the highest matter under the swept path (columns of the grid, the object itself left out) + hang +
                  CLEAR_MM: the object's bottom clears everything on the way
    release       into a container: its measured rim + hang + CLEAR_MM, over the middle of its opening, and only if the
                  object's top outline fits inside the opening (turned with the wrist, FIT_MM to spare);
                  on the table: the table + hang + SET_MM (set down, not dropped, not pressed)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.geometry import SURFACE_Z
from lab import grasp
from lab.depth_bench import Poly
from lab.objects import Obj

TIP_MIN_MM = 2.5   # fingertips at least this far off the table (they drag on it at 1 mm)
GRIP_ON_MM = 5.0   # at least this much of the fingers on the object
CLEAR_MM = 10.0
SET_MM = 1.5
FIT_MM = 4.0
CORRIDOR_MM = 25.0  # half-width of the swept path checked for matter (fingers 20 mm open + their 10 mm depth, roughly)


@dataclass
class Plan:
    ok: bool
    why: str
    grip: grasp.Grip | None = None
    grip_mm: float = 0.0
    hang_mm: float = 0.0
    carry_mm: float = 0.0
    release_mm: float = 0.0  # fingertips above the table when letting go
    place_xy: np.ndarray | None = None
    path_max_mm: float = 0.0
    notes: list = field(default_factory=list)


def grip_height(o: Obj) -> float:
    return float(max(TIP_MIN_MM, min(o.top_mm / 2 - 1.0, o.top_mm - GRIP_ON_MM)))


def column_heights(cols: dict) -> np.ndarray:
    """mm above the table of the highest something in each column (-inf: nothing)."""
    top_i = np.asarray(cols["top"], int)
    return np.where(top_i >= 0, (cols["z0"] + (top_i + 0.5) * cols["vox"] - SURFACE_Z) * 1000, -np.inf)


def path_max(cols: dict, h: np.ndarray, a, b, skip: Obj | None, half_mm: float = CORRIDOR_MM) -> float:
    """The highest matter (mm) within half_mm of the straight path a -> b (base m), not counting `skip`'s columns."""
    vox = cols["vox"]
    X, Y = h.shape
    xs = cols["x0"] + (np.arange(X) + 0.5) * vox
    ys = cols["y0"] + (np.arange(Y) + 0.5) * vox
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L2 = max(float(d @ d), 1e-12)
    t = np.clip(((gx - a[0]) * d[0] + (gy - a[1]) * d[1]) / L2, 0, 1)
    dist = np.hypot(gx - (a[0] + t * d[0]), gy - (a[1] + t * d[1]))
    m = dist <= half_mm / 1000
    if skip is not None and skip.cells is not None:
        m[skip.cells[:, 0], skip.cells[:, 1]] = False
    return float(h[m].max()) if np.isfinite(h[m]).any() else 0.0


def fits(o: Obj, opening: np.ndarray, at, margin_mm: float = FIT_MM) -> float | None:
    """A wrist turn (deg) at which the object's top outline, centred at `at`, lies inside the opening with margin to
    spare; None if none does."""
    c = o.top_footprint.mean(0)
    shrunk = opening.mean(0) + (opening - opening.mean(0)) * (1 - margin_mm / 1000 / np.maximum(
        np.linalg.norm(opening - opening.mean(0), axis=1, keepdims=True), 1e-6))
    poly = Poly(shrunk)
    for turn in np.arange(0, 180, 15):
        a = np.radians(turn)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = (o.top_footprint - c) @ R.T + np.asarray(at)
        if poly.contains_points(pts).all():
            return float(turn)
    return None


def put(o: Obj, dest: Obj | None, objs: list[Obj], cols: dict, cur_tcp, place_xy=None) -> Plan:
    h = column_heights(cols)
    tip = grip_height(o)
    others = [x for x in objs if x.id != o.id and x.top_mm > tip]  # what the fingers could land on at that height
    grips = grasp.from_outline(o.top_footprint, [x.footprint for x in others], tip)
    if not grips:
        return Plan(False, f"no grip on #{o.id}: its top outline doesn't fit the 20 mm jaws with room at "
                           f"{tip:.1f} mm, or the fingers would land on a neighbour")
    g = grips[0]
    hang = tip  # its bottom is on the table
    if dest is not None:
        if not dest.container or dest.opening is None:
            return Plan(False, f"#{dest.id} isn't a container (no open space seen inside it)")
        place = dest.opening.mean(0)
        turn = fits(o, dest.opening, place)
        if turn is None:
            return Plan(False, f"#{o.id} doesn't fit in #{dest.id}'s opening")
        release = dest.rim_mm + hang + CLEAR_MM
    else:
        place = np.asarray(place_xy, float)
        release = hang + SET_MM
    pick = np.array([g.x, g.y])
    here = np.asarray(cur_tcp[:2], float) / 1000
    hi = max(path_max(cols, h, here, pick, o), path_max(cols, h, pick, place, o))
    carry = max(hi + hang + CLEAR_MM, release, tip + 20.0)
    notes = [f"grip #{o.id} at ({g.x * 1000:.0f}, {g.y * 1000:.0f}) mm, jaws at {g.yaw:.0f} deg, fingertips {tip:.1f} mm up "
             f"(its top {o.top_mm:.0f} mm)", f"it hangs {hang:.1f} mm below the fingertips",
             f"highest matter on the way {hi:.0f} mm: carry with the fingertips at {carry:.0f} mm"]
    if dest is not None:
        notes.append(f"release over #{dest.id}'s opening at ({place[0] * 1000:.0f}, {place[1] * 1000:.0f}) mm, "
                     f"fingertips {release:.0f} mm up (rim {dest.rim_mm:.0f} + hang {hang:.0f} + {CLEAR_MM:.0f} clear)")
    else:
        notes.append(f"set down at ({place[0] * 1000:.0f}, {place[1] * 1000:.0f}) mm, fingertips {release:.1f} mm up")
    return Plan(True, "ok", g, tip, hang, carry, release, place, hi, notes)


def rows(plan: Plan, cur_tcp) -> list:
    """The fingertip rows (UF0) for a plan: lab/primitives.pick_and_place with the solved heights."""
    from lab import primitives as P
    drop = plan.release_mm - plan.hang_mm  # the object's bottom above the table when let go
    r, _ = P.pick_and_place(list(cur_tcp), (plan.grip.x, plan.grip.y), plan.grip_mm, plan.grip.yaw, tuple(plan.place_xy),
                            drop_mm=drop, safe_mm=plan.carry_mm)
    return r
