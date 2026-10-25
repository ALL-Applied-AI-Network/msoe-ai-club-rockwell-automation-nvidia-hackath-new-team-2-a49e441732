"""Where can the fingers grip? An analytic search on the fused depth, no learning.

For a target blob, try every jaw angle (yaws) and grip height (levels). At each angle the depth points are rotated so the
jaws close along the rows of a 1 mm grid; then, for every grid cell as the grip centre:
  - jaw gap (12 x 12 mm between the open fingers) must hold some of the target above the grip height,
  - both finger slots (each finger is 12 mm deep along the closing direction + 1 mm margin, 6 mm thick across it,
    just outside the gap; lab/gripper.py) must be clear all the way down to the
    grip height, so the fingers can come straight down beside the material without landing on anything,
  - unseen cells next to seen ones take the lowest nearby height (most are shadows: table hidden behind an object
    from the camera's point of view); cells with nothing seen within 4 mm count as blocked.
  - the fingertips must close below the middle of what's between them: with these hooked fingers a small part is held
    by the lips closing under its widest point (caged in the pocket); lips at or above its middle squeeze it out.
Valid grips are ranked by how much target material sits between the jaws, how centred it is between them, and how close
they are to the middle of the part (a part gripped at one end tips and slides out). Box sums on integral images: fast.
For boxy parts (cubes, bricks) the jaws should close square onto two faces, not across the corners: a 10 mm cube is
14 mm corner to corner, so it fits the 16 mm hook opening diagonally, and held by two edges it spins out or sits crooked.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from calibration.geometry import SURFACE_Z

GAP = 12          # mm between the open fingers' flat faces
GAP_TIP = 16      # mm between the hooked tips: a part that sits entirely in the hook zone has this much room
HOOK_ZONE = 14    # mm above the fingertips where the recess (and so the wider opening) ends
FINGER = 12       # mm finger depth along the closing direction
WIDTH = 6         # mm finger thickness across it
MARGIN = 1        # mm clearance beside the material
CROP = 0.07       # m around the target
YAWS = np.arange(0.0, 180.0, 7.5)
LEVELS = (2.5, 3.5, 4.5, 6.0, 8.0, 11.0, 15.0, 20.0)  # tips at least 2.5 mm off the table, like the twin


@dataclass
class Grip:
    x: float              # base frame, m
    y: float
    height_mm: float      # fingertips above the table
    yaw: float            # degrees; jaws close along (cos yaw, sin yaw)
    material: int         # mm^2 of target between the jaws at that height
    score: float


def _box(I: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """For every cell (i, j): sum of I[i+r0 .. i+r1, j+c0 .. j+c1] (inclusive). Outside the grid counts as 0."""
    P = max(abs(r0), abs(r1), abs(c0), abs(c1)) + 1
    Ip = np.pad(I.astype(np.float64), P)
    S = np.pad(Ip.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = I.shape
    i = np.arange(H)[:, None] + P
    j = np.arange(W)[None, :] + P
    return S[i + r1 + 1, j + c1 + 1] - S[i + r0, j + c1 + 1] - S[i + r1 + 1, j + c0] + S[i + r0, j + c0]


def _raster(pts: np.ndarray, n: int) -> np.ndarray:
    """Points (local a, b in m, z base m) -> n x n grid of max height above the table (mm), NaN where unseen."""
    ia = np.floor(pts[:, 0] * 1000 + n / 2).astype(int)
    ib = np.floor(pts[:, 1] * 1000 + n / 2).astype(int)
    ok = (ia >= 0) & (ia < n) & (ib >= 0) & (ib < n)
    H = np.full(n * n, -np.inf)
    np.maximum.at(H, ia[ok] * n + ib[ok], (pts[ok, 2] - SURFACE_Z) * 1000)
    H[np.isinf(H)] = np.nan
    return H.reshape(n, n)


def faces(target_pts: np.ndarray) -> tuple[float, float]:
    """(the part's edge direction in degrees, mod 90; how boxy it is, 0..1: how much of its tightest rectangle it fills)."""
    p = target_pts[(target_pts[:, 2] - SURFACE_Z) * 1000 > LEVELS[0], :2] * 1000
    if len(p) < 10:
        return 0.0, 0.0
    (_, _), (w, h), ang = cv2.minAreaRect(p.astype(np.float32))
    cells = len(np.unique(np.floor(p).astype(int), axis=0))
    return float(ang % 90), float(np.clip(cells / max(w * h, 1.0), 0, 1))


def find(scene_pts: np.ndarray, target_pts: np.ndarray, top_k: int = 8, max_height_mm: float | None = None) -> tuple[list[Grip], list[Grip]]:
    """(valid grips best first, a sample of rejected candidates for display). max_height_mm: no grip higher than this
    (the caller may know the part's height better than its points do)."""
    c = np.median(target_pts[:, :2], axis=0)
    edge, boxy = faces(target_pts)
    near = scene_pts[np.max(np.abs(scene_pts[:, :2] - c), axis=1) < CROP]
    n = int(2 * CROP * 1000)
    half_gap, slot_in, slot_out, half_w = GAP // 2, GAP // 2, GAP // 2 + FINGER + MARGIN - 1, WIDTH // 2
    best: list[Grip] = []
    rejected: list[Grip] = []
    for yaw in YAWS:
        a = np.radians(yaw)
        skew = abs((yaw - edge + 45) % 90 - 45)  # degrees off square to the faces (45 = across the corners)
        square = 0.03 * skew * max(0.0, (boxy - 0.6) / 0.4)  # only matters for boxy parts (a jack has no faces)
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # world -> (a along the jaws, b across)
        loc = lambda p: np.column_stack([(p[:, :2] - c) @ R.T, p[:, 2]])
        H = _raster(loc(near), n)
        T = ~np.isnan(_raster(loc(target_pts), n))
        # shadows: an unseen cell takes the lowest seen height within 4 mm; still unseen -> blocked
        Hf = np.where(np.isnan(H), ndimage.minimum_filter(np.nan_to_num(H, nan=999.0), size=9), H)
        for L in LEVELS:
            if max_height_mm is not None and L > max(max_height_mm, LEVELS[0]):
                continue
            occ = Hf > L
            mat = occ & T
            top = ndimage.maximum_filter(np.where(T, Hf, 0), size=GAP)
            in_hooks = top <= L + HOOK_ZONE  # the whole part sits between the hooks: 16 mm of room, not 12
            jaw12 = _box(mat, -half_gap, half_gap - 1, -half_w, half_w - 1)
            jaw16 = _box(mat, -GAP_TIP // 2, GAP_TIP // 2 - 1, -half_w, half_w - 1)
            jaw = np.where(in_hooks, jaw16, jaw12)
            if jaw.max() < 6:
                continue
            s_in, s_out = GAP_TIP // 2, GAP_TIP // 2 + FINGER + MARGIN - 1
            slots12 = (_box(occ, -slot_out - 1, -slot_in - 1, -half_w, half_w - 1) + _box(occ, slot_in, slot_out, -half_w, half_w - 1))
            slots16 = (_box(occ, -s_out - 1, -s_in - 1, -half_w, half_w - 1) + _box(occ, s_in, s_out, -half_w, half_w - 1))
            slots = np.where(in_hooks, slots16, slots12)
            other = _box(occ & ~T, -half_gap, half_gap - 1, -half_w, half_w - 1)
            rows = np.arange(n, dtype=float)[:, None] * np.ones((1, n))
            moment = np.where(in_hooks, _box(mat * rows, -GAP_TIP // 2, GAP_TIP // 2 - 1, -half_w, half_w - 1),
                              _box(mat * rows, -half_gap, half_gap - 1, -half_w, half_w - 1))
            off = np.abs(moment / np.maximum(jaw, 1) - rows)
            below_middle = L <= np.maximum(0.5 * top - 1.0, LEVELS[0])  # lips at least 1 mm under the part's widest point
            depth_pref = 1 - np.clip(np.abs(L - 0.4 * top) / np.maximum(top, 1), 0, 1)
            ii, jj = np.mgrid[0:n, 0:n]
            from_middle = np.hypot(ii + 0.5 - n / 2, jj + 0.5 - n / 2)  # mm from the part's centre
            ok = (jaw >= 6) & (slots == 0) & (off < 2.5) & below_middle
            score = jaw / (GAP * WIDTH) - 0.15 * off + 0.5 * depth_pref - 0.01 * other - 0.04 * from_middle - square
            for valid, sel in ((True, ok), (False, (jaw >= 6) & ~ok)):
                idx = np.flatnonzero(sel)
                if len(idx) == 0:
                    continue
                pick = idx[np.argsort(-score.ravel()[idx])[: (3 if valid else 1)]]
                for k in pick:
                    i, j = divmod(int(k), n)
                    la, lb = (i + 0.5 - n / 2) / 1000, (j + 0.5 - n / 2) / 1000
                    xy = c + R.T @ np.array([la, lb])
                    g = Grip(float(xy[0]), float(xy[1]), L, float(yaw), int(jaw.ravel()[k]), float(score.ravel()[k]))
                    (best if valid else rejected).append(g)
    best.sort(key=lambda g: -g.score)
    out: list[Grip] = []
    for g in best:  # keep distinct grips (not the same spot at a neighbouring angle or height)
        if all(np.hypot(g.x - h.x, g.y - h.y) > 0.004 or abs((g.yaw - h.yaw + 90) % 180 - 90) > 20 for h in out):
            out.append(g)
        if len(out) >= top_k:
            break
    return out, rejected[:12]
