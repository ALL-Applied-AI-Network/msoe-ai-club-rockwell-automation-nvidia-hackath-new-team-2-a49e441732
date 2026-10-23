"""Next best view: where should the wrist camera look next to see what it hasn't seen yet?

The robot's map is a 1 mm height map; cells never seen are NaN. The ones that matter are unseen cells near the things
it cares about (the hidden sides of a blob, where the fingers would go). For each candidate camera pose, trace a ray
from the camera to each of those cells through the current height map: a cell would be revealed if it falls inside the
image, within the depth range, and nothing already known stands in the way. Pick the view that reveals the most per
second of arm travel; stop when the best one would reveal almost nothing.

Candidate poses: the fingertip 70 mm above the table with the fingers straight down, the wrist turned to 0/90/180/270
degrees (the camera sits ~100 mm in front of the fingertip and swings round with the wrist), the camera placed so the
target appears at different spots in the image (straight above it, or off to one side for a slanted look at a side).
The camera pose is computed directly from the arm geometry (no IK).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from calibration.geometry import SURFACE_Z, load
from lab import primitives as P
from lab.fusion import CELL, X_RANGE, Y_RANGE, height_map

TIP_ABOVE_TABLE = 0.070
NEAR = 0.015          # m around a target that counts as "near"
MIN_GAIN = 15         # cells (mm^2): below this a look isn't worth it
RANGE = (0.12, 0.6)   # m: depth the wrist camera can measure (roughly)


@dataclass
class View:
    tip_xy: np.ndarray    # fingertip, base frame m
    yaw: float            # degrees (tcp r)
    cam: np.ndarray       # 4x4 camera pose in the base frame
    gain: int             # unseen cells it would reveal
    seconds: float        # travel + hold
    note: str


def camera_pose(tip_xy, yaw_deg: float, cam: dict | None = None) -> np.ndarray:
    """The wrist depth camera's pose for a fingertip at tip_xy (TIP_ABOVE_TABLE up), fingers down, wrist turned yaw_deg.
    Same convention as the sim: flange x points down the tool, flange y = (cos(yaw + 90), sin(yaw + 90), 0)."""
    cam = cam or load()["wrist_depth"]
    a = np.radians(yaw_deg)
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0.0])
    F = np.eye(4)
    F[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    F[:3, 3] = [tip_xy[0], tip_xy[1], SURFACE_Z + TIP_ABOVE_TABLE + 0.293]
    return F @ cam["T"]


def unseen_near(hm: np.ndarray, footprints: list[np.ndarray]) -> np.ndarray:
    """Flat indices of unseen cells within NEAR of any of the given footprints (flat cell indices)."""
    mask = np.zeros(hm.size, bool)
    for fp in footprints:
        mask[fp] = True
    mask = ndimage.binary_dilation(mask.reshape(hm.shape), iterations=int(NEAR / CELL)).ravel()
    return np.flatnonzero(mask & np.isnan(hm.ravel()))


def visible(hm: np.ndarray, cells: np.ndarray, T_cam: np.ndarray, cam: dict) -> np.ndarray:
    """Which of these cells (flat indices) a camera at T_cam would see, judged against the known heights."""
    if len(cells) == 0:
        return np.zeros(0, bool)
    nx, ny = hm.shape
    ix, iy = np.divmod(cells, ny)
    known = np.nan_to_num(hm, nan=0.0)
    z = SURFACE_Z + np.take(known.ravel(), cells) / 1000 + 0.001  # unseen: assume table height, 1 mm up
    pts = np.column_stack([X_RANGE[0] + (ix + 0.5) * CELL, Y_RANGE[0] + (iy + 0.5) * CELL, z])
    Ti = np.linalg.inv(T_cam)
    pc = pts @ Ti[:3, :3].T + Ti[:3, 3]
    ok = (pc[:, 2] > RANGE[0]) & (pc[:, 2] < RANGE[1])
    u = cam["fx"] * pc[:, 0] / np.maximum(pc[:, 2], 1e-6) + cam["cx"]
    v = cam["fy"] * pc[:, 1] / np.maximum(pc[:, 2], 1e-6) + cam["cy"]
    w, h = cam["size"]
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    # walk each ray from the cell towards the camera; blocked if a known surface rises above it on the way
    o = T_cam[:3, 3]
    s = np.linspace(0.02, 0.9, 40)[:, None, None]
    ray = pts[None] + s * (o - pts)[None]  # (steps, N, 3)
    rx = ((ray[..., 0] - X_RANGE[0]) / CELL).astype(int)
    ry = ((ray[..., 1] - Y_RANGE[0]) / CELL).astype(int)
    inside = (rx >= 0) & (rx < nx) & (ry >= 0) & (ry < ny)
    hgt = np.where(inside, known[np.clip(rx, 0, nx - 1), np.clip(ry, 0, ny - 1)], 0.0)
    blocked = ((SURFACE_Z + hgt / 1000) > ray[..., 2] + 0.0005).any(axis=0)
    return ok & ~blocked


def candidates(target_xy: np.ndarray, cam: dict) -> list[tuple[np.ndarray, float, str]]:
    """(fingertip xy, yaw, description) for views around a target, all reachable."""
    out = []
    for yaw in (0.0, 90.0, 180.0, 270.0):
        T0 = camera_pose(np.zeros(2), yaw, cam)
        off = T0[:2, 3]  # camera position relative to the fingertip for this wrist angle
        for d, label in ((0.0, "straight above"), (0.06, "from the side"), (0.11, "low from the side")):
            for phi in (np.arange(8) * 45.0 if d else [0.0]):
                c = target_xy + d * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
                tip = c - off
                if P.reachable(*tip):
                    side = "" if not d else f" ({['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'][int(phi // 45)]} of it)"
                    out.append((tip, yaw, f"{label}{side}, wrist {yaw:.0f}°"))
    return out


def best_view(points: np.ndarray, footprints: list[np.ndarray], target_xy: np.ndarray, cur_tcp,
              skip: list[tuple[np.ndarray, float]] = ()) -> tuple[View | None, int]:
    """The view revealing the most unseen cells near the footprints, per second; (None, n_unseen) if not worth a look.
    skip: (tip, yaw) of views that already disappointed (revealed much less than predicted)."""
    cam = load()["wrist_depth"]
    hm = height_map(points)
    cells = unseen_near(hm, footprints)
    if len(cells) < MIN_GAIN:
        return None, len(cells)
    cur = np.asarray(cur_tcp[:3], float) / 1000
    best, best_rate = None, 0.0
    for tip, yaw, note in candidates(target_xy, cam):
        if any(np.hypot(*(tip - t)) < 0.005 and yaw == y for t, y in skip):
            continue
        T = camera_pose(tip, yaw, cam)
        gain = int(visible(hm, cells, T, cam).sum())
        secs = np.hypot(*(tip - cur[:2])) / (P.TRAVEL / 1000) + P.HOLD * P.DT
        rate = gain / secs
        if gain >= MIN_GAIN and rate > best_rate:
            best, best_rate = View(tip, yaw, T, gain, secs, note), rate
    return best, len(cells)
