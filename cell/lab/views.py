"""Where to point the wrist depth camera next: at the gaps (lab.evidence's frontier), from whatever angle sees most of
them, per second of moving. Deterministic: no model, no guessing.

    v = best_view(emap, cams, q_now, tip_now)    # {"tip", "wpr", "q", "sees", "seconds", "target"} or None

1. cluster the frontier voxels (the edges of what's known); aim at the biggest clusters
2. candidate camera poses per target: straight down, and tilted 30 / 45 deg from 8 directions, at a standoff the depth
   camera reads well, rolled four ways; each turned into the tool pose that puts the wrist camera there (the camera is
   bolted to the flange: tool pose = camera pose x a fixed offset)
3. keep the reachable ones (IK from the current joints), fingertips well above the table
4. score: frontier voxels inside the camera's view and range, not hidden behind solid evidence (a ray marched through
   the map), divided by the time the move takes
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose
from calibration.kinematics import HOME, tool_ik, tool_pose, wpr_of
from lab import primitives as P

STANDOFF = (0.23,)       # m from the target to the depth camera (it reads well from ~0.12 m)
RANGE = (0.10, 0.55)          # m: depth the wrist camera reads
TILTS = (0.0, 30.0, 45.0)     # deg from looking straight down
MIN_TIP_MM = 45.0             # fingertips at least this far above the table on a look


def _cam_in_tool(cam: dict) -> np.ndarray:
    """The wrist depth camera's pose in the tool frame (constant: both ride on the flange)."""
    return np.linalg.inv(tool_pose(HOME)) @ flange_pose(HOME) @ cam["T"]


def _look_at(target, direction, standoff, roll) -> np.ndarray:
    """A camera pose (OpenCV axes: z forward) at standoff from target, looking along direction, turned by roll."""
    z = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = target - z * standoff
    return T


def targets(emap, k: int = 3) -> list[np.ndarray]:
    """Centres of the biggest frontier clusters."""
    from scipy import ndimage
    fr = emap.frontier()
    if not fr.any():
        return []
    lab, n = ndimage.label(ndimage.binary_dilation(fr, iterations=2))
    sizes = ndimage.sum(fr, lab, range(1, n + 1))
    out = []
    for c in np.argsort(sizes)[::-1][:k]:
        idx = np.argwhere((lab == c + 1) & fr)
        out.append(emap.centers(*idx.mean(0)))
    return out


def visible(emap, pts: np.ndarray, Tc: np.ndarray, cam: dict) -> np.ndarray:
    """Which of pts (base frame) a depth camera at Tc would see: in its view and range, not behind solid voxels."""
    R, t = Tc[:3, :3], Tc[:3, 3]
    pc = (pts - t) @ R
    z = pc[:, 2]
    ok = (z > RANGE[0]) & (z < RANGE[1])
    W, H = cam["size"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pc[:, 0] / z * cam["fx"] + cam["cx"]
        v = pc[:, 1] / z * cam["fy"] + cam["cy"]
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return ok
    solid = emap.solid()
    p = pts[idx]
    dist = np.linalg.norm(p - t, axis=1)
    steps = int(np.ceil(dist.max() / emap.v))
    f = np.linspace(0, 1, steps)[:, None, None]
    ray = t + (p - t)[None] * f  # steps x n x 3
    ijk, inside = emap.index(ray.reshape(-1, 3))
    hit = np.zeros(len(ijk), bool)
    hit[inside] = solid[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    hit = hit.reshape(steps, len(idx))
    last = (dist[None] - f[:, :, 0] * dist[None]) > 2.5 * emap.v  # ignore the voxels right at the target
    blocked = (hit & last).any(0)
    ok[idx[blocked]] = False
    return ok


def best_view(emap, cams: dict, q_now, tip_now, sample: int = 2500) -> dict | None:
    cam = cams["wrist_depth"]
    T_ct = _cam_in_tool(cam)
    fr = emap.frontier_points()
    if len(fr) == 0:
        return None
    if len(fr) > sample:
        fr = fr[np.random.default_rng(0).choice(len(fr), sample, replace=False)]
    best = None
    for target in targets(emap):
        for tilt in TILTS:
            for az in ([0.0] if tilt == 0 else np.arange(0, 360, 45)):
                a, e = np.radians(az), np.radians(tilt)
                direction = np.array([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), -np.cos(e)])
                for standoff in STANDOFF:
                    for roll in np.radians([0, 90, 180, 270]):
                        Tc = _look_at(target, direction, standoff, roll)
                        Tt = Tc @ np.linalg.inv(T_ct)
                        tip = Tt[:3, 3]
                        if (tip[2] - SURFACE_Z) * 1000 < MIN_TIP_MM or not P.reachable(*tip[:2], margin_mm=20):
                            continue
                        vis = visible(emap, fr, Tc, cam)
                        sees = int(vis.sum())
                        if sees < 20 or (best is not None and sees <= best["sees"] * 0.5):
                            continue
                        wpr = wpr_of(Tt[:3, :3])
                        try:
                            q = tool_ik(tip, wpr, q_now)
                        except ValueError:
                            continue
                        seconds = np.linalg.norm(tip - tip_now) / (P.TRAVEL / 1000) + np.abs(q - q_now).max() / 60.0 + 1.5
                        value = sees / seconds
                        if best is None or value > best["value"]:
                            best = {"tip": tip, "wpr": wpr, "q": q, "sees": sees, "seconds": float(seconds), "value": value,
                                    "target": target, "tilt": tilt, "cam": Tc, "expected": fr[vis]}
                        break  # one roll that works is enough
    return best
