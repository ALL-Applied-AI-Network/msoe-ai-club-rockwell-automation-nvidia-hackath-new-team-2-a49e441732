"""The robot's own picture of the table, built only from what its cameras saw (never from the truth).

Points are (N, 3) x, y, z in metres, or (N, 6) with r, g, b (0-255) appended once colour() has been applied.

Level 1 (now): fused depth -> height map -> blobs sticking up from the table -> numbered segments.
A segment is the robot's guess at one rigid object. Touching objects merge into one segment until nudging or
closer looks split them; that's expected, and the score below shows it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from calibration.geometry import SURFACE_Z
from lab.fusion import CELL, X_RANGE, Y_RANGE, height_map

MIN_HEIGHT_MM = 2.5   # above this, a cell isn't table (depth noise near the table is about 1-1.5 mm)
MIN_CELLS = 12        # smaller blobs are noise (12 mm^2)
MAX_HEIGHT_MM = 100.0 # taller than this is the arm or gripper, not a part on the table


def colour(points: np.ndarray, obs: dict, cams: dict, cam: str = "wrist") -> np.ndarray:
    """Append the colour each point has in that camera's image (it sees the same view as the wrist depth stream)."""
    from lab.viz import project_to_image
    img = obs["images"][cam]
    h, w = img.shape[:2]
    uv = np.round(project_to_image(cam, cams, points[:, :3], (w, h), joints=obs.get("joints"))).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    rgb = np.full((len(points), 3), 128.0)
    rgb[ok] = img[uv[ok, 1], uv[ok, 0]]
    return np.column_stack([points[:, :3], rgb])


def mean_colour(points: np.ndarray) -> np.ndarray | None:
    """Average colour of a thing's top surfaces (the sides are darker), or None if the points carry no colour."""
    if points.shape[1] < 6:
        return None
    top = points[points[:, 2] >= np.percentile(points[:, 2], 60)]
    return top[:, 3:6].mean(0)


def drop_self(points: np.ndarray, tcp_pose_uf0) -> np.ndarray:
    """Remove what the cameras see of the robot itself: anything too tall to be a part, and a box around the gripper
    (fingers + palm, 30 mm each side of the fingertip, from 5 mm below it to 130 mm above)."""
    keep = (points[:, 2] - SURFACE_Z) * 1000 < MAX_HEIGHT_MM
    if tcp_pose_uf0 is not None:
        tip = np.asarray(tcp_pose_uf0[:3], float) / 1000 + [0, 0, 0.33]
        a = np.radians(tcp_pose_uf0[5])
        d = points[:, :2] - tip[:2]
        local = np.stack([d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a), -d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)], -1)
        dz = points[:, 2] - tip[2]
        keep &= ~((np.abs(local[:, 0]) < 0.03) & (np.abs(local[:, 1]) < 0.03) & (dz > -0.005) & (dz < 0.13))
    return points[keep]


@dataclass
class Segment:
    id: int
    points: np.ndarray            # (N, 3) base-frame metres
    cells: int                    # footprint in mm^2
    height_mm: float
    centroid: np.ndarray


def segment(points: np.ndarray) -> tuple[list[Segment], np.ndarray]:
    """Fused points -> (segments, height map). Ids are the robot's own, numbered by size."""
    hm = height_map(points)
    filled = np.nan_to_num(hm, nan=0.0)
    mask = ndimage.binary_closing(filled > MIN_HEIGHT_MM, iterations=1)
    labels, n = ndimage.label(mask)
    ix = ((points[:, 0] - X_RANGE[0]) / CELL).astype(int)
    iy = ((points[:, 1] - Y_RANGE[0]) / CELL).astype(int)
    inside = (ix >= 0) & (ix < hm.shape[0]) & (iy >= 0) & (iy < hm.shape[1])
    lab_pt = np.zeros(len(points), int)
    lab_pt[inside] = labels[ix[inside], iy[inside]]
    above = (points[:, 2] - SURFACE_Z) * 1000 > MIN_HEIGHT_MM
    segs = []
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    for lab in np.argsort(-sizes) + 1:
        if sizes[lab - 1] < MIN_CELLS:
            continue
        pts = points[(lab_pt == lab) & above]
        if len(pts) < 10:
            continue
        segs.append(Segment(len(segs) + 1, pts, int(sizes[lab - 1]), float(np.nanmax(hm[labels == lab])), pts[:, :3].mean(0)))
    return segs, hm


def score(segs: list[Segment], truth: list[dict]) -> dict:
    """How the robot's segments compare with the true objects.
    found: true objects that sit in some segment; merged: segments holding more than one true object;
    phantom: segments matching nothing; error_mm: centroid error for one-to-one matches."""
    if not truth:
        return {}
    owner = {}
    for o in truth:
        c = o["mesh"].centroid
        d = [np.min(np.linalg.norm(s.points[:, :2] - c[:2], axis=1)) if len(s.points) else 1 for s in segs]
        if d and min(d) < 0.004:
            owner[o["id"]] = segs[int(np.argmin(d))].id
    counts = {s.id: sum(1 for v in owner.values() if v == s.id) for s in segs}
    singles = [s for s in segs if counts[s.id] == 1]
    errs = []
    if singles:
        tc = {o["id"]: o["mesh"].centroid for o in truth}
        pairs = [(s, next(t for t, v in owner.items() if v == s.id)) for s in singles]
        errs = [np.linalg.norm(np.asarray(s.centroid)[:2] - tc[t][:2]) * 1000 for s, t in pairs]
    return {"objects": len(truth), "segments": len(segs), "found": len(owner),
            "separated": sum(1 for s in segs if counts[s.id] == 1), "merged": sum(1 for s in segs if counts[s.id] > 1),
            "phantom": sum(1 for s in segs if counts[s.id] == 0), "error_mm": float(np.mean(errs)) if errs else None}
