"""Rigid objects from the evidence grid: which columns of space hold matter above the table, and how they group.

    objs = find(cols, xyz, rgb)      cols: the depth service's /columns; xyz, rgb: its /grid (voxels with evidence)

The grid sees edges, not faces (a plain face has nothing to match): a cube comes out as a ring of columns, a box as
two rings (its rim's outer and inner edge). So, top-down:
    matter     columns whose highest something voxel is ABOVE_MM over the table
    clusters   matter joined across gaps of up to JOIN_MM (a closing), each big enough (MIN_MM2) to be a thing
    filled     each cluster with its holes filled: a solid object's footprint
    container  a hole wider than HOLE_MM where rays have reached at least FREE_MM below the rim (space known empty
               inside): open, something can be put in it; its opening is the hole
Per object: the footprint (convex hull, base frame), its outline at the top (where it's gripped), its top (the highest well-filled voxel layer, lab/depth_bench.top),
its size and angle (the smallest rectangle round it), its colour (mean of its voxels'), and for a container the rim
height and the opening. Nothing about what kind of thing it is: that's the LLM's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z
from lab.depth_bench import Poly, top

ABOVE_MM = 3.0
JOIN_MM = 4.0
MIN_MM2 = 40.0
MIN_VOX = 12  # fewer voxels than this over a footprint: a speck, not a thing
TOP_BAND_MM = 3.0
HOLE_MM = 20.0
FREE_MM = 8.0


@dataclass
class Obj:
    id: int
    footprint: np.ndarray  # (n, 2) convex hull, base frame (m)
    top_footprint: np.ndarray  # (n, 2) hull of its voxels within TOP_BAND_MM of its top: its outline where it's
    # gripped (edges smear a few mm sideways lower down: a 10 mm cube's footprint comes out ~17 mm, its top ~11)
    centre: np.ndarray  # (2,) m
    top_mm: float
    size_mm: tuple  # length, width of the smallest rectangle round it
    angle_deg: float  # that rectangle's angle
    rgb: tuple
    voxels: int
    container: bool = False
    rim_mm: float | None = None
    opening: np.ndarray | None = None  # (n, 2) the hole, base frame (m)
    free_inside: float = 0.0  # share of the hole's columns known empty below the rim
    cells: np.ndarray = field(default=None, repr=False)  # (k, 2) its filled columns (row, col) in the grid


def find(cols: dict, xyz, rgb=None) -> list[Obj]:
    vox = cols["vox"]
    X, Y = cols["shape"]
    z0 = cols["z0"]
    top_i = np.asarray(cols["top"], int)
    low_i = np.asarray(cols["free_low"], int)
    h_top = np.where(top_i >= 0, z0 + (top_i + 0.5) * vox - SURFACE_Z, -1) * 1000
    h_low = np.where(low_i >= 0, z0 + (low_i + 0.5) * vox - SURFACE_Z, np.inf) * 1000
    matter = (h_top >= ABOVE_MM).astype(np.uint8)
    k = int(np.ceil(JOIN_MM / (vox * 1000))) * 2 + 1
    closed = cv2.morphologyEx(matter, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    rgb = np.zeros((len(xyz), 3)) if rgb is None else np.asarray(rgb, float).reshape(-1, 3)
    h_vox = (xyz[:, 2] - SURFACE_Z) * 1000

    def to_xy(rc):  # grid (row = x index, col = y index) -> base frame m (column centres)
        return np.stack([cols["x0"] + (rc[:, 0] + 0.5) * vox, cols["y0"] + (rc[:, 1] + 0.5) * vox], 1)

    out = []
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        if comp.sum() * (vox * 1000) ** 2 < MIN_MM2:
            continue
        filled = comp.copy()
        pad = np.pad(filled, 1)
        ff = pad.copy()
        cv2.floodFill(ff, None, (0, 0), 1)  # everything reachable from outside
        filled = (pad | (1 - ff))[1:-1, 1:-1]
        hole = filled & (1 - comp)
        cells = np.argwhere(filled)
        xy = to_xy(cells)
        hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(float)
        inside = Poly(hull).contains_points(xyz[:, :2]) & (h_vox >= ABOVE_MM)
        if inside.sum() < MIN_VOX:
            continue
        t = top(h_vox[inside])
        band = inside & (h_vox >= t - TOP_BAND_MM)
        top_fp = (cv2.convexHull(xyz[band, :2].astype(np.float32)).reshape(-1, 2).astype(float)
                  if band.sum() >= 4 else hull)
        (cx, cy), (w, l), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
        o = Obj(id=len(out) + 1, footprint=hull, top_footprint=top_fp, centre=xy.mean(0), top_mm=round(t, 1),
                size_mm=(round(max(w, l), 1), round(min(w, l), 1)), angle_deg=round(float(ang), 1),
                rgb=tuple(int(c) for c in (rgb[inside].mean(0) if inside.any() else (128, 128, 128))),
                voxels=int(inside.sum()), cells=cells)
        hn, hl, hs, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
        if hn > 1:
            j = 1 + int(np.argmax(hs[1:, cv2.CC_STAT_AREA]))
            hc = np.argwhere(hl == j)
            span = (hc.max(0) - hc.min(0) + 1) * vox * 1000
            if span.min() >= HOLE_MM:
                below = h_low[hc[:, 0], hc[:, 1]] <= t - FREE_MM
                o.free_inside = round(float(below.mean()), 2)
                # a plain inside (a white box) gives nothing to match, so no rays are known to reach down: then a ring
                # with nothing seen in its middle near the rim is open too (a big plain-topped block would fool this)
                empty = h_top[hc[:, 0], hc[:, 1]] < t - FREE_MM
                if o.free_inside >= 0.2 or empty.mean() >= 0.9:
                    o.container, o.rim_mm = True, o.top_mm
                    o.opening = cv2.convexHull(to_xy(hc).astype(np.float32)).reshape(-1, 2).astype(float)
        out.append(o)
    return out


def match(objs: list[Obj], truth: dict, near_mm: float = 15.0) -> dict:
    """Each true object (name -> trimesh, base frame) against the found ones: the found object whose footprint holds
    its centre (or is nearest, within near_mm). Reports misses, merges (one found object for several true ones) and
    extras (found objects matching nothing)."""
    rows, used = {}, {}
    for name, m in truth.items():
        c = m.vertices[:, :2].mean(0)
        hit = [o for o in objs if Poly(o.footprint).contains_points(c[None])[0]]
        if not hit:
            d = [np.linalg.norm(o.centre - c) * 1000 for o in objs]
            hit = [objs[int(np.argmin(d))]] if d and min(d) <= near_mm else []
        o = hit[0] if hit else None
        true_top = (m.vertices[:, 2].max() - SURFACE_Z) * 1000
        rows[name] = {"found": o.id if o else None, "true_top_mm": round(true_top, 1),
                      "top_mm": o.top_mm if o else None, "error_mm": round(o.top_mm - true_top, 1) if o else None,
                      "container": o.container if o else None, "free_inside": o.free_inside if o else None}
        if o:
            used.setdefault(o.id, []).append(name)
    return {"objects": rows, "missed": [n for n, r in rows.items() if r["found"] is None],
            "merged": {k: v for k, v in used.items() if len(v) > 1},
            "extra": [o.id for o in objs if o.id not in used]}
