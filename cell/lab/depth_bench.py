"""Score a depth result against the truth: the sim's objects, or (on the real cell) heights measured with a ruler.

    res = bench(xyz, objects)         xyz: voxel centres with evidence for something (m, base frame)
                                      objects: {name: trimesh in the base frame}
    sheet(xyz, rgb, objects, path)    a top-down picture: measured height map | true height map, side by side

Per object: its true top (highest vertex above the table) and the measured top (the highest 2 mm layer of voxels over
its footprint holding at least a quarter of the fullest layer's count, so a stray voxel above doesn't count); seen = any voxel over it. False matter: voxels more
than FALSE_MM from every object surface and higher than FALSE_MM above the table (there's nothing there).
"""
from __future__ import annotations

import cv2
import numpy as np
import trimesh

from calibration.geometry import SURFACE_Z

FALSE_MM = 4.0
FOOT_MM = 3.0  # an object's footprint, grown by this


class Poly:
    """A convex polygon in the xy plane; contains_points like matplotlib's Path (ray casting, vectorised)."""

    def __init__(self, v):
        self.v = np.asarray(v, float)

    def contains_points(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        x, y = p[:, 0:1], p[:, 1:2]
        a, b = self.v, np.roll(self.v, -1, 0)
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-12) + a[:, 0])
        return cross.sum(1) % 2 == 1


def footprint(mesh: trimesh.Trimesh, grow_mm: float = FOOT_MM) -> Poly:
    xy = mesh.vertices[:, :2]
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(float) if len(xy) > 3 else xy
    c = hull.mean(0)
    d = hull - c
    hull = c + d * (1 + grow_mm / 1000 / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6))
    return Poly(hull)


def top(h_mm: np.ndarray, share: float = 0.25) -> float:
    """The top of a column of voxels: the highest voxel layer holding at least `share` of the fullest layer's count
    (a stray voxel above doesn't make it taller); that layer's centre height (the surface is within +-1 mm of it)."""
    vals, cnt = np.unique(np.round(h_mm, 1), return_counts=True)
    return float(vals[cnt >= share * cnt.max()].max())


def bench(xyz: np.ndarray, objects: dict) -> dict:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    h = (xyz[:, 2] - SURFACE_Z) * 1000
    per = {}
    for name, m in objects.items():
        inside = footprint(m).contains_points(xyz[:, :2]) & (h > 2.0)
        true_top = (m.vertices[:, 2].max() - SURFACE_Z) * 1000
        meas = top(h[inside]) if inside.any() else None
        per[name] = {"true_top_mm": round(true_top, 1), "measured_top_mm": None if meas is None else round(meas, 1),
                     "error_mm": None if meas is None else round(meas - true_top, 1), "voxels": int(inside.sum())}
    if len(xyz):
        from scipy.spatial import cKDTree  # distance to the surfaces, through dense samples of them (< 0.5 mm apart)
        surf = np.concatenate([m.sample(max(20000, int(m.area / 2e-7))) for m in objects.values()])
        d, _ = cKDTree(surf).query(xyz)
        false = (d * 1000 > FALSE_MM) & (h > FALSE_MM)
    else:
        false = np.zeros(0, bool)
    errs = [abs(v["error_mm"]) for v in per.values() if v["error_mm"] is not None]
    return {"objects": per, "seen": sum(v["voxels"] > 0 for v in per.values()), "of": len(per),
            "median_abs_error_mm": round(float(np.median(errs)), 1) if errs else None,
            "max_abs_error_mm": round(float(np.max(errs)), 1) if errs else None,
            "voxels": len(xyz), "false_matter": int(false.sum()),
            "false_fraction": round(float(false.mean()), 3) if len(xyz) else None}


def height_map(xyz, region, px_mm: float = 2.0):
    (x0, x1), (y0, y1) = region
    W, H = int((y1 - y0) * 1000 / px_mm), int((x1 - x0) * 1000 / px_mm)
    img = np.full((H, W), np.nan)
    for p in np.asarray(xyz).reshape(-1, 3):
        r, c = int((x1 - p[0]) * 1000 / px_mm), int((y1 - p[1]) * 1000 / px_mm)
        if 0 <= r < H and 0 <= c < W:
            z = (p[2] - SURFACE_Z) * 1000
            img[r, c] = z if np.isnan(img[r, c]) else max(img[r, c], z)
    return img


def sheet(xyz, objects: dict, path: str, region=((0.22, 0.56), (-0.32, 0.10)), top_mm: float = 50.0) -> None:
    """Measured | true height maps, top-down (robot's x up the page), coloured by height, white = nothing seen."""
    meas = height_map(xyz, region)
    pts = np.concatenate([m.sample(20000) for m in objects.values()]) if objects else np.zeros((0, 3))
    true = height_map(pts, region)
    tiles = []
    for img, label in ((meas, "measured"), (true, "true")):
        v = np.clip(np.nan_to_num(img, nan=0) / top_mm, 0, 1)
        col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[np.isnan(img)] = 255
        col = cv2.resize(col, (col.shape[1] * 3, col.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(col, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        tiles.append(col)
    cv2.imwrite(path, np.hstack([tiles[0], np.full((tiles[0].shape[0], 8, 3), 80, np.uint8), tiles[1]]))


def sim_objects(trial: str, seed: int = 0, api: str = "http://127.0.0.1:8765/api/v1/") -> dict:
    """The Isaac cell's objects now, as meshes in the base frame."""
    import requests
    from lab.trial import build, load_trial
    sim = build(load_trial(trial), seed)
    truth = requests.get(api + "sim/truth", timeout=30).json()["poses"]
    out = {}
    for i in sim.instances:
        v = truth.get(str(i.id))
        if v is None:
            continue
        T = trimesh.transformations.quaternion_matrix(v[3:7])
        T[:3, 3] = v[:3]
        out[f"{i.color} {i.kind} #{i.id}"] = sim.local[i.id].copy().apply_transform(T)
    return out
