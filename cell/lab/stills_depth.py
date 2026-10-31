"""Heights from still wrist-depth shots, each measured from the table that shot sees.

    uv run python -m lab.stills_depth runs/<time>_datarun_real        (any run with stops/NNN.npz: depth_wrist, joints)

Why it works: a depth shot gives two things mixed together: where things are relative to each other (the camera
measures that well) and where the camera was (worked out from the joints, the picture's timing and our calibration,
all of which carry errors). Measured in the robot's frame, every error in the second becomes a fake height (the bare
table read 5-20 mm "tall"). Measured from the table in the same shot, the camera's error is shared by the table and the
object and cancels; a still shot has no timing error at all. So: per shot, the table plane (RANSAC, then a least-squares
fit to its points), every point's height above it; the shots placed in the robot's frame only to line them up, and
combined per 3 mm cell by the median (a ghost in one shot doesn't survive the others). Things are the connected cells
standing up more than ABOVE_MM; their top is the 90th percentile of their cells. Writes <run>/stills_heights.jpg
(the height map, things labelled) and <run>/stills_things.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import depth_to_base, load

CELL = 0.003
ABOVE_MM = 6.0
PLANE_MM = 4.0


def table_relative(p: np.ndarray, rng) -> np.ndarray | None:
    """Each point's height (mm) above the dominant near-horizontal plane in this shot (the table)."""
    sub = p[rng.choice(len(p), min(4000, len(p)), replace=False)]
    best = None
    for _ in range(150):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(n[2]) < 0.9:
            continue
        k = int((np.abs((sub - s[0]) @ n) < PLANE_MM / 1000).sum())
        if best is None or k > best[0]:
            best = (k, n, s[0])
    if best is None:
        return None
    _, n, o = best
    inl = np.abs((p - o) @ n) < PLANE_MM / 1000
    c = p[inl].mean(0)
    _, _, vt = np.linalg.svd(p[inl][::5] - c, full_matrices=False)  # (full: an N x N matrix, GBs)
    n = vt[2] * np.sign(vt[2][2])
    return ((p - c) @ n) * 1000


def height_map(shots, rng=None):
    """shots: [(depth_wrist uint16 mm, joints)] -> (grid of heights mm, NaN where unseen; origin x0, y0 (m))."""
    rng = rng or np.random.default_rng(0)
    cams = load()
    P, H = [], []
    for d, j in shots:
        p = depth_to_base(cams["wrist_depth"], d, stride=2, joints=j)
        h = table_relative(p, rng) if len(p) > 500 else None
        if h is not None:
            P.append(p)
            H.append(h)
    P, H = np.concatenate(P), np.concatenate(H)
    m = (H > -20) & (H < 150)
    P, H = P[m], H[m]
    x0, y0 = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - (x0, y0)) / CELL).astype(int)
    W, Hh = ij.max(0) + 1
    key = ij[:, 0] * Hh + ij[:, 1]
    o = np.argsort(key)
    key, h = key[o], H[o]
    u, st, cn = np.unique(key, return_index=True, return_counts=True)
    g = np.full(W * Hh, np.nan)
    g[u] = [np.median(h[s:s + c]) for s, c in zip(st, cn)]
    return g.reshape(W, Hh), (x0, y0)


def things(grid, origin, min_cells: int = 4):
    mask = (np.nan_to_num(grid, nan=-99) > ABOVE_MM).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_cells:
            continue
        v = grid[lab == k]
        c = (np.array(origin) + (cen[k][::-1] + 0.5) * CELL) * 1000
        out.append({"x_mm": round(float(c[0])), "y_mm": round(float(c[1])), "area_mm2": int(st[k, 4] * 9e-6 * 1e6),
                    "top_mm": round(float(np.percentile(v, 90)), 1),
                    "size_mm": [round(float(st[k, cv2.CC_STAT_HEIGHT] * CELL * 1000)),
                                round(float(st[k, cv2.CC_STAT_WIDTH] * CELL * 1000))]})
    return sorted(out, key=lambda t: -t["area_mm2"])


def picture(grid, found, title: str) -> np.ndarray:
    v = np.clip(np.nan_to_num(grid, nan=-99) / 60.0, 0, 1)
    im = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[np.isnan(grid)] = (40, 40, 40)
    im[(~np.isnan(grid)) & (grid < 4)] = (230, 230, 230)
    s = 2
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    im = np.ascontiguousarray(np.rot90(im, 1))
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(im, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main() -> None:
    run = Path(sys.argv[1])
    shots = []
    for f in sorted(glob.glob(str(run / "stops" / "*.npz"))):
        z = np.load(f)
        if "depth_wrist" in z.files:
            shots.append((z["depth_wrist"], z["joints"]))
    grid, origin = height_map(shots)
    found = things(grid, origin)
    table = grid[np.abs(np.nan_to_num(grid, nan=99)) < 4]
    res = {"shots": len(shots), "table_mm": {"median": round(float(np.median(table)), 2), "spread": round(float(np.std(table)), 2)},
           "things": found[:20]}
    (run / "stills_things.json").write_text(json.dumps(res, indent=1))
    cv2.imwrite(str(run / "stills_heights.jpg"), picture(grid, found, f"{run.name}: {len(shots)} still shots, heights from each shot's table"))
    print(f"{run.name}: {len(shots)} still shots; table {res['table_mm']['median']:+.1f} mm (spread {res['table_mm']['spread']:.1f})")
    for t in found[:10]:
        print(f"   thing at ({t['x_mm']:4d}, {t['y_mm']:4d}) mm, about {t['size_mm'][0]} x {t['size_mm'][1]} mm, top {t['top_mm']:5.1f} mm")


if __name__ == "__main__":
    main()
