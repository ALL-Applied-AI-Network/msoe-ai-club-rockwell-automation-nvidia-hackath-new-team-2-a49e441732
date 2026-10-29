"""Watch the 3D scene get built from a real data run (lab/datarun.py), frame by frame, in Rerun.

    uv run python -m lab.scene3d runs/<time>_datarun_real          -> runs/<...>/scene3d.rrd, then
    uv run rerun runs/<time>_datarun_real/scene3d.rrd

Timeline "frame": every wrist picture, with the camera where the joints put it. Two depth sources, each shown as
"now" (this frame alone) and "all" (everything up to this frame, fused):
    edges/   depth from the colour video itself: each edge pixel of a reference frame is tried at heights above the
             table, projected into nearby frames with their known camera poses, and kept at the height where a
             one-sided patch matches across them (where the match has one clear peak). Every REF_EVERY-th frame.
    orbbec/  the depth camera's own depth, at the stops where the run asked for it.
The fused clouds keep a 2 mm voxel once at least MIN_HITS frames put a point in it: noise and the depth camera's
phantoms change from frame to frame, real edges don't. Heights are coloured blue (table) to red (20 mm up).

The structure-tensor flow (minimum-eigenvector flow over the video volume) can replace the edge sweep: it gives
the same thing, an edge pixel's motion between frames, which the known camera motion turns into depth.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.ndimage import map_coordinates

from calibration.geometry import SURFACE_Z, depth_to_base, flange_pose, load, project

REF_EVERY = 6          # an edge sweep on every 6th frame
H_MM = np.arange(-3.0, 40.01, 0.5)
NEAR = (0.008, 0.09)   # neighbour frames: their camera this far from the reference's (m)
MAX_NBRS = 10
PEAK, MARGIN = 0.7, 0.1  # a clear match: NCC this good, and this far above the best more than 3 mm away
VOX = 0.002
MIN_HITS = 2
REGION = ((0.15, 0.65), (-0.40, 0.40), (-0.004, 0.08))  # where the parts are: x, y, height above the table (m)


def cameras(run: Path):
    raw = json.loads((Path(__file__).resolve().parents[1] / "calibration" / "cameras.json").read_text())
    cams = load()
    w = dict(cams["wrist"])
    sc = 848 / w["size"][0]
    w.update(fx=w["fx"] * sc, size=(848, 480), frame="base")
    return w, np.array(raw["wrist"]["T_flange_cam"]), cams


def rays_to_plane(cam, uv, z):
    w, h = cam["size"]
    z = np.broadcast_to(np.asarray(z, float), (len(uv),))
    xd, yd = (uv[:, 0] - w / 2) / cam["fx"], (uv[:, 1] - h / 2) / cam["fx"]
    x, y = xd, yd
    for _ in range(8):
        d = 1 + cam["k1"] * (x * x + y * y)
        x, y = xd / d, yd / d
    T = cam["T"]
    ray = np.stack([x, y, np.ones_like(x)], -1) @ T[:3, :3].T
    o = T[:3, 3]
    return o + ray * ((z - o[2]) / ray[:, 2])[:, None]


ALONG = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ACROSS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_A, _C = (m.ravel() for m in np.meshgrid(ALONG, ACROSS, indexing="ij"))


def patches(img, uv, nrm):
    """One-sided patches along each pixel's normal (mostly on one side of the edge), bilinear: (n, 30)."""
    tan = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
    xs = uv[:, 0:1] + nrm[:, 0:1] * _A[None] + tan[:, 0:1] * _C[None]
    ys = uv[:, 1:2] + nrm[:, 1:2] * _A[None] + tan[:, 1:2] * _C[None]
    return map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(len(uv), -1)


def ncc(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-6)


def _scores(ref_p, uv, nrm, cam, nbrs, z_mm):
    """Mean NCC over the neighbour frames for each pixel at its own height z_mm (n,); -1 where < 3 frames saw it."""
    P = rays_to_plane(cam, uv, SURFACE_Z + z_mm / 1000)
    tot, n = np.zeros(len(uv)), np.zeros(len(uv))
    for img_k, _, cam_k in nbrs:
        q = project(cam_k, P)
        ok = (q[:, 0] > 8) & (q[:, 0] < 840) & (q[:, 1] > 8) & (q[:, 1] < 472)
        if ok.any():
            tot[ok] += ncc(ref_p[ok], patches(img_k, q[ok], nrm[ok]))
            n[ok] += 1
    return np.where(n >= 3, tot / np.maximum(n, 1), -1.0)


COARSE = np.arange(-3.0, 40.01, 2.0)
FINE = np.arange(-2.0, 2.01, 0.5)


def sweep(ref, nbrs):
    """ref, nbrs: (gray float32, colour BGR, cam). Returns (points base m, height mm, good) for the ref's edge pixels.
    Heights in 2 mm steps, then 0.5 mm around the best; one-sided patches, on whichever side matches better."""
    g, col, cam = ref
    edges = np.zeros(g.shape, np.uint8)
    for ch in cv2.split(cv2.cvtColor(col, cv2.COLOR_BGR2LAB)):
        edges |= cv2.Canny(ch, 10, 30)
    edges[:8] = edges[-8:] = 0
    edges[:, :8] = edges[:, -8:] = 0
    vs, us = np.nonzero(edges)
    if len(us) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, bool)
    uv = np.stack([us, vs], 1).astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[vs, us]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[vs, us]
    nrm0 = -np.stack([gx, gy], 1) / (np.hypot(gx, gy)[:, None] + 1e-6)
    best = None
    for side in (1.0, -1.0):
        nrm = side * nrm0
        ref_p = patches(g, uv, nrm)
        coarse = np.stack([_scores(ref_p, uv, nrm, cam, nbrs, np.full(len(uv), h)) for h in COARSE], 1)
        hc = COARSE[coarse.argmax(1)]
        fine = np.stack([_scores(ref_p, uv, nrm, cam, nbrs, hc + d) for d in FINE], 1)
        h = hc + FINE[fine.argmax(1)]
        peak = fine.max(1)
        far = np.abs(COARSE[None, :] - h[:, None]) > 3.0
        second = np.where(far, coarse, -1).max(1)
        cand = (h, peak, second)
        if best is None:
            best = cand
        else:
            better = peak > best[1]
            best = tuple(np.where(better, c, b) for c, b in zip(cand, best))
    h, peak, second = best
    good = (peak > PEAK) & (peak - second > MARGIN)
    return rays_to_plane(cam, uv, SURFACE_Z + h / 1000), h, good


class Fused:
    """2 mm voxels: how many frames put a point in each; shown once MIN_HITS agree."""

    def __init__(self):
        self.hits: dict = {}

    def add(self, pts: np.ndarray) -> None:
        keys = np.floor(pts / VOX).astype(np.int64)
        for k in {tuple(r) for r in keys}:  # once per frame per voxel: frames agreeing, not pixels
            self.hits[k] = self.hits.get(k, 0) + 1

    def points(self) -> np.ndarray:
        ks = [k for k, n in self.hits.items() if n >= MIN_HITS]
        return (np.array(ks, float) + 0.5) * VOX if ks else np.zeros((0, 3))


def in_region(p: np.ndarray) -> np.ndarray:
    (x0, x1), (y0, y1), (h0, h1) = REGION
    h = p[:, 2] - SURFACE_Z
    return (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (h > h0) & (h < h1)


def colours(p: np.ndarray) -> np.ndarray:
    if len(p) == 0:
        return np.zeros((0, 3), np.uint8)
    h = np.clip((p[:, 2] - SURFACE_Z) * 1000 / 20.0, 0, 1)
    return cv2.applyColorMap((h * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]


def main(run: Path) -> None:
    w, T_fc, cams = cameras(run)
    rows = [json.loads(l) for l in open(run / "frames.jsonl")]
    stops = sorted(run.glob("stops/*.npz"))
    stop_t = {}
    for f in stops:
        z = np.load(f)
        stop_t[float(z["t"])] = f
    rr.init("scene3d", spawn=False)
    rr.save(str(run / "scene3d.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    (x0, x1), (y0, y1), _ = REGION
    rr.log("world/table", rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, SURFACE_Z - 0.001]],
                                    half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, 0.0005]], colors=[[200, 200, 200]]), static=True)
    K = np.array([[w["fx"], 0, 424], [0, w["fx"], 240], [0, 0, 1]])
    frames = []
    for r in rows:
        img = cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg"))
        cam = dict(w, T=flange_pose(r["joints"]) @ T_fc)
        frames.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), img, cam))
    edges_all, orb_all = Fused(), Fused()
    t0 = rows[0]["t"]
    stop_keys = sorted(stop_t)
    used_stops = set()
    report = []
    tic = time.time()
    for k, (r, fr) in enumerate(zip(rows, frames)):
        rr.set_time("frame", sequence=k)
        rr.set_time("seconds", duration=r["t"] - t0)
        T = fr[2]["T"]
        rr.log("world/wrist", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("world/wrist/cam", rr.Pinhole(image_from_camera=K, resolution=[848, 480], image_plane_distance=0.05))
        rr.log("world/wrist/cam", rr.Image(cv2.cvtColor(fr[1], cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        rr.log("info", rr.TextLog(f"frame {k}: {r['phase']}, camera at ({T[0, 3] * 1000:.0f}, {T[1, 3] * 1000:.0f}, "
                                  f"{(T[2, 3] - SURFACE_Z) * 1000:.0f}) mm"))
        # the depth camera, at the stop nearest in time (the full sets were taken right after this frame)
        near = [t for t in stop_keys if abs(t - r["t"]) < 0.6 and t not in used_stops]
        if near:
            used_stops.add(near[0])
            z = np.load(stop_t[near[0]])
            pts = depth_to_base(cams["wrist_depth"], z["depth_wrist"], stride=2, joints=z["joints"])
            pts = pts[in_region(pts)]
            orb_all.add(pts)
            rr.log("world/orbbec/now", rr.Points3D(pts, colors=colours(pts), radii=0.0008))
            fp = orb_all.points()
            rr.log("world/orbbec/all", rr.Points3D(fp, colors=colours(fp), radii=0.001))
        if k % REF_EVERY == 0:
            dist = [np.linalg.norm(f[2]["T"][:3, 3] - T[:3, 3]) for f in frames]
            cand = [i for i, d in enumerate(dist) if NEAR[0] <= d <= NEAR[1]]
            # spread the neighbours: the farthest-point choice over their camera positions
            pick = []
            for i in sorted(cand, key=lambda i: -dist[i]):
                if all(np.linalg.norm(frames[i][2]["T"][:3, 3] - frames[j][2]["T"][:3, 3]) > 0.006 for j in pick):
                    pick.append(i)
                if len(pick) >= MAX_NBRS:
                    break
            if len(pick) >= 3:
                P, h, good = sweep(fr, [frames[i] for i in pick])
                P = P[good]
                P = P[in_region(P)]
                edges_all.add(P)
                rr.log("world/edges/now", rr.Points3D(P, colors=colours(P), radii=0.0008))
                fp = edges_all.points()
                rr.log("world/edges/all", rr.Points3D(fp, colors=colours(fp), radii=0.001))
                report.append((k, len(good), int(good.sum()), len(pick)))
                print(f"frame {k:4d}: {len(pick)} neighbours, {good.sum():5d} of {len(good)} edge pixels with a clear height "
                      f"({time.time() - tic:.0f} s)", flush=True)
    for name, f in (("edges", edges_all), ("orbbec", orb_all)):
        p = f.points()
        np.save(run / f"scene3d_{name}.npy", p)
        print(f"{name}: {len(p)} voxels kept (seen by {MIN_HITS}+ frames)")
    print(f"saved {run / 'scene3d.rrd'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
