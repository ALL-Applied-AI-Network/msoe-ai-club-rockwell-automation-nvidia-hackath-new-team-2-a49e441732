"""Depth along colour edges, from many far-apart frames with known (re-timed) poses: the dense step after tracks.

A corner can be followed from frame to frame; an edge cannot (it slides along itself), but with the camera poses
known it does not need to be: an edge pixel's 3D point lies on its ray, so trying depths along that ray and
checking whether the other frames see the same patch there is a 1D search (along the epipolar line).

For a reference frame:
    edge pixels: Canny on the L, a, b channels (colour edges, so a blue cube on white counts), every EDGE_STRIDE-th
    neighbours: up to MAX_NBRS frames whose camera is BASE_MM apart from it, spread out (wide eyes)
    heights: every COARSE_MM from the table to the top of REGION, then FINE_MM around the best
    patches: one-sided, on each side of the edge (the object's side matches; the background behind it changes)
    kept only with evidence: NCC peak >= PEAK, every height more than 4 mm away MARGIN lower, at least MIN_SEEN
        neighbours seeing it, and its height known to SIGMA_MM (half-width of the peak, from the fine scores)
Returns the points, their 1-sigma along the ray, and their colour.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from calibration.geometry import SURFACE_Z

EDGE_STRIDE = 2
BASE_MM = (15, 120)
MAX_NBRS = 6
COARSE_MM = 2.0
FINE_MM = np.arange(-2.0, 2.01, 0.5)
PEAK, MARGIN = 0.8, 0.1
MIN_SEEN = 4
SIGMA_MM = 2.0
H_RANGE_MM = (-3.0, 120.0)

ALONG = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
ACROSS = np.array([-1.0, 0.0, 1.0])
_A, _C = (m.ravel() for m in np.meshgrid(ALONG, ACROSS, indexing="ij"))


class Lens:
    """The wrist colour camera at 848 x 480 (k1 radial distortion), posed by a 4x4 camera-to-base T."""

    def __init__(self, cam):
        self.fx, self.k1 = cam["fx"], cam["k1"]
        self.cx, self.cy = cam["size"][0] / 2, cam["size"][1] / 2

    def rays(self, T, uv):
        xd, yd = (uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fx
        x, y = xd, yd
        for _ in range(8):
            d = 1 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        r = np.stack([x, y, np.ones_like(x)], -1) @ T[:3, :3].T
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, T, P):
        pc = (P - T[:3, 3]) @ T[:3, :3]
        x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.stack([self.fx * x * d + self.cx, self.fx * y * d + self.cy], -1), pc[:, 2]


def patches(img, uv, nrm):
    tan = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
    xs = uv[:, 0:1] + nrm[:, 0:1] * _A[None] + tan[:, 0:1] * _C[None]
    ys = uv[:, 1:2] + nrm[:, 1:2] * _A[None] + tan[:, 1:2] * _C[None]
    return map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(len(uv), -1)


def ncc(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-6)


def neighbours(T, k):
    c = T[:, :3, 3]
    dist = np.linalg.norm(c - c[k], axis=1) * 1000
    cand = np.nonzero((dist >= BASE_MM[0]) & (dist <= BASE_MM[1]))[0]
    pick = []
    for i in cand[np.argsort(-dist[cand])]:
        if all(np.linalg.norm(c[i] - c[j]) * 1000 > 10 for j in pick):
            pick.append(i)
        if len(pick) >= MAX_NBRS:
            break
    return pick


def edge_points(lens: Lens, grays, colours, T, k):
    """Edge points of frame k with evidence: (P (n, 3), sigma m (n,), rgb (n, 3), stats dict)."""
    g, col = grays[k], colours[k]
    nb = neighbours(T, k)
    empty = (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.uint8), {"edges": 0, "kept": 0, "nbrs": len(nb)})
    if len(nb) < MIN_SEEN:
        return empty
    edges = np.zeros(g.shape, np.uint8)
    for ch in cv2.split(cv2.cvtColor(col, cv2.COLOR_BGR2LAB)):
        edges |= cv2.Canny(ch, 10, 30)
    edges[:10] = edges[-10:] = 0
    edges[:, :10] = edges[:, -10:] = 0
    vs, us = np.nonzero(edges)
    sel = slice(None, None, EDGE_STRIDE)
    vs, us = vs[sel], us[sel]
    if len(us) == 0:
        return empty
    uv = np.stack([us, vs], 1).astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[vs, us]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[vs, us]
    nrm0 = -np.stack([gx, gy], 1) / (np.hypot(gx, gy)[:, None] + 1e-6)
    ray = lens.rays(T[k], uv)
    o = T[k][:3, 3]

    def at_height(h_mm):
        z = SURFACE_Z + np.asarray(h_mm) / 1000
        return o + ray * ((z - o[2]) / ray[:, 2])[:, None]

    def score(ref_p, nrm, h_mm):
        P = at_height(h_mm)
        tot, n = np.zeros(len(uv)), np.zeros(len(uv))
        for j in nb:
            q, depth = lens.project(T[j], P)
            ok = (q[:, 0] > 6) & (q[:, 0] < 842) & (q[:, 1] > 6) & (q[:, 1] < 474) & (depth > 0.03)
            if ok.any():
                tot[ok] += ncc(ref_p[ok], patches(grays[j], q[ok], nrm[ok]))
                n[ok] += 1
        return np.where(n >= MIN_SEEN, tot / np.maximum(n, 1), -1.0), n

    coarse_h = np.arange(H_RANGE_MM[0], H_RANGE_MM[1] + 0.01, COARSE_MM)
    best = None
    for side in (1.0, -1.0):
        nrm = side * nrm0
        ref_p = patches(g, uv, nrm)
        coarse = np.stack([score(ref_p, nrm, np.full(len(uv), h))[0] for h in coarse_h], 1)
        hc = coarse_h[coarse.argmax(1)]
        fs = [score(ref_p, nrm, hc + d) for d in FINE_MM]
        fine = np.stack([s for s, _ in fs], 1)
        seen = fs[len(FINE_MM) // 2][1]
        i = fine.argmax(1)
        h = hc + FINE_MM[i]
        peak = fine.max(1)
        second = np.where(np.abs(coarse_h[None] - h[:, None]) > 4.0, coarse, -1).max(1)
        # half-width of the peak: the fine heights within 0.03 of the best
        width = (fine >= peak[:, None] - 0.03).sum(1) * (FINE_MM[1] - FINE_MM[0]) / 2
        cand = (h, peak, second, width, seen)
        if best is None:
            best = cand
        else:
            better = peak > best[1]
            best = tuple(np.where(better, c, b) for c, b in zip(cand, best))
    h, peak, second, width, seen = best
    sig_h = np.maximum(width, 0.5) / 1000
    sig = sig_h / np.maximum(np.abs(ray[:, 2]), 0.2)  # along the ray
    good = (peak >= PEAK) & (peak - second >= MARGIN) & (seen >= MIN_SEEN) & (sig_h * 1000 <= SIGMA_MM)
    good &= (h > H_RANGE_MM[0] + 1) & (h < H_RANGE_MM[1] - 1)
    P = at_height(h)[good]
    rgb = col[vs[good], us[good], ::-1]
    return P, sig[good], rgb, {"edges": len(uv), "kept": int(good.sum()), "nbrs": len(nb)}
