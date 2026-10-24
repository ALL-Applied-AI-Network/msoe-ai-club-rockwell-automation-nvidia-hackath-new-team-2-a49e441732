"""The evidence map ("the fish tank"): the space over the table as voxels, each holding what the cameras have shown about it.

Every depth pixel is an arrow on a string with a ruler: shot from the camera through the pixel, it says "nothing up to
here, something here, and I can't tell you anything beyond". So a look gives three kinds of evidence, never two:

    in front of the measured depth   the ray passed through: evidence of ABSENCE (empty)
    at the measured depth            something stopped it: evidence of PRESENCE (solid), and where the surface is
    behind it, or no reading         nothing: ABSENCE OF EVIDENCE. Unknown stays unknown; it is never filled in as empty.

Integration goes voxel by voxel (the standard TSDF way): each voxel the camera could see is projected into the depth
image and compared with the depth measured at that pixel; "at" means within the sensor's error at that distance, which
grows with depth. Three arrays:
    L       occupancy as log-odds: + solid, - empty, 0 no evidence. Clamped, so a scene that changes can be overturned.
    looks   how many looks touched the voxel (0 = never seen: different from seen-and-empty)
    sdf, w  the signed distance to the nearest surface (TSDF), averaged over looks with weights w, near surfaces only:
            the surface is where it crosses zero, finer than the voxel size once a few views agree

Below the table nothing can be seen; that's expected and ignored. The grid covers the reachable table.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose

L_FREE, L_HIT = -0.4, 0.85        # log-odds per look (a hit is worth ~2 passes: rays through empty space are many)
L_MIN, L_MAX = -2.0, 3.5          # clamps: room for a scene to change
L_SOLID = 1.2                     # believed solid: about two agreeing hits net of passes
SIG0, SIG2 = 0.0015, 0.010        # depth error (m): SIG0 + SIG2 * depth^2 (calibrate on real data)
W_MAX = 30.0                      # TSDF weight cap (old surfaces can still be overturned)


class EvidenceMap:
    def __init__(self, voxel: float = 0.002, x=(0.15, 0.65), y=(-0.45, 0.45), height: float = 0.12):
        self.v = float(voxel)
        z0 = SURFACE_Z - 3 * self.v  # a few voxels into the table, so its surface is a band of hits
        self.origin = np.array([x[0], y[0], z0])
        self.shape = tuple(int(np.ceil(s / self.v)) for s in (x[1] - x[0], y[1] - y[0], height + 3 * self.v))
        self.L = np.zeros(self.shape, np.float32)
        self.looks = np.zeros(self.shape, np.uint16)
        self.sdf = np.ones(self.shape, np.float32)
        self.w = np.zeros(self.shape, np.float32)
        self.k_table = int(round((SURFACE_Z - z0) / self.v))  # the z index of the table's top
        self.tries = np.zeros(self.shape, np.uint8)  # looks aimed at a gap voxel that still left it unknown

    # ------------------------------------------------------------ geometry

    def centers(self, i, j, k) -> np.ndarray:
        return self.origin + (np.stack([i, j, k], -1) + 0.5) * self.v

    def index(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """(i, j, k) of points, and which are inside the grid."""
        ijk = np.floor((np.asarray(pts) - self.origin) / self.v).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.shape), -1)
        return ijk, ok

    # ------------------------------------------------------------ evidence in

    def integrate(self, depth_mm: np.ndarray, cam: dict, joints=None, max_depth: float = 1.4) -> dict:
        """One depth image from one camera pose. cam: a depth camera from calibration (fx, fy, cx, cy, size, T, frame);
        joints: for the wrist camera. Returns counts of what this look said."""
        T = cam["T"] if cam["frame"] == "base" else flange_pose(joints) @ cam["T"]
        R, t = T[:3, :3], T[:3, 3]
        D = np.asarray(depth_mm, np.float32) / 1000.0
        H, W = D.shape
        far = min(max_depth, float(np.percentile(D[D > 0], 99)) + 0.05) if (D > 0).any() else max_depth
        # the voxels inside the camera's pyramid (its bounding box, clipped to the grid)
        corners = np.array([[(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], 1.0] for u in (0, W) for v in (0, H)])
        pyr = np.vstack([t, *(t + (R @ c) * far for c in corners)])
        lo = np.clip(np.floor((pyr.min(0) - self.origin) / self.v).astype(int), 0, self.shape)
        hi = np.clip(np.ceil((pyr.max(0) - self.origin) / self.v).astype(int) + 1, 0, self.shape)
        if np.any(hi <= lo):
            return {"free": 0, "hit": 0}
        i, j, k = np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        p = self.centers(i, j, k)
        pc = (p - t) @ R  # in the camera's frame (x right, y down, z forward)
        z = pc[:, 2]
        front = z > 0.02
        u = np.full(len(z), -1, int)
        vv = np.full(len(z), -1, int)
        u[front] = np.round(pc[front, 0] / z[front] * cam["fx"] + cam["cx"]).astype(int)
        vv[front] = np.round(pc[front, 1] / z[front] * cam["fy"] + cam["cy"]).astype(int)
        inside = front & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        i, j, k, z, u, vv = i[inside], j[inside], k[inside], z[inside], u[inside], vv[inside]
        d_meas = D[vv, u]
        has = d_meas > 0
        i, j, k, z, d_meas = i[has], j[has], k[has], z[has], d_meas[has]
        sig = SIG0 + SIG2 * d_meas ** 2   # this reading's error along the ray
        tau = np.maximum(1.5 * self.v, 2.0 * sig)
        gap = d_meas - z  # > 0: the voxel is in front of the measured surface
        free = gap > tau
        hit = np.abs(gap) <= tau
        # a reading is only as good as its sensor at that distance: its "solid" vote is spread over a band 2 sigma wide,
        # so each voxel in the band gets less of it (a far, noisy camera paints a faint thick band; a close one a sharp one)
        strength = np.minimum(1.0, 1.5 * self.v / (2.0 * sig))
        self.L[i[free], j[free], k[free]] += L_FREE
        self.L[i[hit], j[hit], k[hit]] += L_HIT * strength[hit]
        np.clip(self.L, L_MIN, L_MAX, out=self.L)
        seen = free | hit
        self.looks[i[seen], j[seen], k[seen]] = np.minimum(self.looks[i[seen], j[seen], k[seen]] + 1, 65000)
        # TSDF, in a band around the surface (3 tau in front, 1 tau behind)
        band = (gap > -tau) & (gap < 3 * tau)
        bi, bj, bk = i[band], j[band], k[band]
        s = np.clip(gap[band] / (3 * tau[band]), -1, 1)
        w0 = self.w[bi, bj, bk]
        self.sdf[bi, bj, bk] = (self.sdf[bi, bj, bk] * w0 + s) / (w0 + 1)
        self.w[bi, bj, bk] = np.minimum(w0 + 1, W_MAX)
        return {"free": int(free.sum()), "hit": int(hit.sum())}

    # ------------------------------------------------------------ what it believes

    def above_table(self) -> np.ndarray:
        """A mask over z: the voxels above the table's top (the space that can hold things)."""
        m = np.zeros(self.shape, bool)
        m[:, :, self.k_table + 3:] = True  # 6 mm up: clear of the table's own band of hits (a few mm of sensor error)
        return m

    def solid(self) -> np.ndarray:
        return self.L >= L_SOLID

    def empty(self) -> np.ndarray:
        return self.L <= -0.8

    def unknown(self) -> np.ndarray:
        return self.looks == 0

    def frontier(self) -> np.ndarray:
        """Unknown voxels above the table touching seen-empty ones: the edges of what we know. Not the inside of
        objects (they touch solid, not empty) and not under the table."""
        from scipy import ndimage
        near_empty = ndimage.binary_dilation(self.L < -0.2, iterations=1)  # any net evidence of empty: one look is enough
        return self.unknown() & near_empty & self.above_table() & (self.tries < 2)

    def missed(self, expected: np.ndarray) -> int:
        """After a look: the gap voxels it was expected to see that are still unknown get a strike; two strikes and they
        stop counting as gaps (something we can't model is in the way, or they can't be seen from anywhere we reach)."""
        if len(expected) == 0:
            return 0
        ijk, ok = self.index(expected)
        ijk = ijk[ok]
        still = self.looks[ijk[:, 0], ijk[:, 1], ijk[:, 2]] == 0
        s = ijk[still]
        self.tries[s[:, 0], s[:, 1], s[:, 2]] = np.minimum(self.tries[s[:, 0], s[:, 1], s[:, 2]] + 1, 255)
        return int(still.sum())

    def stats(self) -> dict:
        """How full the tank is: of the space above the table, how much has evidence; how much is solid; the frontier."""
        a = self.above_table()
        n = int(a.sum())
        seen = int((~self.unknown() & a).sum())
        return {"known_pct": round(100 * seen / max(n, 1), 1), "solid_voxels": int((self.solid() & a).sum()),
                "frontier_voxels": int(self.frontier().sum()), "voxel_mm": self.v * 1000}

    def solid_points(self, above_only: bool = True) -> np.ndarray:
        m = self.solid() & (self.above_table() if above_only else True)
        return self.centers(*np.nonzero(m))

    def frontier_points(self) -> np.ndarray:
        return self.centers(*np.nonzero(self.frontier()))

    def surface_mesh(self):
        """The fused surface (where the TSDF crosses zero, among voxels seen enough), as (vertices, faces)."""
        from skimage.measure import marching_cubes
        vol = np.where(self.w > 1, self.sdf, 1.0)
        if vol.min() > 0:
            return None
        verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(self.v,) * 3)
        return verts + self.origin + self.v / 2, faces
