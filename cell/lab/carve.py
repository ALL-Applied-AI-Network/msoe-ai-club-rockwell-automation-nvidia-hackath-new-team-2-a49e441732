"""Silhouette carving: SAM masks from posed still shots -> solid objects above the table. No depth anywhere.

Each camera view says "everything solid projects inside these outlines". A voxel above the table survives
only if every view that can see it agrees; what survives is the visual hull. For convex objects the hull IS
the object, and the error is one-sided (hull >= object) as long as the masks are dilated enough to cover
segmentation and pose error. Shadows, stickers and prints carve to (near) zero height on their own: the
oblique views place nothing above the table there.

    views = [View(mask=union_of_masks, cam=cams["wrist"], joints=obs["joints"]), ...]
    solid, axes = carve(views)
    for ob in objects(solid, axes): print(ob["height_mm"], ob["size_mm"])
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from calibration.geometry import SURFACE_Z, flange_pose

# The carving grid: the reachable part of the table, from just above the surface to taller than anything here.
GRID_X = (0.20, 0.78)
GRID_Y = (-0.36, 0.36)
GRID_Z = (SURFACE_Z + 0.003, SURFACE_Z + 0.10)
VOX = 0.0025           # 2.5 mm voxels
DILATE_PX = 3          # grown over each mask: segmentation + pose error budget, keeps the error one-sided
MIN_VIEWS = 2          # a voxel is solid only if at least this many views put it inside an outline
MIN_VOXELS = 8         # smaller surviving clumps are noise
NEAR_PX = 60           # calibration pixels: a dissent this close to a mask SAW the thing and always carves
                       # (the top-shaving rays pass just outside an outline); a dissent with no mask
                       # anywhere near probably just missed the thing, and one such miss is forgiven


@dataclass
class View:
    mask: np.ndarray            # bool (H, W): the union of every accepted mask in this shot
    cam: dict                   # a calibration.geometry.load() entry ("table", "side", "wrist")
    joints: np.ndarray | None = None   # obs["joints"] for the wrist camera (it moves with the arm)

    def T_base_cam(self) -> np.ndarray:
        T = self.cam["T"]
        if self.cam["frame"] == "flange":
            if self.joints is None:
                raise ValueError("the wrist camera moves with the arm: pass joints")
            T = flange_pose(self.joints) @ T
        return T


def _axes(vox: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.arange(GRID_X[0], GRID_X[1], vox, dtype=np.float32),
            np.arange(GRID_Y[0], GRID_Y[1], vox, dtype=np.float32),
            np.arange(GRID_Z[0], GRID_Z[1], vox, dtype=np.float32))


def carve(views: list[View], vox: float = VOX, dilate_px: int = DILATE_PX,
          min_views: int = MIN_VIEWS) -> tuple[np.ndarray, tuple]:
    """-> (solid bool (nx, ny, nz), (xs, ys, zs) voxel-centre axes in base metres)."""
    xs, ys, zs = _axes(vox)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    near_kill = np.zeros(len(pts), bool)   # a view that saw the thing said: outside it
    far_out = np.zeros(len(pts), np.uint8)  # views with no outline near: "nothing here at all"
    in_k = np.zeros(len(pts), np.uint8)
    for v in views:
        T = np.linalg.inv(v.T_base_cam()).astype(np.float32)
        pc = pts @ T[:3, :3].T + T[:3, 3]
        front = pc[:, 2] > 0.03
        x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        d = 1 + v.cam["k1"] * (x * x + y * y)
        w, h = v.cam["size"]
        mh, mw = v.mask.shape
        u = (v.cam["fx"] * x * d + w / 2) * (mw / w)
        vv = (v.cam["fx"] * y * d + h / 2) * (mh / h)
        ui, vi = u.astype(np.int32), vv.astype(np.int32)
        inb = front & (ui >= 0) & (ui < mw) & (vi >= 0) & (vi < mh)
        k = max(1, round(dilate_px * mw / w))  # dilate_px is at calibration scale; the mask may be smaller
        kern = np.ones((2 * k + 1,) * 2, np.uint8)
        mask = cv2.dilate(v.mask.astype(np.uint8), kern).astype(bool)
        inside = np.zeros(len(pts), bool)
        inside[inb] = mask[vi[inb], ui[inb]]
        in_k += inside
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        near = np.zeros(len(pts), bool)
        near[inb] = dt[vi[inb], ui[inb]] <= NEAR_PX * mw / w
        near_kill |= inb & ~inside & near
        far_out += (inb & ~inside & ~near).astype(np.uint8)
    solid = (~near_kill & (far_out <= 1) & (in_k >= min_views)).reshape(len(xs), len(ys), len(zs))
    return solid, (xs, ys, zs)


def on_table(mask: np.ndarray, cam: dict, joints=None) -> bool:
    """Everything we care about sits on the table: a mask whose bottom edge doesn't reach the work
    surface inside the workspace (the robot's own arm or gripper, the far wall) is dropped."""
    from calibration.geometry import pixel_to_base
    vs, us = np.nonzero(mask)
    if not len(vs):
        return False
    i = int(np.argmax(vs))
    w, h = cam["size"]
    mh, mw = mask.shape
    p = pixel_to_base(cam, (us[i] * w / mw, vs[i] * h / mh), joints=joints)
    return GRID_X[0] - 0.05 < p[0] < GRID_X[1] + 0.05 and GRID_Y[0] - 0.05 < p[1] < GRID_Y[1] + 0.05


def objects(solid: np.ndarray, axes: tuple, min_voxels: int = MIN_VOXELS) -> list[dict]:
    """Connected clumps of solid voxels -> one dict each: where it is and how big (mm, base frame)."""
    xs, ys, zs = axes
    lab, n = ndimage.label(solid, structure=np.ones((3, 3, 3)))  # diagonals connect: no twin objects from one clump
    out = []
    for i in range(1, n + 1):
        ix, iy, iz = np.nonzero(lab == i)
        if len(ix) < min_voxels:
            continue
        px, py, pz = xs[ix], ys[iy], zs[iz]
        xy = np.stack([px, py], -1) * 1000.0
        c = xy.mean(0)
        u, s, vt = np.linalg.svd(xy - c, full_matrices=False)
        ext = np.ptp(u * s, 0) if len(xy) > 1 else np.zeros(2)
        vox = float(xs[1] - xs[0]) * 1000.0
        out.append({
            "x_mm": round(float(c[0]), 1), "y_mm": round(float(c[1]), 1),
            "top_mm": round(float(pz.max() - SURFACE_Z) * 1000.0 + vox / 2, 1),
            "size_mm": (round(float(ext[0]) + vox, 1), round(float(ext[1]) + vox, 1)),
            "yaw": round(float(np.arctan2(vt[0, 1], vt[0, 0])), 3),
            "voxels": len(ix),
            "ixyz": (ix, iy, iz),
        })
    out.sort(key=lambda o: -o["voxels"])
    return out


def name_objects(obs: list[dict], axes: tuple, per_word: dict[str, np.ndarray], view: View) -> None:
    """Give each carved object the word whose mask (in one top-ish view) covers most of its voxels."""
    xs, ys, zs = axes
    T = np.linalg.inv(view.T_base_cam()).astype(np.float32)
    for ob in obs:
        ix, iy, iz = ob["ixyz"]
        pts = np.stack([xs[ix], ys[iy], zs[iz]], -1)
        pc = pts @ T[:3, :3].T + T[:3, 3]
        x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        d = 1 + view.cam["k1"] * (x * x + y * y)
        w, h = view.cam["size"]
        mh, mw = next(iter(per_word.values())).shape
        u = ((view.cam["fx"] * x * d + w / 2) * (mw / w)).astype(np.int32)
        v = ((view.cam["fx"] * y * d + h / 2) * (mh / h)).astype(np.int32)
        ok = (u >= 0) & (u < mw) & (v >= 0) & (v < mh) & (pc[:, 2] > 0.03)
        best, hits = "?", 0
        for word, m in per_word.items():
            k = int(m[v[ok], u[ok]].sum())
            if k > hits:
                best, hits = word, k
        ob["name"] = best
