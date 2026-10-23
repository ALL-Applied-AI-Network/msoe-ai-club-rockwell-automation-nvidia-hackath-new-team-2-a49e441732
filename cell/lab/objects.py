"""From points to objects: once a thing has moved as one piece, the robot builds a solid 3D model of it.

    convex parts (cubes, dice, dowels, a lying screw): every point seen on it, plus its lowest seen band of points
        dropped onto the table (the hidden bottom rests there), wrapped in a convex hull: a closed solid. Only the lowest
        band, so a part that bulges out above the table (a d20) isn't filled in straight down under its bulge.
    a height-field solid: a column from the table up to each measured height (1 mm grid). Tighter for parts that
        aren't convex (a screw's head is wider than its shank; L shapes), but it fills in under overhangs.
Both can only over-fill the real shape (neither cuts into it), so the robot builds both and keeps the smaller.
What neither can know from above: the space under an arm or a bulge. Soldiers and jacks come out too big.

A model lives in its own frame (origin at its centre, on the table) with a pose in the base frame, so tracking it later
means updating one 4x4 matrix, not rebuilding it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from calibration.geometry import SURFACE_Z
from lab.perception import MIN_HEIGHT_MM, mean_colour



@dataclass
class Model:
    mesh: trimesh.Trimesh          # in the object's own frame
    pose: np.ndarray               # 4x4, object frame -> base frame
    kind: str                      # "convex hull" | "height field"
    colour: np.ndarray | None
    n_points: int
    history: list[str] = field(default_factory=list)

    def world_mesh(self) -> trimesh.Trimesh:
        m = self.mesh.copy()
        m.apply_transform(self.pose)
        return m

    def samples(self, n: int = 1500) -> np.ndarray:
        """Points on the model's surface, in the base frame (for fitting it to new looks)."""
        return np.asarray(self.world_mesh().sample(n))


def build(points: np.ndarray) -> Model | None:
    """A solid model from a thing's points ((N, 3) or (N, 6), base frame)."""
    p = points[(points[:, 2] - SURFACE_Z) * 1000 > MIN_HEIGHT_MM, :3]
    if len(p) < 30:
        return None
    centre = np.array([*np.median(p[:, :2], axis=0), SURFACE_Z])
    pose = np.eye(4)
    pose[:3, 3] = centre
    local = p - centre
    base = local[local[:, 2] <= local[:, 2].min() + 0.002].copy()  # the lowest 2 mm the camera saw
    base[:, 2] = 0.0
    hull = trimesh.Trimesh(np.vstack([local, base])).convex_hull
    field_ = _height_field(local)
    if hull.volume <= field_.volume:
        return Model(hull, pose, "convex hull", mean_colour(points), len(p))
    return Model(field_, pose, "height field", mean_colour(points), len(p))


def _height_field(local: np.ndarray, cell: float = 0.001) -> trimesh.Trimesh:
    """Columns from the table up to the highest point in each 1 mm cell, as one closed mesh."""
    ij = np.floor(local[:, :2] / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    nz = int(np.ceil(local[:, 2].max() / cell)) + 1
    vox = np.zeros((ij[:, 0].max() + 1, ij[:, 1].max() + 1, nz), bool)
    top = np.zeros(vox.shape[:2])
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), local[:, 2])
    k = np.ceil(top / cell).astype(int)
    for i, j in zip(*np.nonzero(k)):
        vox[i, j, :k[i, j]] = True
    m = trimesh.voxel.VoxelGrid(vox).marching_cubes
    m.apply_scale(cell)
    m.apply_translation([lo[0] * cell, lo[1] * cell, -0.5 * cell])
    return m


def compare(model: Model, true_mesh: trimesh.Trimesh) -> dict:
    """How close a model is to the real object: volume ratio, centre error, mean surface distance (both ways)."""
    m = model.world_mesh()
    a, b = np.asarray(m.sample(2000)), np.asarray(true_mesh.sample(2000))
    from scipy.spatial import cKDTree
    d = 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())
    return {"volume_ratio": float(m.volume / true_mesh.volume) if true_mesh.volume > 0 else None,
            "centre_mm": float(np.linalg.norm(m.centroid - true_mesh.centroid) * 1000),
            "surface_mm": float(d * 1000)}


def fit(model: Model, points: np.ndarray, search: float = 0.02, max_dist: float = 0.003) -> tuple[np.ndarray, float, float]:
    """Fit a model to new points near where it is expected (its current pose). Returns (new pose, fitness, rms mm).
    fitness = share of the model's visible surface (not its bottom) with a measured point within 1.5 mm after fitting."""
    import open3d as o3d
    m = model.world_mesh()
    pts, idx = m.sample(3000, return_index=True)
    up = m.face_normals[idx][:, 2] > -0.3  # the bottom rests on the table: nobody sees it
    src = np.asarray(pts)[up]
    c = src.mean(0)
    near = points[np.max(np.abs(points[:, :2] - c[:2]), axis=1) < search, :3]
    near = near[(near[:, 2] - SURFACE_Z) * 1000 > MIN_HEIGHT_MM]
    if len(near) < 20:
        return model.pose, 0.0, float("inf")
    P = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    Q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near))
    r = o3d.pipelines.registration.registration_icp(P, Q, max_dist, np.eye(4),
                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    moved = (np.asarray(src) @ r.transformation[:3, :3].T) + r.transformation[:3, 3]
    from scipy.spatial import cKDTree
    d = cKDTree(near).query(moved)[0]
    return r.transformation @ model.pose, float((d < 0.0015).mean()), float(np.sqrt((d[d < 0.003] ** 2).mean()) * 1000) if (d < 0.003).any() else float("inf")


def claim(model: Model, points: np.ndarray, within: float = 0.002) -> np.ndarray:
    """Mask of points explained by the model (within 2 mm of its surface): they belong to it, not to anything new."""
    m = model.world_mesh()
    near = np.max(np.abs(points[:, :2] - m.centroid[:2]), axis=1) < (m.extents[:2].max() / 2 + 0.005)
    mask = np.zeros(len(points), bool)
    if near.any():
        from scipy.spatial import cKDTree
        d = cKDTree(np.asarray(m.sample(4000))).query(points[near, :3])[0]
        mask[np.flatnonzero(near)] = d < within  # cameras only see surfaces, so near the surface is enough
    return mask
