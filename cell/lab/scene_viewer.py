"""See what the cameras see, in 3D: a synthetic scene rendered through the cell's real depth-camera calibration.

Builds a table with a few awkward objects, renders the table and wrist depth images the way the real Orbbec cameras
would (same intrinsics, poses, noise), turns them back into 3D points with calibration.geometry.depth_to_base (the
same call a policy would use), fuses them into a height map, and logs everything to a Rerun recording.

    uv run python -m lab.scene_viewer          # writes lab/out/scene.rrd and opens the viewer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
import trimesh
from scipy.optimize import least_squares

from calibration.geometry import SURFACE_Z, depth_to_base, flange_pose, load

OUT = Path(__file__).resolve().parent / "out"
JOINT_LIMITS = [(-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360)]
HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286]
TCP_OFFSET = 0.293  # fingertip, metres along the flange's +x (tool) axis


# ---------------------------------------------------------------- scene

def _on_table(m: trimesh.Trimesh, x: float, y: float, yaw_deg: float = 0.0) -> trimesh.Trimesh:
    """Rotate about z, then drop the mesh so its lowest point rests on the table at (x, y)."""
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1]))
    lo = m.bounds[0]
    c = m.bounds.mean(0)
    m.apply_translation([x - c[0], y - c[1], SURFACE_Z - lo[2]])
    return m


def build_scene() -> dict[str, trimesh.Trimesh]:
    mm = 1e-3
    objs: dict[str, trimesh.Trimesh] = {}
    for i in range(3):  # a flush 3x3 tile of 16 mm cubes: one flat slab to a depth camera
        for j in range(3):
            objs[f"cube_{i}{j}"] = _on_table(trimesh.creation.box([16 * mm] * 3), 0.45 + i * 16 * mm, -0.12 + j * 16 * mm)
    shank = trimesh.creation.cylinder(3 * mm, 30 * mm)
    head = trimesh.creation.cylinder(6 * mm, 4 * mm)
    head.apply_translation([0, 0, 17 * mm])
    screw = trimesh.util.concatenate([shank, head])
    screw.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    objs["screw"] = _on_table(screw, 0.36, -0.02, yaw_deg=30)
    wall = trimesh.creation.annulus(18 * mm, 20 * mm, 40 * mm)
    brim = trimesh.creation.annulus(18 * mm, 26 * mm, 2 * mm)
    brim.apply_translation([0, 0, 20 * mm])
    bottom = trimesh.creation.cylinder(20 * mm, 2 * mm)
    bottom.apply_translation([0, 0, -19 * mm])
    objs["cup_with_brim"] = _on_table(trimesh.util.concatenate([wall, brim, bottom]), 0.38, 0.14)
    legs = []
    for axis in np.eye(3):
        leg = trimesh.creation.cylinder(2 * mm, 26 * mm)
        leg.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis))
        legs.append(leg)
    jack = trimesh.util.concatenate(legs)
    jack.apply_transform(trimesh.transformations.rotation_matrix(np.radians(35), [1, 1, 0]))
    objs["jack"] = _on_table(jack, 0.46, 0.06, yaw_deg=20)
    objs["tongue_depressor"] = _on_table(trimesh.creation.box([150 * mm, 18 * mm, 2 * mm]), 0.30, 0.25, yaw_deg=-20)
    return objs


def table_mesh() -> trimesh.Trimesh:
    t = trimesh.creation.box([1.08, 1.47, 0.02])
    t.apply_translation([0.17, 0.01, SURFACE_Z - 0.01])
    return t


# ---------------------------------------------------------------- cameras

def wrist_ik(tip_xyz, yaw_deg: float = 0.0, q0=HOME) -> np.ndarray:
    """Joints that put the fingertip at tip_xyz (base frame, m) with the fingers pointing straight down."""
    yaw = np.radians(yaw_deg)

    def err(q):
        T = flange_pose(q)
        tip = T[:3, 3] + T[:3, 0] * TCP_OFFSET
        return np.concatenate([(tip - tip_xyz) * 10, T[:3, 0] - [0, 0, -1],
                               0.3 * (T[:3, 1] - [np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2), 0])])

    lo, hi = np.array(JOINT_LIMITS, float).T
    sol = least_squares(err, np.clip(q0, lo, hi), bounds=(lo, hi))
    T = flange_pose(sol.x)
    miss = np.linalg.norm(T[:3, 3] + T[:3, 0] * TCP_OFFSET - tip_xyz)
    if miss > 2e-3:
        raise ValueError(f"no IK solution within 2 mm of {tip_xyz} (missed by {miss * 1000:.1f} mm)")
    return sol.x


def camera_pose(cam: dict, joints=None) -> np.ndarray:
    return flange_pose(joints) @ cam["T"] if cam["frame"] == "flange" else cam["T"]


def render_depth(cam: dict, meshes: list[trimesh.Trimesh], joints=None, rng=None) -> np.ndarray:
    """uint16 mm along the optical axis, like the real stream: quantised, noisier with distance, 0 = no return."""
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.core.Tensor(m.vertices, o3d.core.float32), o3d.core.Tensor(m.faces, o3d.core.uint32))
    T = camera_pose(cam, joints)
    w, h = cam["size"]
    v, u = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones((h, w))], -1)
    d_base = d_cam @ T[:3, :3].T  # unnormalised, so the hit distance t is the depth along the optical axis
    rays = np.concatenate([np.broadcast_to(T[:3, 3], d_base.shape), d_base], -1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    t[~np.isfinite(t)] = 0
    if rng is not None:  # roughly an Orbbec Gemini's error: ~1.5 mm at 0.8 m, growing with distance squared
        t = np.where(t > 0, t + rng.normal(0, 1.5e-3 * (t / 0.8) ** 2), 0)
    return np.round(t * 1000).astype(np.uint16)


# ---------------------------------------------------------------- fusion

def height_map(points: np.ndarray, x_range=(0.25, 0.60), y_range=(-0.20, 0.32), cell=0.001) -> np.ndarray:
    """Top-down height above the table (mm) on a 1 mm grid: the max z per cell, NaN where nothing was seen."""
    nx, ny = int(np.ceil((x_range[1] - x_range[0]) / cell)), int(np.ceil((y_range[1] - y_range[0]) / cell))
    ix = ((points[:, 0] - x_range[0]) / cell).astype(int)
    iy = ((points[:, 1] - y_range[0]) / cell).astype(int)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    hm = np.full(nx * ny, -np.inf)
    np.maximum.at(hm, ix[keep] * ny + iy[keep], (points[keep, 2] - SURFACE_Z) * 1000)
    hm[np.isinf(hm)] = np.nan
    return hm.reshape(nx, ny)


# ---------------------------------------------------------------- logging

def log_camera(path: str, cam: dict, T: np.ndarray, depth: np.ndarray) -> None:
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    K = [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]
    rr.log(path, rr.Pinhole(image_from_camera=K, resolution=list(cam["size"]), image_plane_distance=0.05))
    rr.log(f"{path}/depth", rr.DepthImage(depth, meter=1000.0))


def log_points(path: str, pts: np.ndarray) -> None:
    h = np.clip((pts[:, 2] - SURFACE_Z) / 0.05, 0, 1)  # colour by height: table blue, 50 mm+ yellow
    colors = np.stack([h, 0.4 + 0.5 * h, 1 - h], -1)
    rr.log(path, rr.Points3D(pts, colors=colors, radii=0.0006))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rr.init("yaad_scene")
    rr.save(OUT / "scene.rrd")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/robot_base", rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.1,
                                           colors=[[230, 60, 60], [60, 200, 60], [60, 60, 230]],
                                           labels=["x (forward)", "y (left)", "z (up)"]), static=True)
    a = np.linspace(0, 2 * np.pi, 200)
    rr.log("world/reach_550mm", rr.LineStrips3D([np.stack([0.55 * np.cos(a), 0.55 * np.sin(a), np.full_like(a, SURFACE_Z)], -1)],
                                                colors=[[220, 160, 0]]), static=True)

    cams = load()
    rng = np.random.default_rng(0)
    objs = build_scene()
    for name, m in objs.items():
        rr.log(f"world/truth/{name}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                vertex_normals=m.vertex_normals, albedo_factor=[200, 200, 200, 255]),
               static=True)
    meshes = [table_mesh(), *objs.values()]

    all_pts = []
    rr.set_time("view", sequence=0)
    d = render_depth(cams["table_depth"], meshes, rng=rng)
    log_camera("world/table_cam", cams["table_depth"], camera_pose(cams["table_depth"]), d)
    pts = depth_to_base(cams["table_depth"], d, stride=1)
    all_pts.append(pts)
    log_points("world/points/table_cam", pts)
    rr.log("heightmap/table_only", rr.DepthImage(np.nan_to_num(height_map(pts), nan=0.0), meter=1000.0))

    # a wrist sweep: fingertip 120 mm over each spot of interest, camera looking down
    spots = [(0.47, -0.10), (0.37, 0.00), (0.46, 0.06), (0.38, 0.14), (0.32, 0.24)]
    for k, (x, y) in enumerate(spots, start=1):
        rr.set_time("view", sequence=k)
        q = wrist_ik(np.array([x, y, SURFACE_Z + 0.12]))
        d = render_depth(cams["wrist_depth"], meshes, joints=q, rng=rng)
        log_camera("world/wrist_cam", cams["wrist_depth"], camera_pose(cams["wrist_depth"], q), d)
        pts = depth_to_base(cams["wrist_depth"], d, joints=q)
        all_pts.append(pts)
        log_points("world/points/wrist_cam", pts)
        fused = np.concatenate(all_pts)
        log_points("world/points/fused", fused)
        rr.log("heightmap/fused", rr.DepthImage(np.nan_to_num(height_map(fused), nan=0.0), meter=1000.0))
        rr.log("log", rr.TextLog(f"view {k}: wrist over ({x * 1000:.0f}, {y * 1000:.0f}) mm, joints {np.round(q, 1).tolist()}, "
                                 f"{len(pts)} points"))
    print(f"wrote {OUT / 'scene.rrd'}")


if __name__ == "__main__":
    main()
