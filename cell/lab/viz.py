"""One way of showing everything: a fixed Rerun layout, one colour per object id, and helpers for each kind of thing.

Entity layout (what you click in the viewer's left panel):
    world/...            3D, robot base frame (metres): arm, cameras, point clouds, objects, grasps, paths
    cams/<cam>/rgb       2D camera images, with object outlines and ids drawn on top (what a VLM would be shown)
    cams/<cam>/depth     2D depth images
    heightmap            top-down height above the table
    narration            one plain-English line per decision

Timeline "step" = one act() call (or one synthetic view).
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from calibration.geometry import CHAIN, SURFACE_Z, UF0_Z_MM, _rot, project

CAMS_2D = ("table", "wrist", "side")
NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D: what the robot believes", origin="world"),
                rrb.Vertical(*[rrb.Spatial2DView(name=f"{c} camera", origin=f"cams/{c}/rgb") for c in CAMS_2D]),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="height map (mm above table)", origin="heightmap"),
                rrb.TextLogView(name="narration", origin="narration"),
                rrb.Vertical(*[rrb.Spatial2DView(name=f"{c} depth", origin=f"cams/{c}/depth") for c in ("table", "wrist")]),
                column_shares=[2, 3, 2],
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def start(name: str, save_to: Path | None = None, spawn: bool = False) -> None:
    """Begin a recording: to a .rrd file, a live viewer, or both. Logs the static scenery."""
    rr.init(name)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_to, default_blueprint=blueprint())
    if spawn:
        rr.spawn(default_blueprint=blueprint())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/robot_base", rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.1,
                                           colors=[[230, 60, 60], [60, 200, 60], [60, 60, 230]],
                                           labels=["x forward", "y left", "z up"]), static=True)
    a = np.linspace(0, 2 * np.pi, 200)
    ring = np.stack([0.55 * np.cos(a), 0.55 * np.sin(a), np.full_like(a, SURFACE_Z)], -1)
    rr.log("world/reach_550mm", rr.LineStrips3D([ring], colors=[[220, 160, 0]]), static=True)


def set_step(i: int) -> None:
    rr.set_time("step", sequence=i)


def narrate(text: str, level: str = "INFO") -> None:
    rr.log("narration", rr.TextLog(text, level=level))


# ---------------------------------------------------------------- colours

def id_color(i: int) -> list[int]:
    """A distinct, stable colour per object id (golden-ratio hue steps)."""
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return [int(255 * r), int(255 * g), int(255 * b)]


def height_colors(pts: np.ndarray, top_mm: float = 50.0) -> np.ndarray:
    h = np.clip((pts[:, 2] - SURFACE_Z) * 1000 / top_mm, 0, 1)
    return np.stack([h, 0.4 + 0.5 * h, 1 - h], -1)


# ---------------------------------------------------------------- robot

def uf0_to_base(xyz_mm) -> np.ndarray:
    """Fingertip coordinates (UF0, mm) -> robot base frame (m)."""
    p = np.asarray(xyz_mm, float)[..., :3] / 1000.0
    return p + [0, 0, -UF0_Z_MM / 1000.0]


def arm(joints, tcp_offset: float = 0.293) -> None:
    """The arm as a stick figure from forward kinematics, plus the fingertip."""
    R, p, pts = np.eye(3), np.zeros(3), [np.zeros(3)]
    for (off, axis, sign), q in zip(CHAIN, np.radians(np.asarray(joints, float)[:6])):
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q)
        pts.append(p.copy())
    tip = p + R[:, 0] * tcp_offset
    pts.append(tip)
    grip = float(np.asarray(joints, float)[6]) if len(joints) > 6 else 0.0
    rr.log("world/arm", rr.LineStrips3D([np.array(pts)], colors=[[240, 200, 40]], radii=0.006))
    rr.log("world/arm/fingertip", rr.Points3D([tip], colors=[[220, 40, 40] if grip >= 0.5 else [40, 200, 40]], radii=0.008,
                                              labels=["closed" if grip >= 0.5 else "open"]))


def path(waypoints_uf0, name: str = "plan") -> None:
    """A planned list of fingertip targets [x, y, z mm, w, p, r, gripper]: green where open, red where closed."""
    wp = np.asarray(waypoints_uf0, float)
    if wp.ndim != 2 or len(wp) < 2:
        return
    pts = uf0_to_base(wp[:, :3])
    closed = wp[:, 6] >= 0.5
    segs, cols = [], []
    for i in range(len(pts) - 1):
        segs.append(pts[i:i + 2])
        cols.append([220, 60, 60] if closed[i] else [60, 200, 90])
    rr.log(f"world/{name}", rr.LineStrips3D(segs, colors=cols, radii=0.0015))


# ---------------------------------------------------------------- cameras

def camera_frustum(path_: str, cam: dict, T: np.ndarray, depth: np.ndarray | None = None) -> None:
    """A depth camera in 3D: its pose, its viewing cone, and (optionally) its depth image projected out."""
    rr.log(path_, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    K = [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]
    rr.log(path_, rr.Pinhole(image_from_camera=K, resolution=list(cam["size"]), image_plane_distance=0.05))
    if depth is not None:
        rr.log(f"{path_}/depth", rr.DepthImage(depth, meter=1000.0))


def images(obs: dict) -> None:
    for c, img in obs.get("images", {}).items():
        rr.log(f"cams/{c}/rgb", rr.Image(img))
    for c, d in (obs.get("depth") or {}).items():
        rr.log(f"cams/{c}/depth", rr.DepthImage(d, meter=1000.0))


def project_to_image(cam_name: str, cams: dict, pts_base: np.ndarray, image_wh, joints=None) -> np.ndarray:
    """Base-frame points -> pixel coordinates in the image as served (e.g. 640x360), for the colour cameras."""
    cam = cams[cam_name]
    if cam["frame"] == "flange":
        from calibration.geometry import flange_pose
        pts = np.asarray(pts_base, float) @ np.linalg.inv(flange_pose(joints))[:3, :3].T + np.linalg.inv(flange_pose(joints))[:3, 3]
    else:
        pts = pts_base
    uv = project(cam, pts)
    W, H = NATIVE[cam_name]
    return uv * [image_wh[0] / W, image_wh[1] / H]


# ---------------------------------------------------------------- objects

def objects(objs: list[dict], cams: dict | None = None, obs: dict | None = None, kind: str = "objects") -> None:
    """Objects as the robot believes them: each {"id", "points" (N,3 base m) and/or "mesh" (trimesh), "label"?}.
    Same colour in 3D and on every camera image; outlines + "#id label" are drawn on the images (Set-of-Mark style)."""
    for o in objs:
        c = id_color(o["id"])
        text = f"#{o['id']}" + (f" {o['label']}" if o.get("label") else "")
        base = f"world/{kind}/{o['id']}"
        if o.get("mesh") is not None:
            m = o["mesh"]
            rr.log(base, rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces, vertex_normals=m.vertex_normals,
                                   albedo_factor=c + [int(255 * o.get("alpha", 1.0))]))
        if o.get("points") is not None:
            rr.log(f"{base}/points", rr.Points3D(o["points"], colors=c, radii=0.0008))
        pts = o["points"] if o.get("points") is not None else o["mesh"].vertices
        top = pts[np.argmax(pts[:, 2])]
        rr.log(f"{base}/label", rr.Points3D([top + [0, 0, 0.01]], colors=c, radii=0.002, labels=[text]))
        if cams is None or obs is None:
            continue
        for cam_name, img in obs.get("images", {}).items():
            uv = project_to_image(cam_name, cams, pts, (img.shape[1], img.shape[0]), joints=obs.get("joints"))
            ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] > -50) & (uv[:, 0] < img.shape[1] + 50)
            if ok.sum() < 3:
                continue
            hull = _hull(uv[ok])
            rr.log(f"cams/{cam_name}/rgb/{kind}/{o['id']}", rr.LineStrips2D([np.vstack([hull, hull[:1]])], colors=c, radii=1.0,
                                                                          labels=[text]))


def grasps(obj_id: int, centers, yaws_deg, valid, width_mm: float = 12.0, finger=(12.0, 6.0, 76.0), depth_mm: float = 15.0) -> None:
    """Candidate grips as pairs of finger boxes: green = fingers fit, red = blocked. centers in base metres."""
    boxes, halves, rots, cols = [], [], [], []
    for c, yaw, ok in zip(np.asarray(centers, float), yaws_deg, valid):
        a = np.radians(yaw)
        close_dir = np.array([np.cos(a), np.sin(a), 0.0])  # the direction the jaws close along
        for side in (-1, 1):
            off = close_dir * side * (width_mm / 2 + finger[1] / 2) / 1000
            boxes.append(c + off + [0, 0, (finger[2] / 2 - depth_mm) / 1000])
            halves.append([finger[1] / 2000, finger[0] / 2000, finger[2] / 2000])
            rots.append(rr.RotationAxisAngle(axis=[0, 0, 1], radians=a))
            cols.append([60, 200, 90, 160] if ok else [220, 60, 60, 90])
    rr.log(f"world/grasps/{obj_id}", rr.Boxes3D(centers=boxes, half_sizes=halves, rotation_axis_angles=rots, colors=cols))


def heightmap(hm_mm: np.ndarray) -> None:
    rr.log("heightmap", rr.DepthImage(np.nan_to_num(hm_mm, nan=0.0).astype(np.float32), meter=1000.0))


def _hull(pts: np.ndarray) -> np.ndarray:
    """2D convex hull of pixel coordinates."""
    return cv2.convexHull(np.asarray(pts, np.float32)).reshape(-1, 2)
