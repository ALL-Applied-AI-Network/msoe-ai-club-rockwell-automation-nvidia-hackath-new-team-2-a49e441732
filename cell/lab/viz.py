"""One way of showing everything: a fixed Rerun layout, one colour per object id, and helpers for each kind of thing.

Layout:
    ┌──────────────────┬──────────────────┬───────────────┐
    │ World (truth)    │ Robot's mind     │ VLM thinking  │
    │ 3D, sim only     │ 3D: points →     │ objects table │
    │                  │ segments → ...   │ narration     │
    ├──────┬──────┬────┴─┬───────┬───────┴─┬─────────────┤
    │table │ side │ wrist│ table │ wrist   │ height map  │
    │ RGB  │ RGB  │ RGB  │ depth │ depth   │             │
    └──────┴──────┴──────┴───────┴─────────┴─────────────┘

Entity roots: truth/ (what's really there; synthetic runs only), mind/ (what the robot has worked out, never from the
truth), cams/<cam>/rgb|depth (raw inputs, with the robot's segment outlines drawn on the images), heightmap,
vlm, roster, narration. Timeline "step" = one act() call.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from calibration.geometry import CHAIN, SURFACE_Z, UF0_Z_MM, _rot, flange_pose, project

NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}


def blueprint() -> rrb.Blueprint:
    raw = [rrb.Spatial2DView(name=f"{c} camera", origin=f"cams/{c}/rgb") for c in ("table", "side", "wrist")]
    raw += [rrb.Spatial2DView(name=f"{c} depth", origin=f"cams/{c}/depth") for c in ("table", "wrist")]
    raw += [rrb.Spatial2DView(name="height map", origin="heightmap")]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="World (truth)", origin="truth"),
                rrb.Spatial3DView(name="Robot's mind", origin="mind"),
                rrb.Vertical(rrb.TextLogView(name="VLM thinking", origin="vlm"),
                             rrb.TextDocumentView(name="objects", origin="roster"),
                             rrb.TextLogView(name="narration", origin="narration"), row_shares=[2, 2, 2]),
                column_shares=[3, 3, 2],
            ),
            rrb.Horizontal(*raw),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def start(name: str, save_to: Path | None = None, spawn: bool = False) -> None:
    """Begin a recording: to a .rrd file, a live viewer, or both. Logs the static scenery in both 3D views."""
    rr.init(name)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_to, default_blueprint=blueprint())
    if spawn:
        rr.spawn(default_blueprint=blueprint())
    a = np.linspace(0, 2 * np.pi, 200)
    ring = np.stack([0.55 * np.cos(a), 0.55 * np.sin(a), np.full_like(a, SURFACE_Z)], -1)
    corners = np.array([[0.15, -0.45], [0.65, -0.45], [0.65, 0.45], [0.15, 0.45], [0.15, -0.45]])
    table = np.column_stack([corners, np.full(5, SURFACE_Z)])
    for root in ("truth", "mind"):
        rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(f"{root}/robot_base", rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.08,
                                                 colors=[[230, 60, 60], [60, 200, 60], [60, 60, 230]]), static=True)
        rr.log(f"{root}/reach_550mm", rr.LineStrips3D([ring], colors=[[220, 160, 0]]), static=True)
        rr.log(f"{root}/work_area", rr.LineStrips3D([table], colors=[[120, 120, 120]]), static=True)


def set_step(i: int) -> None:
    rr.set_time("step", sequence=i)


def narrate(text: str, level: str = "INFO") -> None:
    rr.log("narration", rr.TextLog(text, level=level))


def vlm(text: str, level: str = "INFO") -> None:
    rr.log("vlm", rr.TextLog(text, level=level))


def roster(markdown: str) -> None:
    rr.log("roster", rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN))


# ---------------------------------------------------------------- colours

def id_color(i: int) -> list[int]:
    """A distinct, stable colour per object id (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 0.95)
    return [int(255 * r), int(255 * g), int(255 * b)]


def height_colors(pts: np.ndarray, top_mm: float = 50.0) -> np.ndarray:
    h = np.clip((pts[:, 2] - SURFACE_Z) * 1000 / top_mm, 0, 1)
    return np.stack([0.35 + 0.6 * h, 0.35 + 0.55 * h, 0.4 - 0.2 * h], -1)  # grey table -> warm tops


def thin(pts: np.ndarray, voxel: float = 0.0015) -> np.ndarray:
    """One point per voxel: keeps the viewer fast without changing what you see."""
    if len(pts) == 0:
        return pts
    _, keep = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[keep]


# ---------------------------------------------------------------- robot

def arm(root: str, joints, tcp_offset: float = 0.293) -> None:
    """The arm as a stick figure from forward kinematics, plus the fingertip (green open, red closed)."""
    R, p, pts = np.eye(3), np.zeros(3), [np.zeros(3)]
    for (off, axis, sign), q in zip(CHAIN, np.radians(np.asarray(joints, float)[:6])):
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q)
        pts.append(p.copy())
    tip = p + R[:, 0] * tcp_offset
    pts.append(tip)
    closed = len(joints) > 6 and float(joints[6]) >= 0.5
    rr.log(f"{root}/arm", rr.LineStrips3D([np.array(pts)], colors=[[240, 200, 40]], radii=0.005))
    rr.log(f"{root}/arm/fingertip", rr.Points3D([tip], colors=[[220, 40, 40] if closed else [40, 200, 40]], radii=0.006))


def uf0_to_base(xyz_mm) -> np.ndarray:
    return np.asarray(xyz_mm, float)[..., :3] / 1000.0 + [0, 0, -UF0_Z_MM / 1000.0]


def path(root: str, waypoints_uf0, name: str = "plan") -> None:
    """Fingertip targets [x, y, z mm, w, p, r, gripper]: green where open, red where closed."""
    wp = np.asarray(waypoints_uf0, float)
    if wp.ndim != 2 or len(wp) < 2:
        return
    pts = uf0_to_base(wp[:, :3])
    segs = [pts[i:i + 2] for i in range(len(pts) - 1)]
    cols = [[220, 60, 60] if wp[i, 6] >= 0.5 else [60, 200, 90] for i in range(len(pts) - 1)]
    rr.log(f"{root}/{name}", rr.LineStrips3D(segs, colors=cols, radii=0.0012))


def camera_frustum(path_: str, cam: dict, T: np.ndarray) -> None:
    rr.log(path_, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    K = [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]
    rr.log(path_, rr.Pinhole(image_from_camera=K, resolution=list(cam["size"]), image_plane_distance=0.04))


def images(obs: dict) -> None:
    for c, img in (obs.get("images") or {}).items():
        rr.log(f"cams/{c}/rgb", rr.Image(img).compress(jpeg_quality=85))
    for c, d in (obs.get("depth") or {}).items():
        rr.log(f"cams/{c}/depth", rr.DepthImage(d, meter=1000.0))


def heightmap(hm_mm: np.ndarray) -> None:
    rr.log("heightmap", rr.DepthImage(np.nan_to_num(hm_mm, nan=0.0).astype(np.float32), meter=1000.0))


# ---------------------------------------------------------------- objects

def project_to_image(cam_name: str, cams: dict, pts_base: np.ndarray, image_wh, joints=None) -> np.ndarray:
    """Base-frame points -> pixels in the image as served (e.g. 640x360), for the colour cameras."""
    cam = cams[cam_name]
    pts = np.asarray(pts_base, float)
    if cam["frame"] == "flange":
        Ti = np.linalg.inv(flange_pose(joints))
        pts = pts @ Ti[:3, :3].T + Ti[:3, 3]
    W, H = NATIVE[cam_name]
    return project(cam, pts) * [image_wh[0] / W, image_wh[1] / H]


def truth_objects(objs: list[dict]) -> None:
    """The real objects (sim only): solid meshes in their own colours, with names."""
    rr.log("truth/objects", rr.Clear(recursive=True))
    for o in objs:
        m = o["mesh"]
        rr.log(f"truth/objects/{o['id']}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                     vertex_normals=m.vertex_normals, albedo_factor=list(o["rgb"]) + [255]))
        top = m.vertices[np.argmax(m.vertices[:, 2])]
        rr.log(f"truth/objects/{o['id']}/name", rr.Points3D([top + [0, 0, 0.008]], radii=0.0001, labels=[o["name"]]))


def segments(segs, cams: dict | None = None, obs: dict | None = None) -> None:
    """The robot's own segments: points in one colour per id, a label, and outlines on every camera image."""
    rr.log("mind/segments", rr.Clear(recursive=True))
    for c in ("table", "side", "wrist"):
        rr.log(f"cams/{c}/rgb/segments", rr.Clear(recursive=True))
    for s in segs:
        col = id_color(s.id)
        rr.log(f"mind/segments/{s.id}", rr.Points3D(thin(s.points, 0.001), colors=col, radii=0.0009))
        top = s.points[np.argmax(s.points[:, 2])]
        rr.log(f"mind/segments/{s.id}/label", rr.Points3D([top + [0, 0, 0.008]], colors=col, radii=0.0015,
                                                          labels=[f"#{s.id}"]))
        if cams is None or obs is None:
            continue
        for cam_name, img in (obs.get("images") or {}).items():
            uv = project_to_image(cam_name, cams, thin(s.points, 0.002), (img.shape[1], img.shape[0]), joints=obs.get("joints"))
            ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] > 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] > 0) & (uv[:, 1] < img.shape[0])
            if ok.sum() < 3:
                continue
            hull = cv2.convexHull(uv[ok].astype(np.float32)).reshape(-1, 2)
            rr.log(f"cams/{cam_name}/rgb/segments/{s.id}", rr.LineStrips2D([np.vstack([hull, hull[:1]])], colors=col,
                                                                           radii=0.8, labels=[f"#{s.id}"]))
