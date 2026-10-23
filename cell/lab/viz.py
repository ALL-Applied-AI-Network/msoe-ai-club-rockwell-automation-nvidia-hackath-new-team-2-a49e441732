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


PRINT = False  # also print narration to the terminal (lab.trial --print)


def narrate(text: str, level: str = "INFO") -> None:
    rr.log("narration", rr.TextLog(text, level=level))
    if PRINT:
        print(("   " if level == "DEBUG" else "") + text, flush=True)


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
    tops, names = [], []
    for o in objs:
        m = o["mesh"]
        rr.log(f"truth/objects/{o['id']}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                     vertex_normals=m.vertex_normals, albedo_factor=list(o["rgb"]) + [255]))
        tops.append(m.vertices[np.argmax(m.vertices[:, 2])] + [0, 0, 0.008])
        names.append(o["name"])
    # all name tags in one entity: click its eye icon (left panel, "truth/labels") to hide or show them all
    rr.log("truth/labels", rr.Points3D(tops, radii=0.0001, labels=names))


def segments(segs, cams: dict | None = None, obs: dict | None = None) -> None:
    """The robot's own segments: points in one colour per id, a label, and outlines on every camera image."""
    rr.log("mind/segments", rr.Clear(recursive=True))
    for c in ("table", "side", "wrist"):
        rr.log(f"cams/{c}/rgb/segments", rr.Clear(recursive=True))
    tops, tags, cols = [], [], []
    for s in segs:
        col = id_color(s.id)
        rr.log(f"mind/segments/{s.id}", rr.Points3D(thin(s.points, 0.001), colors=col, radii=0.0009))
        tops.append(s.points[np.argmax(s.points[:, 2])] + [0, 0, 0.008])
        tags.append(getattr(s, "tag", f"#{s.id}"))
        cols.append(col)
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
    # all id tags in one entity: click its eye icon (left panel, "mind/labels") to hide or show them all
    rr.log("mind/labels", rr.Points3D(tops, colors=cols, radii=0.0015, labels=tags) if tops else rr.Clear(recursive=False))


# ---------------------------------------------------------------- the robot's things (world model)

STATUS_MARK = {"blob": "?", "one": "✓"}


def things(world, cams: dict | None = None, obs: dict | None = None) -> None:
    """The robot's world model: each thing in its id colour, tagged "#7 ?" (untested blob) or "#7 ✓" (moved as one)."""
    ts = list(world.things.values())
    for t in ts:
        t.tag = f"#{t.id} {STATUS_MARK.get(t.status, t.status)}"
    segments(ts, cams, obs)
    rows = ["| id | status | from | size mm² | height mm | what happened |", "|---|---|---|---|---|---|"]
    for t in sorted(ts, key=lambda t: t.id):
        st = "one object ✓" if t.status == "one" else "blob: untested ?"
        rows.append(f"| #{t.id} | {st} | {f'#{t.parent}' if t.parent else ''} | {t.area} | {t.height_mm:.0f} | "
                    f"{'; '.join(t.history[-2:])} |")
    n1 = sum(t.status == "one" for t in ts)
    roster(f"**The robot's world model**: {len(ts)} things, {n1} confirmed as one object, {len(ts) - n1} untested\n\n"
           + "\n".join(rows))


def clear_plan() -> None:
    rr.log("mind/plan", rr.Clear(recursive=True))


def plan_grip(good, bad, chosen, place_xy=None) -> None:
    """Candidate grips as finger pairs (faint green = possible, red = blocked), the chosen one bright, and the drop spot."""
    clear_plan()

    def fingers(gs, color):
        centers, halves, rots = [], [], []
        for g in gs:
            a = np.radians(g.yaw)
            d = np.array([np.cos(a), np.sin(a), 0.0])
            for side in (-1, 1):
                c = np.array([g.x, g.y, SURFACE_Z + g.height_mm / 1000 + 0.038]) + d * side * 0.009
                centers.append(c)
                halves.append([0.003, 0.006, 0.038])
                rots.append(rr.RotationAxisAngle(axis=[0, 0, 1], radians=a))
        return rr.Boxes3D(centers=centers, half_sizes=halves, rotation_axis_angles=rots, colors=[color] * len(centers))

    if bad:
        rr.log("mind/plan/blocked_grips", fingers(bad, [220, 60, 60, 70]))
    if good:
        rr.log("mind/plan/possible_grips", fingers(good, [60, 200, 90, 60]))
    rr.log("mind/plan/chosen_grip", fingers([chosen], [40, 255, 120, 230]))
    if place_xy is not None:
        a = np.linspace(0, 2 * np.pi, 40)
        ring = np.stack([place_xy[0] + 0.012 * np.cos(a), place_xy[1] + 0.012 * np.sin(a), np.full_like(a, SURFACE_Z + 0.001)], -1)
        rr.log("mind/plan/drop_spot", rr.LineStrips3D([ring], colors=[[255, 255, 255]], radii=0.0012, labels=["drop here"]))


def plan_rake(start_xy, end_xy, height_mm) -> None:
    clear_plan()
    z = SURFACE_Z + height_mm / 1000
    rr.log("mind/plan/rake", rr.Arrows3D(origins=[[start_xy[0], start_xy[1], z]],
                                         vectors=[[end_xy[0] - start_xy[0], end_xy[1] - start_xy[1], 0]],
                                         colors=[[255, 150, 40]], radii=0.003, labels=["rake"]))


# ---------------------------------------------------------------- animation (sim)

def set_sim_time(t: float) -> None:
    rr.set_time("sim_time", duration=float(t))


def truth_setup(cell) -> None:
    """Log every object's and the gripper's mesh once, in its own frame; animation then only moves their frames."""
    from lab.catalog import COLORS
    import trimesh
    for i in cell.instances:
        m = cell.local[i.id]
        rr.log(f"truth/objects/{i.id}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                  vertex_normals=m.vertex_normals, albedo_factor=list(COLORS[i.color]) + [255]),
               static=True)
    t, w, L = 0.006, 0.012, 0.076
    finger = trimesh.creation.box((t, w, L))
    finger.apply_translation([0, 0, L / 2])
    palm = trimesh.creation.box((0.05, 0.04, 0.03))
    palm.apply_translation([0, 0, L + 0.015])
    for root in ("truth", "mind"):
        for name, m, col in (("finger_l", finger, [170, 170, 178]), ("finger_r", finger, [170, 170, 178]), ("palm", palm, [70, 70, 75])):
            rr.log(f"{root}/gripper/{name}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                       vertex_normals=m.vertex_normals, albedo_factor=col + [255]), static=True)


class Animator:
    """on_frame callback for SimCell.play: moves the true objects, the gripper and the arm every simulated 0.1 s."""

    def __init__(self, cell, every: int = 1):
        self.cell, self.every, self.n = cell, every, 0
        self.q = np.array(cell.q, float)

    def __call__(self, cell) -> None:
        self.n += 1
        if self.n % self.every:
            return
        from lab.synth import wrist_ik
        set_sim_time(cell.t)
        tops = []
        for i in cell.instances:
            T = cell.body_pose(f"obj{i.id}")
            rr.log(f"truth/objects/{i.id}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            tops.append(T[:3, 3] + [0, 0, 0.022])
        rr.log("truth/labels", rr.Points3D(tops, radii=0.0001, labels=[i.name for i in cell.instances]))
        for root in ("truth", "mind"):
            for name, body in (("finger_l", "finger_l"), ("finger_r", "finger_r"), ("palm", "hand")):
                T = cell.body_pose(body)
                rr.log(f"{root}/gripper/{name}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        try:
            self.q = wrist_ik(cell.tip(), np.degrees(cell.actual_yaw()), q0=self.q)
        except ValueError:
            pass
        joints = np.append(self.q, cell.grip)
        arm("truth", joints)
        arm("mind", joints)
