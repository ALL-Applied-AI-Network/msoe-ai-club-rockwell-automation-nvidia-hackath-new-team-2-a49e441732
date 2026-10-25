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
import re
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
    raw += [rrb.Spatial2DView(name="height map", origin="heightmap"),
            rrb.Spatial2DView(name="VLM close-up", origin="vlm_crop")]
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
        collapse_panels=False,  # keep the entity tree, selection and timeline panels visible
    )


def app_blueprint() -> rrb.Blueprint:
    """The chat page's view: only what a person can take in. The robot's mind (big, aimed at the table), every sensor
    in a strip under it, one log of everything it thinks and does. The truth, the object roster and Rerun's own
    panels are a click away, not on screen."""
    eye = rrb.EyeControls3D(position=(0.88, 0.02, 0.50), look_target=(0.36, -0.08, 0.0), eye_up=(0, 0, 1))
    sensors = [rrb.Spatial2DView(name=f"{c} camera", origin=f"cams/{c}/rgb") for c in ("table", "side", "wrist")]
    sensors += [rrb.Spatial2DView(name=f"{c} depth", origin=f"cams/{c}/depth") for c in ("table", "wrist")]
    sensors = [rrb.Spatial2DView(name="evidence (top view)", origin="evidence_top")] + sensors
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Tabs(rrb.Spatial3DView(name="Robot's mind", origin="mind", eye_controls=eye),
                         rrb.Spatial3DView(name="World (truth)", origin="truth", eye_controls=eye), active_tab=0),
                rrb.Grid(*sensors, grid_columns=3),
                row_shares=[3, 2],
            ),
            rrb.Tabs(rrb.TextLogView(name="log", origin="/", contents=["/vlm", "/narration"]),
                     rrb.TextDocumentView(name="objects", origin="roster"), active_tab=0),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def start(name: str, save_to: Path | None = None, spawn: bool = False) -> None:
    """Begin a recording: to a .rrd file, a live viewer, or both. Logs the static scenery in both 3D views."""
    rr.init(re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_"))  # Rerun renames ids with spaces or brackets: keep them plain
    sinks = []
    global LOG
    LOG = None
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        sinks.append(rr.FileSink(str(save_to)))
        LOG = open(Path(save_to).parent / "narration.txt", "w", buffering=1)  # the text panels, greppable
    if spawn:
        rr.spawn(connect=False)  # starts a viewer if none is running
        sinks.append(rr.GrpcSink())
    if sinks:
        rr.set_sinks(*sinks, default_blueprint=blueprint())  # file AND viewer: the file is always complete
    _scenery()


def _scenery() -> None:
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


def start_served(name: str, save_to: Path | None = None, grpc_port: int = 9876, web_port: int = 9090) -> str:
    """Like start(), but the recording is served: a web viewer (http://localhost:web_port, for the chat page) and any
    number of native viewers (`rerun <uri>`) can watch it live. Returns the gRPC URI."""
    global LOG
    rr.init(re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_"))
    sinks = [rr.GrpcServerSink(bind_ip="127.0.0.1", port=grpc_port)]
    LOG = None
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        sinks.append(rr.FileSink(str(save_to)))
        LOG = open(Path(save_to).parent / "narration.txt", "w", buffering=1)
    rr.set_sinks(*sinks, default_blueprint=app_blueprint())
    _scenery()
    uri = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
    rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
    return uri


def clear_scene() -> None:
    """Everything the robot believed and saw, gone from the viewer (a fresh start); the static scenery comes back."""
    for path in ("mind", "truth", "cams", "evidence_top", "heightmap", "vlm_crop", "narration", "vlm"):
        rr.log(path, rr.Clear(recursive=True))
    _scenery()


def set_step(i: int) -> None:
    rr.set_time("step", sequence=i)


PRINT = False  # also print narration to the terminal (lab.trial --print)


LOG = None  # narration.txt next to the recording
LISTENERS: list = []  # f(panel, text, level), called for every narration / VLM line (the chat follows the robot)


def _log(panel: str, text: str, level: str) -> None:
    if LOG is not None:
        LOG.write(f"{panel:9s} {level:5s} {text}\n")
    for f in LISTENERS:
        try:
            f(panel, text, level)
        except Exception:
            pass


def narrate(text: str, level: str = "INFO") -> None:
    rr.log("narration", rr.TextLog(text, level=level))
    _log("narration", text, level)
    if PRINT:
        print(("   " if level == "DEBUG" else "") + text, flush=True)


def vlm(text: str, level: str = "INFO") -> None:
    rr.log("vlm", rr.TextLog(text, level=level))
    _log("vlm", text, level)


def label_crop(r: dict) -> None:
    """The latest picture sent to the VLM (close-up | where it is), captioned with its answer."""
    a = r.get("answer") or {}
    img = r["picture"].copy()
    cap = f"#{r['id']}: " + (f"{a.get('color', '')} {a.get('name')} ({float(a.get('confidence') or 0):.0%})" if a else "no answer")
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, cap, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    rr.log("vlm_crop", rr.Image(img).compress(jpeg_quality=85))


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
    _, keep = np.unique(np.floor(pts[:, :3] / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[keep]


# ---------------------------------------------------------------- robot

def arm(root: str, joints, tcp_offset: float = 0.293) -> None:
    """The arm as a stick figure from forward kinematics, base to wrist."""
    R, p, pts = np.eye(3), np.zeros(3), [np.zeros(3)]
    for (off, axis, sign), q in zip(CHAIN, np.radians(np.asarray(joints, float)[:6])):
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q)
        pts.append(p.copy())
    # stop at the wrist (flange): below it is the gripper, drawn as itself
    rr.log(f"{root}/arm", rr.LineStrips3D([np.array(pts)], colors=[[240, 200, 40]], radii=0.005))


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
        p = thin(s.points, 0.001)
        seen = p[:, 3:6].astype(np.uint8) if p.shape[1] >= 6 else col  # the colours the camera saw, if we have them
        rr.log(f"mind/segments/{s.id}", rr.Points3D(p[:, :3], colors=seen, radii=0.0009))
        tops.append(s.points[np.argmax(s.points[:, 2]), :3] + [0, 0, 0.008])
        tags.append(getattr(s, "tag", f"#{s.id}"))
        cols.append(col)
        if cams is None or obs is None:
            continue
        for cam_name, img in (obs.get("images") or {}).items():
            uv = project_to_image(cam_name, cams, thin(s.points, 0.002)[:, :3], (img.shape[1], img.shape[0]), joints=obs.get("joints"))
            ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] > 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] > 0) & (uv[:, 1] < img.shape[0])
            if ok.sum() < 3:
                continue
            hull = cv2.convexHull(uv[ok].astype(np.float32)).reshape(-1, 2)
            rr.log(f"cams/{cam_name}/rgb/segments/{s.id}", rr.LineStrips2D([np.vstack([hull, hull[:1]])], colors=col,
                                                                           radii=0.8, labels=[f"#{s.id}"]))
    # all id tags in one entity: click its eye icon (left panel, "mind/labels") to hide or show them all
    rr.log("mind/labels", rr.Points3D(tops, colors=cols, radii=0.0015, labels=tags) if tops else rr.Clear(recursive=False))


# ---------------------------------------------------------------- the robot's things (world model)

STATUS_MARK = {"blob": "?", "one": "✓", "stuck": "⊘", "bin": "▣ box"}


def things(world, cams: dict | None = None, obs: dict | None = None) -> None:
    """The robot's world model: each thing in its id colour, tagged "#7 ?" (untested blob) or "#7 ✓" (moved as one)."""
    ts = list(world.things.values())
    for t in ts:
        name = (getattr(t, "label", None) or {}).get("name")
        t.tag = f"#{t.id} {STATUS_MARK.get(t.status, t.status)}" + (f" {name}" if name else "")
    segments(ts, cams, obs)
    # confirmed things with a model: show the solid object (in the colour the camera saw) instead of its points
    rr.log("mind/objects", rr.Clear(recursive=True))
    for t in ts:
        if getattr(t, "model", None) is None:
            continue
        m = t.model.world_mesh()
        col = [int(c) for c in t.model.colour] if t.model.colour is not None else id_color(t.id)
        rr.log(f"mind/objects/{t.id}", rr.Mesh3D(vertex_positions=m.vertices, triangle_indices=m.faces,
                                                 vertex_normals=m.vertex_normals, albedo_factor=col + [255]))
        rr.log(f"mind/segments/{t.id}", rr.Clear(recursive=False))
    # where each thing is headed: an arrow to its box
    arrows = [(t.centroid, world.things[t.dest].centroid, t.dest) for t in ts
              if getattr(t, "dest", None) in world.things and str(getattr(t, "plan", "")).startswith("sort")]
    if arrows:
        rr.log("mind/plan_boxes", rr.Arrows3D(origins=[a[0] + [0, 0, 0.02] for a in arrows],
                                              vectors=[(b - a) * [1, 1, 0] for a, b, _ in arrows],
                                              colors=[id_color(d) for _, _, d in arrows], radii=0.0008))
    else:
        rr.log("mind/plan_boxes", rr.Clear(recursive=False))
    rows = ["| id | status | VLM says | group | plan | model | from | size mm² | height mm | what happened |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(ts, key=lambda t: t.id):
        st = {"one": "one object ✓", "stuck": "stuck ⊘", "bin": "box ▣"}.get(t.status, "blob: untested ?")
        md = f"{t.model.kind}, {t.model.world_mesh().volume * 1e9:.0f} mm³" if getattr(t, "model", None) else ""
        lab = getattr(t, "label", None) or {}
        said = (f"{lab.get('color', '')} {lab.get('name', '')} ({float(lab.get('confidence') or 0):.0%}"
                + (f", {lab.get('count')} things" if int(lab.get('count') or 1) > 1 else "") + ")") if lab else ""
        if getattr(t, "colour", None):  # boxes: the colour measured from the camera, which the director is told
            said += f" · camera: {t.colour}"
        rows.append(f"| #{t.id} | {st} | {said} | {getattr(t, 'group', '') or ''} | {getattr(t, 'plan', '') or ''} | {md} | {f'#{t.parent}' if t.parent else ''} | {t.area} | {t.height_mm:.0f} | "
                    f"{'; '.join(t.history[-2:])} |")
    n1, n2 = sum(t.status == "one" for t in ts), sum(t.status == "stuck" for t in ts)
    done = getattr(world, "sorted", [])
    sorted_md = ("\n\n**Sorted**: " + "; ".join(f"#{d['id']} {d.get('color') or ''} {d.get('name') or '?'} → box #{d['bin']}"
                                               + ("" if d.get("whole", True) else " (part)") for d in done)) if done else ""
    nb = sum(t.status == "bin" for t in ts)
    roster(f"**The robot's world model**: {len(ts) - nb} things, {n1} confirmed as one object, {n2} stuck, "
           f"{len(ts) - n1 - n2 - nb} untested" + (f", {nb} boxes" if nb else "") + "\n\n"
           + "\n".join(rows) + sorted_md)


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
                c = np.array([g.x, g.y, SURFACE_Z + g.height_mm / 1000 + 0.038]) + d * side * 0.012
                centers.append(c)
                halves.append([0.006, 0.003, 0.038])
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
    from lab import gripper as G
    parts = [("finger_l", G.finger_body_mesh(-1), [175, 175, 180]), ("finger_r", G.finger_body_mesh(1), [175, 175, 180])]
    parts += [(f"body{k}", m, list(c)) for k, (m, c) in enumerate(G.body_meshes())]
    for root in ("truth", "mind"):
        for name, m, col in parts:
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
        from calibration.kinematics import tool_ik
        set_sim_time(cell.t)
        tops = []
        for i in cell.instances:
            T = cell.body_pose(f"obj{i.id}")
            rr.log(f"truth/objects/{i.id}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            tops.append(T[:3, 3] + [0, 0, 0.022])
        rr.log("truth/labels", rr.Points3D(tops, radii=0.0001, labels=[i.name for i in cell.instances]))
        for root in ("truth", "mind"):
            for name, body in (("finger_l", "finger_l"), ("finger_r", "finger_r"), ("body0", "hand"), ("body1", "hand")):
                T = cell.body_pose(body)
                rr.log(f"{root}/gripper/{name}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        try:
            wpr = cell.actual_wpr() if hasattr(cell, "actual_wpr") else (-180.0, 0.0, np.degrees(cell.actual_yaw()))
            self.q = tool_ik(cell.tip(), wpr, q0=self.q)
        except ValueError:
            pass
        joints = np.append(self.q, cell.grip)
        arm("truth", joints)
        arm("mind", joints)


def unseen(cells: np.ndarray, hm_shape) -> None:
    """Cells the robot hasn't seen but cares about, as purple dots on the table (mind view)."""
    from lab.fusion import CELL, X_RANGE, Y_RANGE
    if len(cells) == 0:
        rr.log("mind/unseen", rr.Clear(recursive=False))
        return
    ix, iy = np.divmod(cells, hm_shape[1])
    pts = np.column_stack([X_RANGE[0] + (ix + 0.5) * CELL, Y_RANGE[0] + (iy + 0.5) * CELL, np.full(len(ix), SURFACE_Z + 0.0005)])
    rr.log("mind/unseen", rr.Points3D(pts, colors=[[170, 60, 230]], radii=0.0006))


def next_view(v) -> None:
    """The chosen next camera pose, as a small camera cone in the mind view, labelled with what it should reveal."""
    from calibration.geometry import load
    cam = load()["wrist_depth"]
    camera_frustum("mind/plan/next_view", cam, v.cam)
    rr.log("mind/plan/next_view_label", rr.Points3D([v.cam[:3, 3]], colors=[[170, 60, 230]], radii=0.003,
                                                    labels=[f"next look: +{v.gain} mm²"]))
