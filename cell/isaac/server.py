"""Isaac Sim behind the real cell's HTTP API: the policy can't tell it from the cell (or from lab's MuJoCo cell).

Runs on the 4090 box, with its window on that box's monitor (the truth), while the policy runs on the Mac and Rerun
shows the robot's mind:

    # on the server (isaac/run_server.sh does this):
    DISPLAY=:1 ~/isaacsim/python.sh isaac/server.py            # --headless for no window
    # on the Mac:
    uv run python -m lab.trial run sort_mixed --policy sort --cell isaac

The scene is the hackathon twin's (TWIN, default ~/fanuc-er4ia-sim): table, ER-4iA, the measured gripper with its
contact physics. Everything the policy sees is built from OUR calibration (calibration/cameras.json), so pixels and
depth line up with the real cell's, lens distortion included (rendered as pinhole, then warped to the k1 model).

The real API (cell_client.py), same shapes:
    GET  /api/v1/time                       {"t"}
    GET  /api/v1/run                        {"trials": [...]}        one trial per /sim/reset
    POST /api/v1/trials/<id>/start
    GET  /api/v1/observation?cameras=table,wrist&width=&height=&format=png&depth=table,wrist
    POST /api/v1/actions {"tcp": [[x, y, z mm, w, p, r deg, grip], ...], "dt": 0.1}   (UF0, fingertip; or "joints")
Sim-only extras (lab/isaaccell.py uses them):
    POST /api/v1/sim/reset {"objects": [...], "instruction"}   new trial: these objects at these poses (base frame)
    GET  /api/v1/sim/wait                   blocks until every queued action has played; returns the per-row truth
    GET  /api/v1/sim/truth                  every object's pose now
    POST /api/v1/sim/end                    end the trial (run.py's loop stops)
Joel's hands (a person changing the table mid-trial; the robot isn't told, it has to notice):
    POST /api/v1/sim/add {"objects": [...]}           new objects (usually above the table: a cup poured out)
    POST /api/v1/sim/remove {"ids": [...]}            take objects away
    POST /api/v1/sim/move {"poses": {id: pose}}       put objects somewhere else (a box tipped out: its contents)

Time is simulated: each 0.1 s row is 0.1 s of physics, however long it takes to compute. Idle, nothing moves.
"""
from __future__ import annotations

import argparse
import base64
import faulthandler
import signal
import collections
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> prints every thread's stack (a hang)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # cell/: calibration
TWIN = Path(os.environ.get("TWIN", "~/fanuc-er4ia-sim")).expanduser()
sys.path.insert(0, str(TWIN))

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8765)
p.add_argument("--host", default="127.0.0.1", help="127.0.0.1: reach it through an SSH tunnel")
p.add_argument("--headless", action="store_true")
p.add_argument("--physics-hz", type=int, default=240)
args = p.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1600, "height": 1000})

import cv2  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation, XFormPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import cell as twin  # noqa: E402
import scene_config as C  # noqa: E402
from calibration.geometry import UF0_Z_MM, load as load_cams  # noqa: E402
from calibration.kinematics import HOME, tcp_of, tool_ik  # noqa: E402

JUMP_DEG = 20.0  # a row whose joints are this far from the last row's: a wrist flip the controller would refuse
TORSION_M = 0.004  # torsional friction patch radius on objects (m): a finger pad's contact, roughly
BLOCKED_MM = 12.0  # the fingertip this far from where it was told to be after a row: something is in the way (a fault)
BASE_Z = C.TABLE_TOP_Z  # the robot base in Isaac's world: base frame = world - (0, 0, 0.90)
TCP_BOX = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]  # the cell's limits, mm in UF0
APERTURE = 20.955  # mm; any value works, the focal length is scaled to match
OBJ_ROOT = "/World/Objects"


# ---------------------------------------------------------------- cameras: ours, from cameras.json

def _usd_matrix(T: np.ndarray) -> Gf.Matrix4d:
    """An OpenCV-convention camera pose (x right, y down, z forward) as a USD camera transform (looks down -z, y up)."""
    M = T @ np.diag([1.0, -1.0, -1.0, 1.0])
    return Gf.Matrix4d(*M.T.ravel().tolist())  # USD matrices are row-vector: transpose


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


class Cam:
    """One calibrated camera: a pinhole render at native size, warped to the calibration's own model."""

    def __init__(self, stage, name: str, cal: dict):
        self.name, self.cal = name, cal
        W, H = cal["size"]
        self.size = (W, H)
        self.depth = bool(cal.get("depth"))
        fx = cal["fx"]
        fy = cal.get("fy", fx)
        wide = 1.0 if self.depth else 0.94  # colour: render a little wider, so the distortion warp has margin at the edges
        # the wrist cameras ride on J6_link (the same frame as our FK's flange), like the gripper meshes do
        T = cal["T"].copy()
        if cal["frame"] == "base":
            path = f"/World/CalCams/{name}"
            T[2, 3] += BASE_Z
        else:
            path = f"{twin.FLANGE}/CalCam_{name}"
        cam = UsdGeom.Camera.Define(stage, path)
        self.op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
        self.op.Set(_usd_matrix(T))
        cam.CreateFocalLengthAttr(fx * wide * APERTURE / W)
        cam.CreateHorizontalApertureAttr(APERTURE)
        cam.CreateVerticalApertureAttr(APERTURE * H / W * fx / fy)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
        self.path = path
        # the pinhole render has its centre at (W/2, H/2) and focal fx; the calibration may add k1 or move the centre
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        if self.depth:
            x, y = (u - cal["cx"]) / fx, (v - cal["cy"]) / fy
        else:
            xd, yd = (u - W / 2) / fx, (v - H / 2) / fx
            x, y = xd.copy(), yd.copy()
            for _ in range(8):  # undo d = 1 + k1 r^2 by fixed-point iteration
                d = 1 + cal["k1"] * (x * x + y * y)
                x, y = xd / d, yd / d
        self.map = (x * fx * wide + W / 2).astype(np.float32), (y * fy * wide + H / 2).astype(np.float32)
        self.rp = None
        self.ann = None
        self.fabric = None

    def place(self, q_deg) -> None:
        pass  # the wrist cameras follow J6_link by themselves

    def attach(self):
        self.rp = rep.create.render_product(self.path, self.size)
        self.ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane" if self.depth else "rgb")
        self.ann.attach([self.rp])

    def read(self) -> np.ndarray:
        """RGB uint8 (H, W, 3), or depth uint16 mm (0 = none), in the calibration's own model."""
        a = np.asarray(self.ann.get_data())
        if self.depth:
            mm = np.where(np.isfinite(a) & (a > 0) & (a < 10), a * 1000, 0).astype(np.float32)
            return cv2.remap(mm, *self.map, cv2.INTER_NEAREST).astype(np.uint16)
        return cv2.remap(np.ascontiguousarray(a[..., :3]), *self.map, cv2.INTER_LINEAR)


# ---------------------------------------------------------------- the cell

class Cell:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / args.physics_hz, rendering_dt=1 / 30)
        self.stage = omni.usd.get_context().get_stage()
        twin.build(self.stage, with_objects=False)
        cal = load_cams()
        self.cams = {n: Cam(self.stage, n, c) for n, c in cal.items()}
        UsdGeom.Xform.Define(self.stage, OBJ_ROOT)
        twin.enable_ccd(self.world)
        self.world.reset()
        for c in self.cams.values():
            c.attach()
        self._arm()
        self.lock = threading.Lock()
        self.rows: collections.deque = collections.deque()  # (row, dt, kind)
        self.jobs: collections.deque = collections.deque()  # (fn, args, event, box) to run on the main thread
        self.frames: list = []
        self.t = 0.0
        self.playing = False
        self.objects: list[dict] = []
        self.fault = None
        self.taken: set[int] = set()
        self.looks: dict = {}
        self.view = None
        self.trial = None
        self.n_trials = 0
        for _ in range(30):  # let the MDL materials compile
            self.world.render()

    def _arm(self):
        self.arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        self.arm.initialize()
        names = list(self.arm.dof_names)
        self.arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        self.jaw_idx = {j: names.index(j) for j in twin.JAW_JOINTS.values()}
        self.q = np.array(HOME, float)
        self.grip = 0.0
        full = self.arm.get_joint_positions().copy()
        full[self.arm_idx] = np.radians(self.q)
        for j, v in twin.jaw_targets(C.JAW_OPEN).items():
            full[self.jaw_idx[j]] = v
        self.arm.set_joint_positions(full)
        for _ in range(10):  # physics steps, so the links (and the render) are really at home
            self._command(self.q, 0.0)
            self.world.step(render=False)

    def _command(self, q_deg, grip: float) -> None:
        full = self.arm.get_joint_positions().copy()
        full[self.arm_idx] = np.radians(q_deg)
        for j, v in twin.jaw_targets(C.JAW_CLOSED if grip >= 0.5 else C.JAW_OPEN).items():
            full[self.jaw_idx[j]] = v
        self.arm.apply_action(ArticulationAction(joint_positions=full))

    # ------------------------------------------------------------ objects

    def reset(self, objects: list[dict], instruction: str) -> dict:
        """A new trial: these objects, at these poses (base frame), then a second of physics to settle."""
        self.world.stop()
        if self.stage.GetPrimAtPath(OBJ_ROOT).IsValid():
            self.stage.RemovePrim(OBJ_ROOT)
        UsdGeom.Xform.Define(self.stage, OBJ_ROOT)
        self.looks = {}
        self.taken = set()  # ids in Joel's pocket
        for o in objects:
            self._add(o, self.looks)
        self.objects = list(objects)
        self.world.reset()
        self._arm()
        self._new_view()
        with self.lock:
            self.rows.clear()
            self.frames = []
        self.t = 0.0
        for _ in range(args.physics_hz):  # settle
            self._command(self.q, 0.0)
            self.world.step(render=False)
        self.world.render()
        self.n_trials += 1
        self.trial = {"trial_id": f"isaac-{self.n_trials:03d}", "index": self.n_trials - 1, "layout": "sim",
                      "status": "running", "instruction": instruction, "scored": False}
        return {"trial": self.trial, "truth": self.truth()}

    def _new_view(self) -> None:
        """The physics view the truth is read through: rebuilt whenever objects come or go."""
        self.n_views = getattr(self, "n_views", 0) + 1
        self.view = None
        if self.objects:
            self.view = RigidPrim(prim_paths_expr=[f"{OBJ_ROOT}/obj_{o['id']}" for o in self.objects], name=f"objs{self.n_views}")
            self.view.initialize()

    # ------------------------------------------------------------ Joel's hands: changes to the table, mid-trial

    def add(self, objects: list[dict], fall_s: float = 1.5) -> dict:
        """New objects (poses in the base frame, usually above the table: a cup poured out), then let them fall. The
        robot's clock doesn't move: this is a person's hand, not the robot's time."""
        for o in objects:
            self._add(o, self.looks)
        self.objects += objects
        self.world.step(render=False)  # physics picks up the new bodies
        self._new_view()
        self._settle(fall_s)
        return {"truth": self.truth()}

    def remove(self, ids: list[int]) -> dict:
        """Take objects off the table (a hand picks them up). They go to "Joel's pocket", on the floor well away from
        the cell and every camera: deleting a body mid-simulation would invalidate the physics views (the arm's too)."""
        idx = {int(o["id"]): k for k, o in enumerate(self.objects)}
        ks = [idx[int(i)] for i in ids if int(i) in idx and int(i) not in self.taken]
        if ks:
            p = np.array([[-2.0 - 0.1 * (len(self.taken) + n), 2.0, 0.05] for n in range(len(ks))])
            q = np.tile([1.0, 0, 0, 0], (len(ks), 1))
            self.view.set_world_poses(positions=p, orientations=q, indices=np.array(ks))
            self.view.set_velocities(np.zeros((len(ks), 6)), indices=np.array(ks))
            self.taken |= {int(self.objects[k]["id"]) for k in ks}
        self._settle(0.2)
        return {"truth": self.truth()}

    def move(self, poses: dict) -> dict:
        """Put objects somewhere else, by hand: {id: [x, y, z (base, m), qw, qx, qy, qz]}; then let them fall."""
        idx = {str(o["id"]): k for k, o in enumerate(self.objects)}
        ks = [idx[str(i)] for i in poses]
        p = np.array([poses[i][:3] for i in poses], float) + [0, 0, BASE_Z]
        q = np.array([poses[i][3:7] for i in poses], float)
        self.view.set_world_poses(positions=p, orientations=q, indices=np.array(ks))
        self.view.set_velocities(np.zeros((len(ks), 6)), indices=np.array(ks))
        self._settle(1.5)
        return {"truth": self.truth()}

    def _settle(self, seconds: float) -> None:
        for _ in range(int(seconds * args.physics_hz)):
            self._command(self.q, self.grip)
            self.world.step(render=(_ % 8 == 0) and not args.headless)

    def _look(self, rgb, looks: dict):
        key = tuple(int(c) for c in rgb)
        if key not in looks:
            path = f"{OBJ_ROOT}/Looks/c_{key[0]}_{key[1]}_{key[2]}"
            mat = UsdShade.Material.Define(self.stage, path)
            sh = UsdShade.Shader.Define(self.stage, path + "/s")
            sh.CreateIdAttr("UsdPreviewSurface")
            c = np.array(key) / 255.0
            lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)  # our colours are sRGB; shaders want linear
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*lin.tolist()))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            looks[key] = mat
        return looks[key]

    def _add(self, o: dict, looks: dict) -> None:
        """A rigid body: a convex-hull mesh, or (boxes) a set of cuboids, since a hull would fill the well."""
        path = f"{OBJ_ROOT}/obj_{o['id']}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        pos = np.array(o["pos"], float) + [0, 0, BASE_Z]
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        w, x, y, z = o["quat"]
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        mat = self._look(o["rgb"], looks)
        shapes = []
        if o.get("boxes"):
            for k, b in enumerate(o["boxes"]):
                cube = UsdGeom.Cube.Define(self.stage, f"{path}/part{k}")
                cube.CreateSizeAttr(1.0)
                cx = UsdGeom.Xformable(cube.GetPrim())
                cx.AddTranslateOp().Set(Gf.Vec3d(*b["center"]))
                cx.AddScaleOp().Set(Gf.Vec3f(*b["size"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                shapes.append(cube.GetPrim())
        else:
            mesh = UsdGeom.Mesh.Define(self.stage, f"{path}/geom")
            v, f = np.asarray(o["vertices"], float), np.asarray(o["faces"], int)
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
            mesh.CreateFaceVertexCountsAttr([3] * len(f))
            mesh.CreateFaceVertexIndicesAttr(f.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            shapes.append(mesh.GetPrim())
        for s in shapes:
            UsdShade.MaterialBindingAPI.Apply(s).Bind(mat)
            twin.bind_physics_material(s)
            twin.small_contact_offsets(s)
            # torsional friction: PhysX's default has none, so a part held between two pads spins freely about the
            # line between them (a box held by its wall hung like a pendulum). A pad ~ a few mm across resists that.
            col = PhysxSchema.PhysxCollisionAPI.Apply(s)
            col.CreateTorsionalPatchRadiusAttr().Set(TORSION_M)
            col.CreateMinTorsionalPatchRadiusAttr().Set(TORSION_M / 2)
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(o["mass"]))
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        px.CreateAngularDampingAttr().Set(0.08)
        px.CreateLinearDampingAttr().Set(0.02)
        px.CreateSolverPositionIterationCountAttr().Set(16)
        px.CreateSolverVelocityIterationCountAttr().Set(2)
        px.CreateEnableCCDAttr().Set(True)

    def truth(self) -> dict:
        """{id: [x, y, z (base, m), qw, qx, qy, qz]}"""
        if self.view is None or not self.objects:
            return {}
        pos, quat = self.view.get_world_poses()
        pos, quat = np.asarray(pos), np.asarray(quat)
        return {str(o["id"]): [*(pos[k] - [0, 0, BASE_Z]).tolist(), *quat[k].tolist()] for k, o in enumerate(self.objects)
                if int(o["id"]) not in self.taken}

    # ------------------------------------------------------------ motion

    def play_row(self, row, dt: float, kind: str) -> None:
        if kind == "tcp":
            tip = np.array([row[0], row[1], row[2] - UF0_Z_MM]) / 1000
            try:
                goal = tool_ik(tip, row[3:6], self.q)
            except ValueError as e:
                print(f"[isaac-cell] {e}; holding", flush=True)
                goal = self.q
            jump = np.abs(goal - self.q)
            if jump.max() > JUMP_DEG:  # the real controller faults on a move that needs the wrist to flip
                with self.lock:
                    dropped = len(self.rows)
                    self.rows.clear()
                    self.fault = (f"refused: the next pose needs J{int(jump.argmax()) + 1} to turn {jump.max():.0f} deg in "
                                  f"{dt:.1f} s (a wrist flip); stopped, {dropped} queued rows dropped")
                print(f"[isaac-cell] FAULT {self.fault}", flush=True)
                goal = self.q
        else:
            goal = np.asarray(row[:6], float)
        grip = float(row[6]) if len(row) > 6 else self.grip
        n = max(1, int(round(dt * args.physics_hz)))
        every = max(1, n // 3)  # ~30 frames a second of simulated time in the window
        q0 = self.q.copy()
        for s in range(1, n + 1):
            self._command(q0 + (goal - q0) * s / n, grip)
            self.world.step(render=(s % every == 0) and not args.headless)
        self.t += dt
        lag = np.linalg.norm(tcp_of(self.joints_now()[:6])[0] - tcp_of(goal)[0]) * 1000
        self.max_lag = max(getattr(self, "max_lag", 0.0), lag)
        if lag > BLOCKED_MM:  # blocked: the real cell would fault here instead of pushing harder. Stop where it is.
            self.q = self.joints_now()[:6]
            self.grip = grip
            self._command(self.q, grip)
            with self.lock:
                dropped = len(self.rows)
                self.rows.clear()
                self.fault = (f"collision: the fingertip is {lag:.0f} mm from its target at t = {self.t:.1f} s "
                              f"(something is in the way); stopped, {dropped} queued rows dropped")
            print(f"[isaac-cell] FAULT {self.fault}", flush=True)
        else:
            self.q, self.grip = goal, grip
        tip, wpr = tcp_of(self.q)
        with self.lock:
            self.frames.append({"t": self.t, "poses": self.truth(), "tip": tip.tolist(), "wpr": wpr.tolist(), "grip": grip,
                                "joints": goal.tolist(), "fault": self.fault})
            self.fault = None

    def joints_now(self) -> np.ndarray:
        q = np.degrees(np.asarray(self.arm.get_joint_positions())[self.arm_idx])
        return np.append(q, self.grip)

    def observe(self, cameras, size, fmt, quality, depth) -> dict:
        want = [c for c in cameras if c in self.cams] + [f"{d}_depth" for d in depth if f"{d}_depth" in self.cams]
        t0 = time.time()
        q_now = self.joints_now()[:6]
        for c in want:
            self.cams[c].place(q_now)
        for _ in range(3):  # a few frames so every render product has the current pose
            self.world.render()
        t1 = time.time()
        out_img, out_depth = {}, {}
        for c in want:
            img = self.cams[c].read()
            if self.cams[c].depth:
                ok, buf = cv2.imencode(".png", img)
                out_depth[c.replace("_depth", "")] = base64.b64encode(buf.tobytes()).decode()
            else:
                img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
                ext, opts = (".jpg", [cv2.IMWRITE_JPEG_QUALITY, quality]) if fmt == "jpeg" else (".png", [])
                ok, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), opts)
                out_img[c] = base64.b64encode(buf.tobytes()).decode()
        q = self.joints_now()
        tip, wpr = tcp_of(q[:6])
        with self.lock:
            queued = len(self.rows)
        obs = {"t": self.t, "status": "running" if (queued or self.playing) else "idle", "queued": queued,
               "trial": self.trial, "joints": q.tolist(), "max_lag_mm": round(getattr(self, "max_lag", 0.0), 1),
               "tcp_pose": [tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, *wpr.tolist()],
               "images": out_img}
        if out_depth:
            obs["depth"] = out_depth
        print(f"[isaac-cell] observe: render {t1 - t0:.2f} s, read+encode {time.time() - t1:.2f} s", flush=True)
        return obs

    def debug(self, prims) -> dict:
        from pxr import Usd
        out = {"joints": self.joints_now().tolist(), "dof_names": list(self.arm.dof_names)}
        for path in [twin.FLANGE, *[c.path for c in self.cams.values()], *[p for p in prims if p]]:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                out[path] = np.array(M).T.tolist()
        return out

    # ------------------------------------------------------------ the main loop (Isaac must run on this thread)

    def run_job(self, fn, *a):
        box, ev = {}, threading.Event()
        with self.lock:
            self.jobs.append((fn, a, ev, box))
        ev.wait()
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["value"]

    def loop(self):
        while app.is_running():
            with self.lock:
                job = self.jobs.popleft() if self.jobs else None
                row = self.rows.popleft() if (job is None and self.rows) else None
                self.playing = row is not None
            if job is not None:
                fn, a, ev, box = job
                print(f"[isaac-cell] job {fn.__name__}", flush=True)
                try:
                    box["value"] = fn(*a)
                except Exception as e:
                    traceback.print_exc()
                    box["error"] = f"{type(e).__name__}: {e}"
                ev.set()
            elif row is not None:
                self.play_row(*row)
                with self.lock:
                    self.playing = False
            else:
                self.world.render()  # idle: keep the window alive, no physics (time stands still)


# ---------------------------------------------------------------- HTTP

def handler(cell: Cell):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method: str):
            u = urlparse(self.path)
            path = u.path.removeprefix("/api/v1/").strip("/")
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            try:
                self._send(200, self._do(method, path, q, body))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _do(self, method, path, q, body):
            if path == "time":
                return {"t": cell.t}
            if path == "run":
                return {"trials": [cell.trial] if cell.trial else []}
            if path.startswith("trials/") and path.endswith("/start"):
                if cell.trial:
                    cell.trial["status"] = "running"
                return {}
            if path == "observation":
                cams = [c for c in q.get("cameras", "table").split(",") if c]
                size = (int(q.get("width", 640)), int(q.get("height", 360)))
                depth = [d for d in q.get("depth", "").split(",") if d]
                return cell.run_job(cell.observe, cams, size, q.get("format", "png"), int(q.get("quality", 90)), depth)
            if path == "actions":
                dt = float(body.get("dt", 0.1))
                if not 0.1 <= dt <= 1.0:
                    raise ValueError("dt must be in [0.1, 1.0]")
                kind = "tcp" if "tcp" in body else "joints"
                rows = np.atleast_2d(np.asarray(body[kind], float))
                if len(rows) > 100:
                    raise ValueError("at most 100 rows per chunk")
                if kind == "tcp":
                    for k, (lo, hi) in enumerate(TCP_BOX):
                        if rows[:, k].min() < lo or rows[:, k].max() > hi:
                            raise ValueError(f"tcp {'xyz'[k]} outside [{lo}, {hi}] mm (UF0)")
                with cell.lock:
                    if body.get("replace"):
                        cell.rows.clear()
                    cell.rows.extend((r, dt, kind) for r in rows.tolist())
                return {"queued": len(cell.rows)}
            if path == "sim/reset":
                return cell.run_job(cell.reset, body["objects"], body.get("instruction", ""))
            if path == "sim/wait":
                while True:
                    with cell.lock:
                        if not cell.rows and not cell.playing and not cell.jobs:
                            frames, cell.frames = cell.frames, []
                            return {"t": cell.t, "frames": frames}
                    time.sleep(0.01)
            if path == "sim/add":
                return cell.run_job(cell.add, body["objects"])
            if path == "sim/remove":
                return cell.run_job(cell.remove, body["ids"])
            if path == "sim/move":
                return cell.run_job(cell.move, body["poses"])
            if path == "sim/truth":
                return {"t": cell.t, "poses": cell.run_job(cell.truth)}
            if path == "sim/debug":  # world transforms (row-vector USD matrices, transposed to column form) of prims
                return cell.run_job(cell.debug, q.get("prims", "").split(","))
            if path == "sim/end":
                if cell.trial:
                    cell.trial["status"] = "done"
                return {}
            raise ValueError(f"no such endpoint: {method} /{path}")
    return H


def main():
    cell = Cell()
    srv = ThreadingHTTPServer((args.host, args.port), handler(cell))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[isaac-cell] ready on http://{args.host}:{args.port}/api/v1", flush=True)
    try:
        cell.loop()
    finally:
        srv.shutdown()
        app.close()


if __name__ == "__main__":
    main()
