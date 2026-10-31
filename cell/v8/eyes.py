"""V8 stage two: the dorsal stream. Shapes, not names.

A look is a wrist photo sent to SAM. Every mask claims a volume: the cone from the camera's
optical center through the mask, cut at the table. Evidence is three-valued per pixel (Joel's rule):

    mask                ->  something along those rays        (keep)
    in frame, not mask  ->  provably nothing along those rays (carve away)
    outside the frame   ->  unknown                           (never carve)

with one refinement for occlusion: a voxel hidden behind SOMEONE ELSE's mask is unknown too - that
ray hit the other thing, it proves nothing about the space behind it. A blob's shape is the
intersection of its shots' allowed regions, on a voxel grid resting on the table (gravity is the
one hardcoded prior: things sit on the table, so the grid starts there).

Only the WRIST camera carves, because only the wrist camera can be aimed. The mind decides WHAT
to look at; three_vantages() is motion planning deciding FROM WHERE (one top view, two low views
from different sides - the low ones are what height is made of - each checked for reach, each
placed so the blob fits in frame with margin; a cut-off mask makes the tool back up and retake).

No label, no name, no VLM anywhere in this file.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose, load
from calibration.kinematics import FLANGE_FROM_TOOL, TCP_OFFSET, tool_ik, wpr_of

WORDS = ["object", "small object"]   # measured: "object" is salience-hungry, "small object" catches sparse scenes
GRID_X = (0.20, 0.78)
GRID_Y = (-0.36, 0.36)
GRID_TOP = 0.14                      # m above the table: taller than anything in the cell
VOX = 0.004


class Grid:
    def __init__(self) -> None:
        self.xs = np.arange(GRID_X[0], GRID_X[1], VOX)
        self.ys = np.arange(GRID_Y[0], GRID_Y[1], VOX)
        self.zs = np.arange(SURFACE_Z + 0.002, SURFACE_Z + GRID_TOP, VOX)
        g = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.pts = np.stack([a.ravel() for a in g], -1)          # (N, 3) base frame, m
        self.shape = (len(self.xs), len(self.ys), len(self.zs))


@dataclass
class Blob:
    """An anonymous shape: a volume of space that evidence says is occupied. No name, ever (here)."""
    id: int
    keep: np.ndarray                 # bool over grid.pts: still possible after every shot so far
    shots: int = 1
    born: float = field(default_factory=time.time)

    def summary(self, grid: Grid) -> str:
        p = grid.pts[self.keep]
        if not len(p):
            return f"blob #{self.id}: carved away to nothing"
        x, y = p[:, 0].mean() * 1000, p[:, 1].mean() * 1000
        sx, sy = np.ptp(p[:, 0]) * 1000 + VOX * 1000, np.ptp(p[:, 1]) * 1000 + VOX * 1000
        top = (p[:, 2].max() - SURFACE_Z) * 1000
        sure = "height still a guess (one view)" if self.shots < 2 else f"seen from {self.shots} angles"
        return (f"blob #{self.id}: around ({x:.0f}, {y:.0f}) mm, footprint ~{sx:.0f} x {sy:.0f} mm, "
                f"up to {top:.0f} mm tall - {sure}")

    def centre(self, grid: Grid) -> np.ndarray | None:
        p = grid.pts[self.keep]
        return p.mean(0) if len(p) else None

    def extent(self, grid: Grid) -> float:
        p = grid.pts[self.keep]
        return float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2]))) if len(p) else 0.05


def aim(eye: np.ndarray, target: np.ndarray, T_flange_cam: np.ndarray,
        roll: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The (fingertip, wpr) that puts the wrist camera at eye looking at target - pure geometry,
    shared by the flyer and by offline checks so the tested math IS the flown math. roll (rad)
    spins the camera about its view axis: it changes nothing about what is framed, so it is a
    free parameter the arm may spend to make the pose reachable."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-3:                  # looking straight down: any sideways works
        x = np.array([0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    T_bc = np.eye(4)
    T_bc[:3, :3] = np.column_stack([x, y, z])
    T_bc[:3, 3] = eye
    T_bf = T_bc @ np.linalg.inv(T_flange_cam)
    tip = T_bf[:3, 3] + T_bf[:3, 0] * TCP_OFFSET
    return tip, wpr_of(T_bf[:3, :3] @ FLANGE_FROM_TOOL.T)


def ik_for(eye: np.ndarray, target: np.ndarray, T_flange_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tip, wpr) of a REACHABLE pose framing target from eye, spending camera roll to find one.
    Raises ValueError with the last miss when no roll works."""
    err: Exception | None = None
    for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4):
        tip, wpr = aim(eye, target, T_flange_cam, roll)
        try:
            tool_ik(tip, wpr, tol_mm=3.0)
            return tip, wpr
        except ValueError as e:
            err = e
    raise ValueError(f"no camera roll reaches that vantage: {err}")


def pixels_of(pts: np.ndarray, cam: dict, joints) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base-frame points -> (u, v in calibration pixels, in_front). Same pinhole + one radial term
    as calibration.geometry.project, done here so we also get the depth sign."""
    T = np.linalg.inv(flange_pose(joints) @ cam["T"])
    pc = pts @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = pc[:, 0] / z, pc[:, 1] / z
    d = 1 + cam["k1"] * (x * x + y * y)
    w, h = cam["size"]
    return cam["fx"] * x * d + w / 2, cam["fx"] * y * d + h / 2, z > 0.02


@dataclass
class Shot:
    """One wrist photo and what SAM said about it, with the pose it was ACTUALLY taken from."""
    k: int
    image: np.ndarray
    joints: np.ndarray
    masks: list[dict]                # [{"mask" bool HxW, "cut": touches the frame edge}]


class Eyes:
    """The dorsal stream over one cell. look() births blobs from wherever the camera is;
    inspect(id) aims three vantages at one blob and intersects the cones."""

    def __init__(self, cell, view=None) -> None:
        self.cell = cell
        self.view = view             # v8.view hooks; None is fine (headless)
        self.grid = Grid()
        self.cams = load()
        self.blobs: list[Blob] = []
        self._ids = itertools.count(1)
        self._shots = itertools.count(1)

    # ---------------- the evidence machinery ----------------

    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        """Just the picture and where it was taken from - no SAM yet, so flights can be batched
        (move, snap, move, snap; the thinking happens after the moving, Joel's one ok'd hardcode)."""
        obs = self.cell.observe(cams=("wrist",))
        return obs["images"]["wrist"], np.asarray(obs["joints"], float)

    def _see(self, img: np.ndarray, joints: np.ndarray) -> Shot:
        from lab import sam
        found = sam.segment_many(img, WORDS, scale=2.0)
        masks: list[dict] = []
        for d in sorted(found, key=lambda d: -d["area"]):     # dedupe: two words often find one thing
            m = d["mask"]
            if any((m & e["mask"]).sum() > 0.5 * min(m.sum(), e["mask"].sum()) for e in masks):
                continue
            edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
            masks.append({"mask": m, "cut": bool(edge)})
        shot = Shot(next(self._shots), img, joints, masks)
        if self.view:
            self.view.shot(shot, self.cams["wrist"], self.grid)
        return shot

    def _snap(self) -> Shot:
        img, joints = self._grab()
        return self._see(img, joints)

    def _carve(self, shot: Shot) -> list[Blob]:
        """Apply one shot to every blob (three-valued), then birth blobs for unclaimed masks."""
        cam = self.cams["wrist"]
        u, v, front = pixels_of(self.grid.pts, cam, shot.joints)
        w, h = cam["size"]
        mh, mw = (shot.masks[0]["mask"].shape if shot.masks else (720, 1280))
        ui = np.clip((u * mw / w), 0, mw - 1).astype(np.int32)
        vi = np.clip((v * mh / h), 0, mh - 1).astype(np.int32)
        in_frame = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cones = [m["mask"][vi, ui] & in_frame for m in shot.masks]
        any_mask = np.zeros(len(self.grid.pts), bool)
        for c in cones:
            any_mask |= c

        # which mask is which blob: the one whose cone overlaps the blob's surviving volume most.
        # (Projecting the blob's centre point missed side views - the centre sits high in the slack
        # volume - so a side view birthed a REPLACEMENT instead of intersecting. Overlap is the
        # 'views work together' rule, literally.)
        pairs = []
        for bi, b in enumerate(self.blobs):
            for ci, c in enumerate(cones):
                ov = int(np.count_nonzero(b.keep & c))
                if ov >= max(4, 0.05 * b.keep.sum()):
                    pairs.append((ov, bi, ci))
        claimed: set[int] = set()
        owner: dict[int, int] = {}
        for ov, bi, ci in sorted(pairs, reverse=True):
            if bi in owner or ci in claimed:
                continue
            owner[bi], claimed = ci, claimed | {ci}
        for bi, b in enumerate(self.blobs):
            own = owner.get(bi)
            if own is not None:
                b.keep &= cones[own] | ~in_frame | (any_mask & ~cones[own])   # mine, unknown, or occluded
                b.shots += 1
            else:
                b.keep &= ~in_frame | any_mask   # not seen here: clear table carves, other masks occlude

        born = []
        for i, m in enumerate(shot.masks):
            if i in claimed or m["cut"]:          # a cut-off mask births nothing: bad viewpoint (Joel's edge rule)
                continue
            keep = cones[i].copy()
            if keep.sum() >= 4:
                b = Blob(next(self._ids), keep)
                self.blobs.append(b)
                born.append(b)
        self.blobs = [b for b in self.blobs if b.keep.any()]
        if self.view:
            self.view.blobs(self.blobs, self.grid)
        return born

    # ---------------- aiming: motion planning, not the LLM ----------------

    def _obstacles(self) -> list:
        """Every blob's current volume as an armcheck box: the arm plans around what the eyes
        believe is there - inspected ones included, since looking never requires touching."""
        from lab.armcheck import Box
        out = []
        for b in self.blobs:
            p = self.grid.pts[b.keep]
            if not len(p):
                continue
            c = (p.min(0) + p.max(0)) / 2
            half = (p.max(0) - p.min(0)) / 2 + 0.005
            out.append(Box(np.array([c[0], c[1], c[2]]), 0.0, half, f"blob #{b.id}"))
        return out

    def _fly_cam(self, eye: np.ndarray, target: np.ndarray) -> None:
        """Put the WRIST CAMERA at eye, looking at target - the datarun's proven way: planned in
        JOINT space (tcp rows to tilted poses make the wrist writhe near J5=0 and the controller
        refuses them - joint_move's own docstring), armchecked, then sent. Works on the Isaac twin
        and the real cell alike (P.send speaks both). Verifies arrival from observed joints."""
        from calibration.kinematics import tool_ik, tool_pose
        from lab import armcheck
        from lab import primitives as P
        tip, wpr = ik_for(eye, target, self.cams["wrist"]["T"])   # ValueError = the caller's verdict
        if (tip[2] - SURFACE_Z) * 1000 < 95.0 or not P.reachable(*tip[:2], margin_mm=15):
            raise ValueError(f"fingertips would be {(tip[2] - SURFACE_Z) * 1000:.0f} mm up, or out of reach")
        q_now = np.asarray(self.cell.observe(cams=())["joints"], float)[:6]
        q = tool_ik(tip, wpr, q_now)
        T_now = tool_pose(q_now)
        rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q, tip, wpr,
                            height_mm=95.0, settle=5)
        if rows is None:
            raise ValueError("no safe joint path to that vantage")
        rep = armcheck.check_joints(np.asarray(rows)[:, :6], 0.0, obstacles=self._obstacles())
        if not rep.ok:
            raise ValueError(f"the move fails the arm check: {rep.problem}")
        from lab.scan import unsafe   # the padded real-mesh self-collision model (both real faults fail it)
        for k, r in enumerate(rows):
            if k % 2 == 0 and (why := unsafe(np.asarray(r, float))):
                raise ValueError(f"self-collision risk on the way: {why}")
        print(f"fly: cam -> {np.round(eye, 3)} looking at {np.round(target, 3)}, "
              f"{len(rows)} joint rows", flush=True)
        P.send(self.cell, rows)
        j = np.asarray(self.cell.observe(cams=())["joints"], float)
        cam_now = (flange_pose(j) @ self.cams["wrist"]["T"])[:3, 3]
        off = float(np.linalg.norm(cam_now - eye)) * 1000
        if off > 40:
            raise RuntimeError(f"flight failed: asked the camera to {np.round(eye, 3)}, it is at "
                               f"{np.round(cam_now, 3)} ({off:.0f} mm away) - the cell did not follow")

    def _vantages(self, blob: Blob) -> list[tuple[np.ndarray, np.ndarray]]:
        """Three (eye, target) pairs: one from above, two from ~40 degrees up at different sides -
        the low ones are what height is made of, and 40 degrees still sees where things meet the
        table (gravity rule). The standoff comes from the camera's actual field of view, so the
        blob's WHOLE current volume (slack height included) fits in frame with margin; every
        vantage is IK-checked here, roll included - this is motion planning's job, not the LLM's."""
        cam = self.cams["wrist"]
        c = blob.centre(self.grid)                                # the VOLUME's centre: aiming here
        foot = np.array([c[0], c[1], SURFACE_Z + 0.01])           # puts base AND slack crown in frame
        half = np.arctan(min(cam["size"]) / 2 / cam["fx"])        # the tight half-angle of the frame
        dist = float(np.clip(blob.extent(self.grid) / 2 / np.tan(half) * 1.6 + 0.04, 0.20, 0.42))
        cands = [foot + np.array([0.0, 0.0, dist])]
        el = np.radians(40.0)
        for a in (np.pi, np.pi * 0.6, -np.pi * 0.6, np.pi * 1.4, 0.0):   # base side first: reachable
            az = np.arctan2(foot[1], foot[0]) + a
            cands.append(foot + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]))
        out = []
        for eye in cands:
            try:
                ik_for(eye, c, cam["T"])
            except ValueError:
                continue
            out.append((eye, c))
            if len(out) == 3:
                break
        return out

    # ---------------- what the mind calls ----------------

    def look_at(self, x_mm: float, y_mm: float) -> str:
        """Move the head: hover the camera over that table point (top-down first, a 50-degree
        tilt from the base side when straight-down is out of reach), then look."""
        tgt = np.array([x_mm / 1000.0, y_mm / 1000.0, SURFACE_Z + 0.01])
        el = np.radians(50.0)
        az = np.arctan2(tgt[1], tgt[0]) + np.pi
        cands = [tgt + np.array([0.0, 0.0, 0.34])]
        for d in (0.30, 0.38):
            cands.append(tgt + d * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]))
        last: Exception | None = None
        for eye in cands:
            try:
                self._fly_cam(eye, tgt)
                break
            except (ValueError, RuntimeError) as e:
                last = e
        else:
            return f"can't get the camera over ({x_mm:.0f}, {y_mm:.0f}): {last}"
        return self.look()

    def look(self) -> str:
        shot = self._snap()
        born = self._carve(shot)
        cut = sum(m["cut"] for m in shot.masks)
        lines = [f"took a wrist photo: SAM found {len(shot.masks)} region(s)"
                 + (f", {cut} cut off by the frame edge (those birth nothing - reposition to see them whole)"
                    if cut else "")]
        lines += [b.summary(self.grid) for b in self.blobs] or ["nothing known yet"]
        if born:
            lines.append(f"new this look: {', '.join(f'#{b.id}' for b in born)}")
        return "\n".join(lines)

    def inspect(self, blob_id: int) -> str:
        b = next((b for b in self.blobs if b.id == blob_id), None)
        if b is None:
            return f"there is no blob #{blob_id}; current blobs: " + (
                ", ".join(f"#{x.id}" for x in self.blobs) or "none")
        story, taken = [], []
        for n, (eye, tgt) in enumerate(self._vantages(b), 1):   # move, snap, move, snap - SAM after
            try:
                self._fly_cam(eye, tgt)
            except (ValueError, RuntimeError) as e:   # unreachable, or the cell did not follow:
                story.append(f"vantage {n}: {e}")      # the decider hears the verdict, whole
                continue
            taken.append((n, self._grab()))
        for n, (img, joints) in taken:
            shot = self._see(img, joints)
            self._carve(shot)
            story.append(f"vantage {n}: {len(shot.masks)} region(s)")
        story.append(b.summary(self.grid))
        return "\n".join(story)

    def state(self) -> str:
        return "\n".join(b.summary(self.grid) for b in self.blobs) or "nothing known yet"
