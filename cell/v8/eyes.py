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
    """An anonymous shape: a volume of space that evidence says is occupied. The eyes never name
    it; name is written by the mind's label tool (the ventral stream), after the shape is carved."""
    id: int
    keep: np.ndarray                 # bool over grid.pts: still possible after every shot so far
    shots: int = 1
    born: float = field(default_factory=time.time)
    name: str = ""                   # the mind's label, once it asks for one
    best_crop: np.ndarray | None = None   # the closest photo of it so far (for labeling)
    best_px: int = 0                 # that crop's mask size: bigger = closer = better

    def summary(self, grid: Grid) -> str:
        p = grid.pts[self.keep]
        who = f"{self.name} (blob #{self.id})" if self.name else f"blob #{self.id}"
        if not len(p):
            return f"{who}: carved away to nothing"
        x, y = p[:, 0].mean() * 1000, p[:, 1].mean() * 1000
        sx, sy = np.ptp(p[:, 0]) * 1000 + VOX * 1000, np.ptp(p[:, 1]) * 1000 + VOX * 1000
        top = (p[:, 2].max() - SURFACE_Z) * 1000
        sure = "height still a guess (one view)" if self.shots < 2 else f"seen from {self.shots} angles"
        return (f"{who}: around ({x:.0f}, {y:.0f}) mm, footprint ~{sx:.0f} x {sy:.0f} mm, "
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


def ik_for(eye: np.ndarray, target: np.ndarray, T_flange_cam: np.ndarray,
           floor_mm: float = 55.0) -> tuple[np.ndarray, np.ndarray]:
    """(tip, wpr) of a REACHABLE pose framing target from eye, spending camera roll to find one.
    Camera roll swings the fingertips around the wrist, so the fingertip floor and reach margins
    are checked INSIDE the roll search: a low camera pose is legal whenever SOME roll keeps the
    fingers up and in reach. Raises ValueError with the last miss when no roll works."""
    from lab import primitives as P
    err: Exception | None = None
    for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4):
        tip, wpr = aim(eye, target, T_flange_cam, roll)
        up_mm = (tip[2] - SURFACE_Z) * 1000
        if up_mm < floor_mm or not P.reachable(tip[0], tip[1], margin_mm=15):
            err = ValueError(f"fingertips would be {up_mm:.0f} mm up, or out of reach")
            continue
        try:
            tool_ik(tip, wpr, tol_mm=3.0)
            return tip, wpr
        except ValueError as e:
            err = e
    raise ValueError(f"no camera roll reaches that vantage: {err}")


def pixels_of(pts: np.ndarray, cam: dict, joints=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base-frame points -> (u, v in calibration pixels, in_front). Same pinhole + one radial term
    as calibration.geometry.project, done here so we also get the depth sign.
    joints: pass them for the wrist camera (its T is flange->cam); None for a fixed camera."""
    T = np.linalg.inv(cam["T"] if joints is None else flange_pose(joints) @ cam["T"])
    pc = pts @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = pc[:, 0] / z, pc[:, 1] / z
    d = 1 + cam["k1"] * (x * x + y * y)
    w, h = cam["size"]
    return cam["fx"] * x * d + w / 2, cam["fx"] * y * d + h / 2, z > 0.02


@dataclass
class Shot:
    """One photo and what SAM said about it, with the pose it was ACTUALLY taken from.
    cam "wrist" uses the observed joints; "table"/"side" are fixed and need none."""
    k: int
    image: np.ndarray
    joints: np.ndarray
    masks: list[dict]                # [{"mask" bool HxW, "cut": touches the frame edge}]
    cam: str = "wrist"


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
        from lab import primitives as P
        # the coverage ledger: which patch of the REACHABLE table surface some shot has actually
        # framed. What was never in any frame has never been looked at - and un-looked-at table is
        # how whole objects get missed, so scan() chases these holes until none are left.
        self._reach = np.array([[P.reachable(x, y, margin_mm=15) for y in self.grid.ys]
                                for x in self.grid.xs])
        self.seen = np.zeros_like(self._reach)

    # ---------------- the evidence machinery ----------------

    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        """Just the picture and where it was taken from - no SAM yet, so flights can be batched
        (move, snap, move, snap; the thinking happens after the moving, Joel's one ok'd hardcode)."""
        obs = self.cell.observe(cams=("wrist",))
        return obs["images"]["wrist"], np.asarray(obs["joints"], float)

    def _see(self, img: np.ndarray, joints: np.ndarray, cam: str = "wrist") -> Shot:
        from lab import sam
        found = sam.segment_many(img, WORDS, scale=2.0)
        masks: list[dict] = []
        for d in sorted(found, key=lambda d: -d["area"]):     # dedupe: two words often find one thing
            m = d["mask"]
            if any((m & e["mask"]).sum() > 0.5 * min(m.sum(), e["mask"].sum()) for e in masks):
                continue
            edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
            masks.append({"mask": m, "cut": bool(edge)})
        shot = Shot(next(self._shots), img, joints, masks, cam=cam)
        if self.view:
            self.view.shot(shot, self.cams[cam], self.grid)
        return shot

    def _snap(self) -> Shot:
        img, joints = self._grab()
        return self._see(img, joints)

    def _carve(self, shot: Shot) -> list[Blob]:
        """Apply one shot to every blob (three-valued), then birth blobs for unclaimed masks."""
        cam = self.cams[shot.cam]
        jnts = shot.joints if shot.cam == "wrist" else None
        u, v, front = pixels_of(self.grid.pts, cam, jnts)
        w, h = cam["size"]
        mh, mw = (shot.masks[0]["mask"].shape if shot.masks else (720, 1280))
        ui = np.clip((u * mw / w), 0, mw - 1).astype(np.int32)
        vi = np.clip((v * mh / h), 0, mh - 1).astype(np.int32)
        in_frame = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if shot.cam == "wrist":   # the ledger counts only the aimable eye's own looks
            self.seen |= in_frame.reshape(self.grid.shape)[:, :, 0]
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
                self._keep_crop(b, shot, own)
            else:
                b.keep &= ~in_frame | any_mask   # not seen here: clear table carves, other masks occlude

        born = []
        for i, m in enumerate(shot.masks):
            if i in claimed or m["cut"]:          # a cut-off mask births nothing: bad viewpoint (Joel's edge rule)
                continue
            keep = cones[i].copy()
            if keep.sum() >= 4:
                b = Blob(next(self._ids), keep)
                self._keep_crop(b, shot, i)
                self.blobs.append(b)
                born.append(b)
        self._gravity()
        self.blobs = [b for b in self.blobs if b.keep.any()]
        if self.view:
            self.view.blobs(self.blobs, self.grid)
        return born

    def _gravity(self) -> None:
        """Nothing floats: gravity is the one licensed physics prior. A connected piece of a blob
        with nothing under it - not the table, not another blob's volume (stacks stay legal) - is
        carving debris, not matter: drop it."""
        from scipy import ndimage
        occ = np.zeros(self.grid.shape, bool)
        for b in self.blobs:
            occ |= b.keep.reshape(self.grid.shape)
        for b in self.blobs:
            vol = b.keep.reshape(self.grid.shape)
            lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
            if n <= 1:
                if n == 1 and not vol[:, :, 0].any():   # one piece, hovering whole: check support
                    under = np.zeros_like(vol)
                    under[:, :, :-1] = vol[:, :, 1:]
                    under[:, :, :-2] |= vol[:, :, 2:]
                    if not (under & occ & ~vol).any():
                        b.keep = np.zeros_like(b.keep)
                continue
            kept = np.zeros_like(vol)
            for k in range(1, n + 1):
                comp = lab == k
                if comp[:, :, 0].any():                 # stands on the table
                    kept |= comp
                    continue
                under = np.zeros_like(comp)
                under[:, :, :-1] = comp[:, :, 1:]       # the cells directly beneath the piece
                under[:, :, :-2] |= comp[:, :, 2:]      # (contact tolerance: voxel rounding leaves seams)
                if (under & occ & ~vol).any():          # rests on another blob: a stack, not debris
                    kept |= comp
            b.keep = kept.ravel()

    def _keep_crop(self, b: Blob, shot: Shot, mi: int) -> None:
        """Remember the closest photo of this blob (biggest mask = closest view): the labeling
        tool sends exactly this crop to the vision model, so the name is read off the best look."""
        m = shot.masks[mi]["mask"]
        px = int(m.sum())
        if px <= b.best_px:
            return
        vs, us = np.nonzero(m)
        mh, mw = m.shape
        ih, iw = shot.image.shape[:2]
        x0, x1 = int(us.min() * iw / mw), int(us.max() * iw / mw)
        y0, y1 = int(vs.min() * ih / mh), int(vs.max() * ih / mh)
        r = max(x1 - x0, y1 - y0) // 2 + 24
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        crop = shot.image[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if crop.size and min(crop.shape[:2]) < 300:   # blow tiny crops up so the model sees pixels
            s = 300 / min(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if crop.size:
            b.best_crop, b.best_px = crop, px

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

    def home(self) -> None:
        """The un-cornering reflex: plain joint interpolation back to HOME, a configuration every
        plan can start from. The branch-keeping planner can corner itself (a j2/j4 squeeze, or a
        flight the cell refused mid-path leaves the arm stranded): from HOME it never does."""
        from calibration.kinematics import HOME
        from lab import armcheck
        from lab.scan import unsafe
        q0 = np.asarray(self.cell.observe(cams=())["joints"], float)[:6]
        tgt = np.asarray(HOME, float)[:6]
        n = max(10, int(np.abs(tgt - q0).max() / 3.0))
        rows = [[*(q0 + (tgt - q0) * k / n).tolist(), 0.0] for k in range(1, n + 1)]
        rep = armcheck.check_joints(np.asarray(rows)[:, :6], 0.0)
        if not rep.ok:
            raise RuntimeError(f"can't even get home: {rep.problem}")
        for k, r in enumerate(rows):
            if k % 2 == 0 and (why := unsafe(np.asarray(r, float))):
                raise RuntimeError(f"can't even get home: {why}")
        print("home: un-cornering, re-planning from the home pose", flush=True)
        self.cell.play(np.asarray(rows, float), kind="joints")

    def _fly_cam(self, eye: np.ndarray, target: np.ndarray) -> None:
        """Put the WRIST CAMERA at eye, looking at target - the datarun's proven way: planned in
        JOINT space (tcp rows to tilted poses make the wrist writhe near J5=0 and the controller
        refuses them - joint_move's own docstring), armchecked, then sent. Works on the Isaac twin
        and the real cell alike (P.send speaks both). Verifies arrival from observed joints."""
        tip, wpr = ik_for(eye, target, self.cams["wrist"]["T"])   # an illegal POSE: homing can't help
        try:
            self._fly_rows(tip, wpr, eye)
        except (ValueError, RuntimeError):
            self.home()                       # cornered or stranded: the reflex, then once more
            self._fly_rows(tip, wpr, eye)

    def _fly_rows(self, tip: np.ndarray, wpr: np.ndarray, eye: np.ndarray) -> None:
        from calibration.kinematics import tool_ik, tool_pose
        from lab import armcheck
        from lab import primitives as P
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
        print(f"fly: cam -> {np.round(eye, 3)}, {len(rows)} joint rows", flush=True)
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
        out = []
        try:
            ik_for(foot + np.array([0.0, 0.0, dist]), c, cam["T"])
            out.append((foot + np.array([0.0, 0.0, dist]), c))
        except ValueError:
            pass
        # two elevations: 40 deg shapes the sides, and a LOW 22 deg grazes the top - without the low
        # looks a tall thing keeps a pyramid tent over it (steep silhouettes can never cut it off).
        # Low poses fight the 95 mm fingertip floor, so many azimuths are tried to find legal ones.
        base_az = np.arctan2(foot[1], foot[0])
        for el_deg, quota in ((40.0, 2), (22.0, 2)):
            el, got = np.radians(el_deg), 0
            for a in (np.pi, np.pi * 0.6, -np.pi * 0.6, np.pi * 1.4, 0.0,
                      np.pi * 0.3, -np.pi * 0.3, np.pi * 1.7):   # base side first: reachable
                az = base_az + a
                eye = foot + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                try:
                    ik_for(eye, c, cam["T"])
                except ValueError:
                    continue
                out.append((eye, c))
                got += 1
                if got == quota:
                    break
        return out

    # ---------------- what the mind calls ----------------

    def scan(self) -> str:
        """The opening move, hardcoded on purpose (we are on the clock): a few top-down shots
        covering the table, flown first, SAM'd after - finds the columns of stuff in one sweep."""
        # the fixed cameras first: the table cam births everything (nothing can be missed),
        # the side cam bounds every height with one standing low profile
        obs = self.cell.observe(cams=("table", "side"))
        for name in ("table", "side"):
            if obs["images"].get(name) is not None:
                self._carve(self._see(obs["images"][name], np.asarray(obs["joints"], float), cam=name))
        flights = [((x, y, SURFACE_Z + 0.33), (x, y, SURFACE_Z + 0.01))
                   for x, y in ((0.34, -0.17), (0.34, 0.17), (0.52, -0.17), (0.52, 0.17))]
        # plus two borderline-horizontal looks across the table: they bound every height at once
        flights += [((0.26, -0.26, SURFACE_Z + 0.10), (0.56, 0.08, SURFACE_Z + 0.02)),
                    ((0.26, 0.26, SURFACE_Z + 0.10), (0.56, -0.08, SURFACE_Z + 0.02))]
        taken, skipped = [], []
        for eye, tgt in flights:
            try:
                self._fly_cam(np.array(eye), np.array(tgt))
            except (ValueError, RuntimeError) as e:
                skipped.append(f"({eye[0] * 1000:.0f}, {eye[1] * 1000:.0f}): {e}")
                continue
            taken.append(self._grab())
        for img, joints in taken:
            self._carve(self._see(img, joints))
        # chase coverage holes: any reachable patch no frame has touched gets its own look,
        # until the whole reachable surface has been seen (or the holes stop being fixable)
        chased = 0
        for _ in range(5):
            pct, hole = self._coverage()
            if hole is None:
                break
            self.look_at(hole[0], hole[1])
            chased += 1
        pct, hole = self._coverage()
        lines = [f"scanned the table: {len(taken)} sweep shots"
                 + (f" + {chased} chasing unseen patches" if chased else "")
                 + f"; {pct:.0f}% of the reachable surface has been looked at"
                 + (f" (still unseen around ({hole[0]:.0f}, {hole[1]:.0f}))" if hole else "")
                 + (f"; {len(skipped)} spots unreachable" if skipped else "")]
        lines += [b.summary(self.grid) for b in self.blobs] or ["nothing found"]
        return "\n".join(lines)

    def _coverage(self) -> tuple[float, tuple[float, float] | None]:
        """(percent of the reachable surface seen, centre of the biggest unseen patch worth a look)."""
        from scipy import ndimage
        r = self._reach
        pct = 100.0 * np.count_nonzero(self.seen & r) / max(1, np.count_nonzero(r))
        lab, n = ndimage.label(r & ~self.seen)
        if n == 0:
            return pct, None
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        k = int(np.argmax(sizes))
        if sizes[k] < 150:            # smaller than ~50 x 50 mm: nothing bin-sized hides there
            return pct, None
        cx, cy = ndimage.center_of_mass(lab == k + 1)
        return pct, (float(self.grid.xs[int(cx)]) * 1000, float(self.grid.ys[int(cy)]) * 1000)

    def look_at(self, x_mm: float, y_mm: float) -> str:
        """Move the head: hover the camera over that table point (top-down first, a 50-degree
        tilt from the base side when straight-down is out of reach), then look."""
        from lab import primitives as P
        if not P.reachable(x_mm / 1000.0, y_mm / 1000.0):
            return (f"({x_mm:.0f}, {y_mm:.0f}) is outside my reach (about 550 mm from the base, "
                    "and inside the table bounds): pick a nearer point")
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

    def roster(self) -> list[dict]:
        """What the robot believes, as rows for the page: one per blob, numbers in mm."""
        out = []
        for b in self.blobs:
            p = self.grid.pts[b.keep]
            if not len(p):
                continue
            out.append({"id": b.id, "name": b.name or "blob",
                        "x": round(float(p[:, 0].mean()) * 1000), "y": round(float(p[:, 1].mean()) * 1000),
                        "sx": round(float(np.ptp(p[:, 0])) * 1000 + 4), "sy": round(float(np.ptp(p[:, 1])) * 1000 + 4),
                        "top": round(float(p[:, 2].max() - SURFACE_Z) * 1000), "shots": b.shots})
        return out

    def state(self) -> str:
        return "\n".join(b.summary(self.grid) for b in self.blobs) or "nothing known yet"

    # ---------------- hands: three motions, checks are the only authority ----------------
    # grab_to and drop_into move GRASPABLE things (narrow side fits the jaws), slide-in first
    # (descend into proven-empty space beside it, slide until it sits between the fingers: a miss
    # is a non-event, not a fault), top pick only when no side is clear. The world model updates
    # ONLY on what the verification looks found - never on what the motion intended.

    GRASP_MM = 16.0   # the widest narrow-side the jaws take (pens, bolts, small blocks)

    def yaw_of(self, b: Blob) -> float:
        """The hull footprint's long axis (rad): a slide must come at a pen broadside."""
        p = self.grid.pts[b.keep][:, :2]
        if len(p) < 3:
            return 0.0
        _, _, vt = np.linalg.svd(p - p.mean(0))
        return float(np.arctan2(vt[0, 1], vt[0, 0]))

    def grip_mm(self, b: Blob) -> float:
        """How wide the hull is across its NARROW side: what the jaws would have to swallow."""
        p = self.grid.pts[b.keep][:, :2]
        if len(p) < 3:
            return 999.0
        q = p - p.mean(0)
        _, _, vt = np.linalg.svd(q)
        return float(np.ptp(q @ vt[1])) * 1000 + VOX * 1000

    def _hover(self, x_mm: float, y_mm: float, height: float = 0.25) -> None:
        """The camera above a point, top-down first, tilted fallback (same family as look_at)."""
        tgt = np.array([x_mm / 1000.0, y_mm / 1000.0, SURFACE_Z + 0.01])
        el, az = np.radians(50.0), np.arctan2(tgt[1], tgt[0]) + np.pi
        last: Exception | None = None
        for eye in [tgt + np.array([0.0, 0.0, height]),
                    tgt + 0.28 * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])]:
            try:
                self._fly_cam(eye, tgt)
                return
            except (ValueError, RuntimeError) as e:
                last = e
        raise ValueError(f"can't get the camera over ({x_mm:.0f}, {y_mm:.0f}): {last}")

    def _spot_masks(self, x_mm: float, y_mm: float, height: float = 0.25) -> list[tuple[float, float]]:
        """Fly over a point and answer the binary question factory: where are SAM's things near
        here, in mm? Verification only - these shots never carve (updates come from verdicts)."""
        from calibration.geometry import pixel_to_base
        self._hover(x_mm, y_mm, height)
        img, j = self._grab()
        shot = self._see(img, j)
        cam = self.cams["wrist"]
        w, h = cam["size"]
        out = []
        for m in shot.masks:
            vs, us = np.nonzero(m["mask"])
            mh, mw = m["mask"].shape
            p = pixel_to_base(cam, (us.mean() * w / mw, vs.mean() * h / mh), joints=j)
            out.append((float(p[0]) * 1000, float(p[1]) * 1000))
        return out

    def _translate(self, b: Blob, to_mm: tuple[float, float]) -> None:
        """A CONFIRMED move teleports the rigid hull in the model: bookkeeping after the verdict,
        never a prediction (Joel: update on validation, not on movement)."""
        p = self.grid.pts[b.keep]
        c = p[:, :2].mean(0) * 1000
        dx = int(round((to_mm[0] - c[0]) / 1000 / VOX))
        dy = int(round((to_mm[1] - c[1]) / 1000 / VOX))
        vol = np.roll(b.keep.reshape(self.grid.shape), (dx, dy), axis=(0, 1))
        if dx > 0:
            vol[:dx] = False
        elif dx < 0:
            vol[dx:] = False
        if dy > 0:
            vol[:, :dy] = False
        elif dy < 0:
            vol[:, dy:] = False
        b.keep = vol.ravel()
        if self.view:
            self.view.blobs(self.blobs, self.grid)

    def _grasp_rows(self, b: Blob, at_mm: tuple[float, float], place_mm: tuple[float, float],
                    travel_mm: float, drop_mm: float | None):
        """Slide-in first, top pick as fallback; planned in joint space on the proven path."""
        try:
            return self._plan_grasp(b, at_mm, place_mm, travel_mm, drop_mm)
        except ValueError:
            self.home()                       # cornered: the reflex, then plan once more from HOME
            return self._plan_grasp(b, at_mm, place_mm, travel_mm, drop_mm)

    def _plan_grasp(self, b: Blob, at_mm, place_mm, travel_mm: float, drop_mm: float | None):
        from calibration.kinematics import tcp_uf0
        from lab import armcheck
        from lab import moves as MV
        cur_q = np.asarray(self.cell.observe(cams=())["joints"], float)
        cur = tcp_uf0(cur_q)
        jaw = float(np.degrees(self.yaw_of(b)) + 90.0)
        at = (at_mm[0] / 1000.0, at_mm[1] / 1000.0)
        place = (place_mm[0] / 1000.0, place_mm[1] / 1000.0)
        length = max(1.0, max(float(np.ptp(self.grid.pts[b.keep][:, 0])),
                              float(np.ptp(self.grid.pts[b.keep][:, 1]))) * 1000)

        rounds = []              # every OTHER believed thing as a keep-out circle for the slide entry
        for o in self._others(b):
            po = self.grid.pts[o.keep]
            rad = max(float(np.ptp(po[:, 0])), float(np.ptp(po[:, 1]))) * 500 + 20
            rounds.append((float(po[:, 0].mean()) * 1000, float(po[:, 1].mean()) * 1000, rad))

        from lab import primitives as P

        def clear(xy) -> bool:
            # the slide's entry point must be free of every other believed thing AND legal for the
            # arm: near enough that fingertips may come low (the fitted far-low rule starts ~485 mm
            # out), so slides naturally come from the base side instead of being refused outright
            if np.hypot(xy[0], xy[1]) * 1000 > 460 or not P.reachable(xy[0], xy[1], margin_mm=15):
                return False
            return all(np.hypot(xy[0] * 1000 - ox, xy[1] * 1000 - oy) > orad for ox, oy, orad in rounds)

        rows, how = MV.push_grasp(cur, at, jaw, length, place, travel_mm, drop_mm, clear)
        if not rows:
            rows, how = MV.pick_top(cur, at, jaw, place, travel_mm, drop_mm), f"top pick (slide: {how})"
        else:
            how = "slide-in grasp"
        rep = armcheck.check(rows, cur_q, obstacles=self._obstacles_except(b))
        if not rep.ok:
            raise ValueError(f"the move fails the arm check: {rep.problem}")
        jrows, why = MV.joints(rows, cur_q)
        if jrows is None:
            raise ValueError(f"no safe joint path: {why}")
        return jrows, how

    def _others(self, b: Blob):
        return [o for o in self.blobs if o.id != b.id and o.keep.any()]

    def _obstacles_except(self, b: Blob) -> list:
        from lab.armcheck import Box
        out = []
        for o in self._others(b):
            p = self.grid.pts[o.keep]
            c = (p.min(0) + p.max(0)) / 2
            half = (p.max(0) - p.min(0)) / 2 + 0.005
            out.append(Box(np.array([c[0], c[1], c[2]]), 0.0, half, f"blob #{o.id}"))
        return out

    def grab_to(self, blob_id: int, x_mm: float, y_mm: float) -> str:
        b = next((b for b in self.blobs if b.id == blob_id), None)
        if b is None:
            return f"there is no blob #{blob_id}"
        gw = self.grip_mm(b)
        if gw > self.GRASP_MM:
            return (f"blob #{blob_id} is {gw:.0f} mm across its narrow side; the jaws take about "
                    f"{self.GRASP_MM:.0f} mm. It can be labeled and used as a container, never moved.")
        for o in self._others(b):   # the destination must be clear of everything we believe in
            p = self.grid.pts[o.keep]
            if np.hypot(p[:, 0].mean() * 1000 - x_mm, p[:, 1].mean() * 1000 - y_mm) < \
               (max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 1000 + gw) / 2 + 25:
                return (f"({x_mm:.0f}, {y_mm:.0f}) is too close to blob #{o.id}: pick an empty spot "
                        "at least ~25 mm clear of everything")
        c = b.centre(self.grid)
        sx, sy = c[0] * 1000, c[1] * 1000
        cents = self._spot_masks(sx, sy)                      # fresh fix + the neighbour count
        near = sorted((h for h in cents if np.hypot(h[0] - sx, h[1] - sy) < 35),
                      key=lambda h: np.hypot(h[0] - sx, h[1] - sy))
        n_before = sum(np.hypot(h[0] - sx, h[1] - sy) < 55 for h in cents)
        if not near:
            return f"blob #{blob_id} is NOT at ({sx:.0f}, {sy:.0f}) any more: look there before moving it"
        gx, gy = near[0]
        jrows, how = self._grasp_rows(b, (gx, gy), (x_mm, y_mm), 120.0, None)
        print(f"grab: blob #{blob_id} ({gx:.0f}, {gy:.0f}) -> ({x_mm:.0f}, {y_mm:.0f}), {how}, "
              f"{len(jrows)} rows", flush=True)
        self.cell.play(np.asarray(jrows, float), kind="joints")
        # the verdicts - and ONLY the verdicts - update the world. One shot answers both
        # questions when source and destination fit in one frame (half the verification flights).
        if np.hypot(x_mm - sx, y_mm - sy) < 200:
            cents_v = self._spot_masks((sx + x_mm) / 2, (sy + y_mm) / 2, height=0.32)
            found = [h for h in cents_v if np.hypot(h[0] - x_mm, h[1] - y_mm) < 45]
            n_after = sum(np.hypot(h[0] - sx, h[1] - sy) < 55 for h in cents_v)
        else:
            found = [h for h in self._spot_masks(x_mm, y_mm) if np.hypot(h[0] - x_mm, h[1] - y_mm) < 45]
            n_after = (sum(np.hypot(h[0] - sx, h[1] - sy) < 55 for h in self._spot_masks(sx, sy))
                       if found else 99)
        if not found:
            return (f"MISS: nothing found at ({x_mm:.0f}, {y_mm:.0f}) after the {how}. "
                    f"Blob #{blob_id} should still be near ({gx:.0f}, {gy:.0f}): look, then retry.")
        self._translate(b, found[0])
        src = ("the old spot is clear" if n_after == 0 else
               f"{n_after} thing(s) still near the old spot (was {n_before})" +
               ("" if n_after < n_before else " - the count did NOT drop: maybe a neighbour was moved instead"))
        return (f"moved blob #{blob_id} with a {how}: confirmed at ({found[0][0]:.0f}, {found[0][1]:.0f}); {src}")

    def drop_into(self, blob_id: int, container_id: int):
        """Carry a graspable thing above an open-top container and let go; returns the story so far
        plus the straight-down wrist shot for the yes/no glance (the caller asks the VLM)."""
        b = next((x for x in self.blobs if x.id == blob_id), None)
        cont = next((x for x in self.blobs if x.id == container_id), None)
        if b is None or cont is None:
            return f"no such blob (#{blob_id} or #{container_id})", None
        gw = self.grip_mm(b)
        if gw > self.GRASP_MM:
            return f"blob #{blob_id} is too wide to grasp ({gw:.0f} mm): it cannot be put anywhere", None
        pc = self.grid.pts[cont.keep]
        cx, cy = pc[:, 0].mean() * 1000, pc[:, 1].mean() * 1000
        top = (pc[:, 2].max() - SURFACE_Z) * 1000
        c = b.centre(self.grid)
        sx, sy = c[0] * 1000, c[1] * 1000
        near = sorted((h for h in self._spot_masks(sx, sy) if np.hypot(h[0] - sx, h[1] - sy) < 35),
                      key=lambda h: np.hypot(h[0] - sx, h[1] - sy))
        if not near:
            return f"blob #{blob_id} is NOT at ({sx:.0f}, {sy:.0f}) any more: look before moving it", None
        drop = top + 15.0
        if np.hypot(cx, cy) > 440:   # a far bin: beyond ~485 mm out the fingertips must stay high
            drop = max(drop, 85.0)   # (the fitted far-low rule) - drop from higher, gravity still lands it
        jrows, how = self._grasp_rows(b, near[0], (cx, cy), max(130.0, top + 60.0), drop_mm=drop)
        print(f"drop: blob #{blob_id} -> into #{container_id} at ({cx:.0f}, {cy:.0f}), {how}", flush=True)
        self.cell.play(np.asarray(jrows, float), kind="joints")
        self._hover(cx, cy)
        img, _ = self._grab()
        return f"dropped with a {how} above blob #{container_id} (top {top:.0f} mm)", img

    def confirm_into(self, blob_id: int, container_id: int, inside: bool) -> None:
        b = next((x for x in self.blobs if x.id == blob_id), None)
        cont = next((x for x in self.blobs if x.id == container_id), None)
        if b is None or cont is None:
            return
        if inside:
            pc = self.grid.pts[cont.keep]
            self._translate(b, (pc[:, 0].mean() * 1000, pc[:, 1].mean() * 1000))
            b.name = (b.name or "blob") if "(in #" in (b.name or "") else \
                     f"{b.name or 'blob'} (in #{container_id})"
        if self.view:
            self.view.blobs(self.blobs, self.grid)

    def pile_spots(self, counts: list[int]) -> list[list[tuple[float, float]]]:
        """The layout service: a TIDY LINE of piles a person can read at a glance - group anchors
        evenly spaced across the comfortable middle of the workspace, items gridded 38 mm apart
        within each group, the whole line nudged outward until it clears everything believed in.
        (Emergent farthest-point spots hugged corners and looked like noise: Joel vetoed them.)"""
        k = len(counts)
        if k == 0:
            return []
        rounds = []
        for b in self.blobs:
            p = self.grid.pts[b.keep]
            if len(p):
                rounds.append((float(p[:, 0].mean()) * 1000, float(p[:, 1].mean()) * 1000,
                               max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))) * 500 + 30))
        span = min(120.0 * (k - 1), 440.0)
        ys = [0.0] if k == 1 else list(np.linspace(-span / 2, span / 2, k))
        for ax in (320.0, 380.0, 440.0, 260.0):        # the line slides in x until it clears everything
            spots = []
            ok = True
            for ay, n in zip(ys, counts):
                offs = [(dx * 38 - 38, dy * 38 - 38) for dy in range(3) for dx in range(3)]
                sp = [(ax + ox, ay + oy) for ox, oy in offs][:n]
                from lab import primitives as P
                for x, y in sp:
                    if not P.reachable(x / 1000, y / 1000, margin_mm=20) or                        any(np.hypot(x - ox_, y - oy_) < orad + 25 for ox_, oy_, orad in rounds):
                        ok = False
                        break
                if not ok:
                    break
                spots.append(sp)
            if ok:
                return spots
        return []   # no clear line fits: the caller says so and the mind shrinks the job
