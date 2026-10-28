"""Objects from a close look, only where the two streams agree. SAM outlines things in the wrist camera's picture; depth
says what stands up off the table. Both land on one grid (the depth image's pixels), and:

  - outline and raised depth overlap: an object. Its footprint is the overlap (what both see), its height the depth
    inside it against the table just around it. agree = the share of the outline that stands up (1 = they match).
  - outline, nothing raised under it ("flat"), or most of the outline flat ("partial"), or no depth there: an issue,
    and the arm looks again, closer. A print, a shadow or a sticker stays flat and is dropped.
  - raised depth no outline covers: small or low = sensor noise (paper wrinkles), dropped. Clearly real: an
    "unknown" (never picked, always steered around) and an issue: look again, and maybe SAM needs another word.
    Cups and boxes (tall and wide) count from depth alone: walls are walls.

    objs, masks, issues = measure(obs, cams, prompts)
    Objects().merge(objs)                    # ids kept by position across looks

Also: where the table camera's SAM outlines are on the table (regions worth a close look).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, pixel_to_base
from lab import sam
from lab.things import colour_name
from lab.viz import project_to_image

MIN_HEIGHT_MM = 3.0   # an outline whose inside stands less than this above its ring is flat: a print, a wrinkle, a shadow
RING_PX = 14          # the local table: this far around the outline (wrist picture, 1280 px wide)
BOX_MM = 40.0         # a top face this wide on both sides: a cup or box to sort into
SAME_MM = 12.0        # two measurements this close are one object


@dataclass
class Obj:
    id: int
    xy: np.ndarray
    size_mm: tuple
    angle: float
    top_mm: float
    kind: str = "part"              # "part", "box", or "unknown" (depth only: steered around, never picked)
    color: str = "?"
    rgb: tuple = (128, 128, 128)
    what: str = ""                  # the SAM prompt that outlined it
    label: str | None = None        # what the ventral stream calls it
    looks: int = 1
    pixels: int = 0                 # depth pixels inside the outline (how much the measurement rests on)
    outline: np.ndarray | None = None  # (N, 2) base m: the footprint both streams agree on (what grips are planned on)
    agree: float = 1.0              # share of SAM's outline that depth sees standing up
    history: list = field(default_factory=list)

    @property
    def pts(self) -> np.ndarray:
        """Its box's corners, top and bottom (what the picture crop and the table-camera outline need)."""
        a = np.radians(self.angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = np.array([[sx * self.size_mm[0], sy * self.size_mm[1]] for sx in (-.5, .5) for sy in (-.5, .5)]) / 1000 @ R.T + self.xy
        return np.vstack([np.column_stack([c, np.full(4, z)]) for z in (SURFACE_Z, SURFACE_Z + self.top_mm / 1000)])

    def footprint(self) -> np.ndarray:
        """Its outline from above: where SAM's outline and raised depth overlap, else its measured box."""
        return self.outline if self.outline is not None else self.pts[:4, :2]

    @property
    def name(self) -> str:
        return f"{self.color} {self.what or self.kind} #{self.id}"

    def describe(self) -> str:
        return (f"{self.name}: at ({self.xy[0] * 1000:.0f}, {self.xy[1] * 1000:.0f}) mm, {self.size_mm[0]:.0f} x "
                f"{self.size_mm[1]:.0f} mm, {self.top_mm:.0f} mm tall ({self.pixels} depth px, {self.looks} looks)")


RAISE_MM = 3.0         # depth (smoothed over 5 x 5 pixels) this far above the local table counts as standing up
NOISE_MM = 6.0         # a raised patch no outline covers, lower than this: paper wrinkles and sensor noise
NOISE_PX = 60          # or smaller than this (depth pixels at 848 wide)
BIG_MM = 25.0          # tall and wider than BOX_MM: a cup or box, from depth alone
AGREE = 0.3            # under this share of an outline standing up, look again (real cubes: 26-72 %)


@dataclass
class Issue:
    kind: str              # "flat", "partial", "no_depth" (an outline depth doesn't back) or "depth_only"
    xy: np.ndarray
    text: str


def measure(obs: dict, cams: dict, prompts: list[str], masks: list | None = None) -> tuple[list[Obj], list, list[Issue]]:
    """One wrist look -> (objects both streams agree on, SAM's masks, the disagreements worth another look)."""
    img = obs["images"]["wrist"]
    H, W = img.shape[:2]
    if masks is None:
        masks = [m for p in prompts for m in sam.segment(img, p, scale=1.0)]
    masks = _dedupe(masks)
    D = obs["depth"]["wrist"]
    Hd, Wd = D.shape
    pts, px = depth_to_base(cams["wrist_depth"], D, with_pixels=True, joints=obs["joints"])
    if len(pts) < 500:
        return [], masks, []
    # height of every depth pixel above this view's own table plane (fitted to the low points: tilt, offset, paper)
    z = pts[:, 2]
    low = np.abs(z - SURFACE_Z) < 0.008
    plane = np.array([0.0, 0.0, SURFACE_Z])
    for _ in range(2):
        if low.sum() < 500:
            break
        A = np.column_stack([pts[low, 0], pts[low, 1], np.ones(low.sum())])
        plane = np.linalg.lstsq(A, z[low], rcond=None)[0]
        low &= np.abs(z - (pts[:, :2] @ plane[:2] + plane[2])) < 0.003  # drop what stands on it, refit
    h = (z - (pts[:, :2] @ plane[:2] + plane[2])) * 1000
    hg = np.zeros((Hd, Wd), np.float32)
    hg[px[:, 1], px[:, 0]] = h
    valid = np.zeros((Hd, Wd), bool)
    valid[px[:, 1], px[:, 0]] = True
    idx = np.full((Hd, Wd), -1, int)
    idx[px[:, 1], px[:, 0]] = np.arange(len(pts))
    smooth = cv2.medianBlur(hg, 5)                       # pool neighbours: single-pixel speckle doesn't stand up
    raised = valid & (smooth > RAISE_MM)
    raised = cv2.morphologyEx(raised.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    # SAM's masks on the depth grid: label of the outline each depth pixel falls in (-1: none)
    uv = project_to_image("wrist", cams, pts, (W, H), joints=obs["joints"])
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    lab_img = np.full((H, W), -1, np.int16)
    for k, m in enumerate(masks):                        # largest first: a part inside a cup's outline keeps its own
        lab_img[m["mask"].astype(bool)] = k
    lab = np.full((Hd, Wd), -1, np.int16)
    lab[px[ok, 1], px[ok, 0]] = lab_img[v[ok], u[ok]]
    k_ring = max(2, int(round(RING_PX * Wd / 1280)))
    out, issues = [], []
    to_xy = lambda region: pts[idx[region]][:, :2]

    for k, m in enumerate(masks):
        ys, xs = np.nonzero(m["mask"])
        if len(xs) == 0 or xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
            continue  # cut off by the picture's edge: a look centred on it measures it whole
        where = pixel_to_base(cams["wrist"], (np.array([xs.mean(), ys.mean()]) + 0.5) * np.array(cams["wrist"]["size"]) / [W, H],
                              z=SURFACE_Z, joints=obs["joints"])[:2]
        M = (lab == k) & valid
        if M.sum() < 15:
            issues.append(Issue("no_depth", where, f"'{m['prompt']}' outline with no depth under it"))
            continue
        ring = cv2.dilate(M.astype(np.uint8), np.ones((2 * k_ring + 1,) * 2, np.uint8)).astype(bool) & (lab == -1) & valid
        table = float(np.median(hg[ring])) if ring.sum() >= 30 else 0.0
        up = M & (smooth - table > RAISE_MM)
        up = cv2.morphologyEx(up.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        agree = float(up.sum() / M.sum())
        top = float(np.percentile(hg[M] - table, 90))
        if up.sum() < 10 or top < MIN_HEIGHT_MM:
            issues.append(Issue("flat", where, f"'{m['prompt']}' outline, but nothing stands up under it ({top:.0f} mm)"))
            continue
        # Shape: SAM's outline bounds it (depth smears edges: the raised patch inside a real 15 mm cube comes out anywhere
        # from 11 x 7 to 20 x 19 mm); depth vouches that it stands up and says how tall. The outline goes on the plane
        # of its top, so it's where the fingers meet it.
        cs, _ = cv2.findContours(m["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c_uv = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        c_uv = c_uv[::max(1, len(c_uv) // 120)]
        z_top = float(np.median(pts[idx[M]][:, :2] @ plane[:2]) + plane[2]) + top / 1000
        outline = pixel_to_base(cams["wrist"], (c_uv + 0.5) * np.array(cams["wrist"]["size"]) / [W, H],
                                z=z_top, joints=obs["joints"])[:, :2]
        (cx, cy), (a, b), ang = cv2.minAreaRect((outline * 1000).astype(np.float32))
        long_, short = max(a, b), min(a, b)
        mk = m["mask"].astype(np.uint8)
        rgb = tuple(int(c) for c in np.median(img[cv2.erode(mk, np.ones((3, 3), np.uint8)).astype(bool)], axis=0))
        o = Obj(0, np.array([cx, cy]) / 1000, (long_, short), (ang if a >= b else ang + 90) % 180, top,
                "box" if short > BOX_MM else "part", colour_name(rgb), rgb, m["prompt"], pixels=int(up.sum()),
                outline=outline, agree=agree)
        out.append(o)
        spill = (cv2.dilate(up.astype(np.uint8), np.ones((2 * k_ring + 1,) * 2, np.uint8)).astype(bool) & raised & (lab == -1)).sum()
        if o.kind == "part" and (agree < AGREE or spill > up.sum()):
            issues.append(Issue("partial", o.xy, f"{o.name}: only {agree:.0%} of the outline stands up"
                                + (", and depth spills past it" if spill > up.sum() else "")))

    # raised depth no outline covers (and that isn't the edge of an outlined thing)
    near_mask = cv2.dilate((lab >= 0).astype(np.uint8), np.ones((2 * k_ring + 1,) * 2, np.uint8)).astype(bool)
    n, cc, st, _ = cv2.connectedComponentsWithStats((raised & ~(lab >= 0)).astype(np.uint8))
    noise = 0
    for c in range(1, n):
        x, y, w, hh, area = st[c]
        if x <= 1 or y <= 1 or x + w >= Wd - 1 or y + hh >= Hd - 1:
            continue
        region = cc == c
        if (region & near_mask).sum() > 0.5 * area:
            continue  # the rim of an outlined object (counted as its spill)
        top = float(np.percentile(hg[region], 90))
        if area < NOISE_PX * (Wd / 848) ** 2 or top < NOISE_MM:
            noise += 1
            continue
        xy = to_xy(region)
        (cx, cy), (a, b), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
        big = top > BIG_MM and min(a, b) > BOX_MM
        rim = region & (hg > 0.8 * top)
        rgb = tuple(int(q) for q in np.median(_colours(img, pts[idx[rim]], obs, cams), axis=0)) if rim.any() else (128, 128, 128)
        o = Obj(0, np.array([cx, cy]) / 1000, (max(a, b), min(a, b)), (ang if a >= b else ang + 90) % 180, top,
                "box" if big else "unknown", colour_name(rgb), rgb, "container (depth)" if big else "depth only",
                pixels=int(area))
        out.append(o)
    if noise:
        issues.append(Issue("noise", np.zeros(2), f"{noise} small raised patches dropped as noise"))
    return out, masks, issues


def _colours(img, pts3, obs, cams):
    H, W = img.shape[:2]
    uv = project_to_image("wrist", cams, pts3, (W, H), joints=obs["joints"])
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    return img[v, u]


def _dedupe(masks: list, iou: float = 0.6) -> list:
    keep = []
    for m in sorted(masks, key=lambda m: -m["area"]):
        if all((m["mask"] & k["mask"]).sum() / max((m["mask"] | k["mask"]).sum(), 1) < iou for k in keep):
            keep.append(m)
    return keep


class DepthOnly:
    """Raised depth no outline covers, kept across looks: one that a look from somewhere else finds again within
    SAME_MM is real (an "unknown"); one a look covering its spot doesn't find again was noise."""

    def __init__(self):
        self.cands: list[dict] = []          # {"obj", "seen_from": [camera xy], "hits"}

    def update(self, found: list[Obj], cam_xy, seen_xy, seen_r: float) -> list[Obj]:
        """found: this look's depth-only patches. Returns the ones confirmed by this look (seen twice, from 2+ cm apart)."""
        confirmed, kept = [], []
        for o in found:
            c = next((c for c in self.cands if np.linalg.norm(c["obj"].xy - o.xy) * 1000 < SAME_MM), None)
            if c is None:
                kept.append({"obj": o, "seen_from": [np.asarray(cam_xy)], "hits": 1})
                continue
            c["obj"] = o
            if min(np.linalg.norm(np.asarray(cam_xy) - q) for q in c["seen_from"]) > 0.02:
                c["seen_from"].append(np.asarray(cam_xy))
                c["hits"] += 1
            if c["hits"] >= 2:
                confirmed.append(o)
            else:
                kept.append(c)
            self.cands = [x for x in self.cands if x is not c]  # by identity: == on dicts of arrays throws
        # candidates this look covered and didn't find again: noise
        self.cands = [c for c in self.cands if np.linalg.norm(c["obj"].xy - np.asarray(seen_xy)) >= seen_r] + kept
        return confirmed


def _same_look(a: Obj, b: Obj) -> bool:
    """Could these two measurements be one thing? Not if they differ in kind or in a clear colour (cubes in a pile sit
    closer than SAME_MM: matched by position alone, a yellow cube took over a blue one's id and went in the blue cup)."""
    if (a.kind == "box") != (b.kind == "box"):
        return False
    plain = ("white", "gray", "black", "?")
    return a.color == b.color or a.color in plain or b.color in plain


class Objects:
    """The objects measured so far, ids kept by position (a newer measurement of the same object replaces the older)."""

    def __init__(self):
        self.all: dict[int, Obj] = {}
        self.next_id = 1

    def merge(self, found: list[Obj], seen_xy=None, seen_r: float = 0.0) -> list[Obj]:
        """found: this look's objects. seen_xy, seen_r: the area this look covered well: an object we knew there that
        this look didn't find is gone (picked, or never real)."""
        matched = set()
        for o in found:
            near = [k for k in self.all.values() if k.id not in matched and np.linalg.norm(k.xy - o.xy) * 1000 <
                    max(SAME_MM, 0.4 * min(min(k.size_mm), min(o.size_mm)))  # a 60 mm box's centre wanders more
                    and _same_look(k, o)]
            if near:
                k = min(near, key=lambda k: np.linalg.norm(k.xy - o.xy))
                o.id, o.label, o.looks = k.id, k.label, k.looks + 1
            else:
                o.id = self.next_id
                self.next_id += 1
            matched.add(o.id)
            self.all[o.id] = o
        if seen_xy is not None:
            for i in [i for i, k in self.all.items() if i not in matched
                      and np.linalg.norm(k.xy - np.asarray(seen_xy)) < seen_r]:
                del self.all[i]
        return found

    def parts(self) -> list[Obj]:
        return [o for o in self.all.values() if o.kind == "part"]

    def boxes(self) -> list[Obj]:
        return [o for o in self.all.values() if o.kind == "box"]


def regions(masks: list, table_px) -> list[dict]:
    """Table camera outlines -> where they are on the table: [{"xy", "prompt", "area_px"}]."""
    out = []
    for m in masks:
        v, u = np.nonzero(m["mask"])
        if len(u):
            out.append({"xy": table_px(float(u.mean()), float(v.mean())), "prompt": m["prompt"], "area_px": int(len(u))})
    return out


def spots(regs: list[dict], spread: float = 0.05) -> list[np.ndarray]:
    """Close-look spots covering the regions: regions within `spread` of a spot share it (a look sees ~10 cm)."""
    left = [np.asarray(r["xy"]) for r in regs]
    out = []
    while left:
        c = left.pop(0)
        group = [c] + [p for p in left if np.linalg.norm(p - c) < spread]
        left = [p for p in left if np.linalg.norm(p - c) >= spread]
        out.append(np.mean(group, axis=0))
    return out


def surface_z() -> float:
    return SURFACE_Z
