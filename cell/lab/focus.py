"""Objects from a close look: SAM outlines each object in the wrist camera's picture, and the depth inside each outline
is compared with the table just around it. One question per object (how tall, where is its top face) answered by
hundreds of depth pixels, instead of one question per 2 mm voxel answered by a few: a 10 mm cube stands out of
sensor noise and paper wrinkles (a wrinkle is a gentle slope; the ring around an object is its own local table).

    objs = measure(obs, cams, prompts)       # [Obj]: xy, top-face size and angle, height, colour, what SAM was asked
    Objects().merge(objs)                    # ids kept by position across looks

Also: where the table camera's SAM outlines are on the table (regions worth a close look).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base
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
    kind: str = "part"
    color: str = "?"
    rgb: tuple = (128, 128, 128)
    what: str = ""                  # the SAM prompt that outlined it
    label: str | None = None        # what the ventral stream calls it
    looks: int = 1
    pixels: int = 0                 # depth pixels inside the outline (how much the measurement rests on)
    history: list = field(default_factory=list)

    @property
    def pts(self) -> np.ndarray:
        """Its box's corners, top and bottom (what the picture crop and the table-camera outline need)."""
        a = np.radians(self.angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = np.array([[sx * self.size_mm[0], sy * self.size_mm[1]] for sx in (-.5, .5) for sy in (-.5, .5)]) / 1000 @ R.T + self.xy
        return np.vstack([np.column_stack([c, np.full(4, z)]) for z in (SURFACE_Z, SURFACE_Z + self.top_mm / 1000)])

    @property
    def name(self) -> str:
        return f"{self.color} {self.what or self.kind} #{self.id}"

    def describe(self) -> str:
        return (f"{self.name}: at ({self.xy[0] * 1000:.0f}, {self.xy[1] * 1000:.0f}) mm, {self.size_mm[0]:.0f} x "
                f"{self.size_mm[1]:.0f} mm, {self.top_mm:.0f} mm tall ({self.pixels} depth px, {self.looks} looks)")


def measure(obs: dict, cams: dict, prompts: list[str], masks: list | None = None) -> tuple[list[Obj], list]:
    """One wrist look -> the objects SAM outlines in it that stand out of their local table. Returns (objects, masks)."""
    img = obs["images"]["wrist"]
    H, W = img.shape[:2]
    if masks is None:
        masks = [m for p in prompts for m in sam.segment(img, p, scale=1.0)]
    if not masks:
        return big_from_depth(obs, cams), []
    pts, _ = depth_to_base(cams["wrist_depth"], obs["depth"]["wrist"], with_pixels=True, joints=obs["joints"])
    uv = project_to_image("wrist", cams, pts, (W, H), joints=obs["joints"])
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts, u, v = pts[ok], u[ok], v[ok]
    k = max(1, int(round(RING_PX * W / 1280)))
    anymask = np.zeros((H, W), np.uint8)
    for m in masks:
        anymask |= m["mask"].astype(np.uint8)
    others = cv2.dilate(anymask, np.ones((5, 5), np.uint8)).astype(bool)
    out = []
    for m in _dedupe(masks):
        ys, xs = np.nonzero(m["mask"])
        if len(xs) == 0 or xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
            continue  # cut off by the picture's edge: a look centred on it measures it whole
        mk = m["mask"].astype(np.uint8)
        inner = cv2.erode(mk, np.ones((5, 5), np.uint8)).astype(bool)        # edge pixels mix object and table
        ring = cv2.dilate(mk, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool) & ~others
        ins, rng = inner[v, u], ring[v, u]
        if ins.sum() < 15 or rng.sum() < 30:
            continue
        table = float(np.median(pts[rng, 2]))
        z = pts[ins, 2]
        top = float(np.percentile(z, 90))
        h = (top - table) * 1000
        if h < MIN_HEIGHT_MM:
            continue
        face = pts[ins][z > table + 0.6 * (top - table)]  # the top face (or a cup's rim)
        if len(face) < 6:
            continue
        (cx, cy), (a, b), ang = cv2.minAreaRect((face[:, :2] * 1000).astype(np.float32))
        long_, short = max(a, b), min(a, b)
        angle = (ang if a >= b else ang + 90) % 180
        rgb = tuple(int(c) for c in np.median(img[cv2.erode(mk, np.ones((3, 3), np.uint8)).astype(bool)], axis=0))
        out.append(Obj(0, np.array([cx, cy]) / 1000, (long_, short), angle, h, "box" if short > BOX_MM else "part",
                       colour_name(rgb), rgb, m["prompt"], pixels=int(ins.sum())))
    out += big_from_depth(obs, cams, skip_xy=[o.xy for o in out if o.kind == "box"])
    return out, masks


BIG_MM = 25.0          # depth alone finds anything standing this tall (cups, boxes): no outline needed


def big_from_depth(obs: dict, cams: dict, skip_xy=()) -> list[Obj]:
    """Cups and boxes from depth alone: connected pixels standing over BIG_MM above the table, wider than BOX_MM.
    Depth is good at big things (the real cups came out right); SAM is for the small ones."""
    D = obs["depth"]["wrist"]
    pts, px = depth_to_base(cams["wrist_depth"], D, with_pixels=True, joints=obs["joints"])
    Hd, Wd = D.shape
    tall = np.zeros((Hd, Wd), np.uint8)
    hz = (pts[:, 2] - SURFACE_Z) * 1000
    sel = hz > BIG_MM
    tall[px[sel, 1], px[sel, 0]] = 1
    tall = cv2.morphologyEx(tall, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(tall)
    idx = np.full((Hd, Wd), -1, int)
    idx[px[:, 1], px[:, 0]] = np.arange(len(pts))
    img = obs["images"]["wrist"]
    H, W = img.shape[:2]
    out = []
    for c in range(1, n):
        x, y, w, h, area = st[c]
        if area < 300 or x <= 1 or y <= 1 or x + w >= Wd - 1 or y + h >= Hd - 1:
            continue  # small, or cut off by the edge of the picture
        ii = idx[lab == c]
        ii = ii[ii >= 0]
        if len(ii) < 100:
            continue
        P = pts[ii]
        top = float(np.percentile(P[:, 2], 95))
        face = P[P[:, 2] > SURFACE_Z + 0.8 * (top - SURFACE_Z)]  # the rim
        (cx, cy), (a, b), ang = cv2.minAreaRect((face[:, :2] * 1000).astype(np.float32))
        if min(a, b) <= BOX_MM:
            continue
        if any(np.linalg.norm(np.array([cx, cy]) / 1000 - np.asarray(q)) < 0.02 for q in skip_xy):
            continue
        uv = project_to_image("wrist", cams, face, (W, H), joints=obs["joints"])
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
        rgb = tuple(int(q) for q in np.median(img[v, u], axis=0))
        out.append(Obj(0, np.array([cx, cy]) / 1000, (max(a, b), min(a, b)), (ang if a >= b else ang + 90) % 180,
                       (top - SURFACE_Z) * 1000, "box", colour_name(rgb), rgb, "container (depth)", pixels=len(ii)))
    return out


def _dedupe(masks: list, iou: float = 0.6) -> list:
    keep = []
    for m in sorted(masks, key=lambda m: -m["area"]):
        if all((m["mask"] & k["mask"]).sum() / max((m["mask"] | k["mask"]).sum(), 1) < iou for k in keep):
            keep.append(m)
    return keep


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
            near = [k for k in self.all.values() if np.linalg.norm(k.xy - o.xy) * 1000 < SAME_MM and k.id not in matched]
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
