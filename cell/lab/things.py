"""Things on the table, read off the evidence map: connected solid evidence clear of the table, described well enough
to pick (where, footprint and face angle, height) and to sort (colour from the table camera, box or part).

    things = Things()
    found = things.update(emap, table_rgb, cams)      # list[Thing]; ids carry over between updates
    report = score(found, cell.object_meshes())       # sim only: against the truth

Deterministic: no model. Noise is handled by the map (a lone speck never becomes solid) and by size: a thing has to
be bigger than MIN_VOXELS to count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from calibration.geometry import SURFACE_Z, project

MIN_VOXELS = 12        # at 2 mm voxels: ~100 mm^3 of surface evidence (a 10 mm cube's top alone is ~25 voxels)
BOX_MM = 40.0          # a footprint this wide on both sides: a box to sort into, not a part
MATCH_MM = 15.0        # an update's thing this close to an earlier one keeps its id
SURE = 0.4             # and once this share of its voxels is confirmed solid (real parts: 0.73-0.93; noise that
                       # leans solid but never got confirmed: ~0.1)
MIN_LAYERS = 2         # and confirmed evidence at two or more heights: table noise that passes the rest is one 2 mm
                       # layer at 7-9 mm (in Isaac it made "white parts" near the robot, and the arm went to pick them)
SHARP_MM = 4.0         # a thing counts once some reading of it was this precise (expected error along the ray); seen
                       # only by far, noisy readings (the table camera, ~1 cm out there) it's a "maybe": look closer


@dataclass
class Thing:
    id: int
    xy: np.ndarray                 # centre of the footprint, base frame, m
    size_mm: tuple                 # footprint (long, short), mm
    angle: float                   # the long side's direction, deg (mod 180)
    top_mm: float                  # highest evidence above the table
    voxels: int
    sure: float                    # 0..1: how much of it is confirmed solid (vs only leaning)
    color: str = "?"
    rgb: tuple = (128, 128, 128)
    kind: str = "part"             # part | box
    sharp_mm: float = 99.0         # the most precise reading among its voxels (median of their best), mm
    layers: int = 0                # distinct voxel heights with confirmed evidence
    pts: np.ndarray = field(default=None, repr=False)   # its voxel centres (base, m)

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind} #{self.id}"

    def describe(self) -> str:
        return (f"{self.name}: at ({self.xy[0] * 1000:.0f}, {self.xy[1] * 1000:.0f}) mm, "
                f"{self.size_mm[0]:.0f} x {self.size_mm[1]:.0f} mm, {self.top_mm:.0f} mm tall")


def colour_name(rgb) -> str:
    """A colour word by hue (works on real colours, not just the sim's exact ones): low saturation -> white / gray /
    black by brightness; otherwise the hue band."""
    px = np.uint8([[np.clip(rgb, 0, 255)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0].astype(float)  # h 0..180
    if s < 60:
        return "white" if v > 190 else "black" if v < 70 else "gray"
    if h < 8 or h >= 165:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


class Things:
    def __init__(self):
        self.known: list[Thing] = []
        self.maybe: list[Thing] = []   # seen only by imprecise readings: worth a closer look, not a pick
        self.next_id = 1

    def update(self, emap, table_rgb=None, cams=None) -> list[Thing]:
        occ = (emap.solid() | emap.leaning()) & emap.clear_of_table()
        lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        if n == 0:
            self.known = []
            return []
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        solid = emap.solid()
        found = []
        for c in np.nonzero(sizes >= MIN_VOXELS)[0] + 1:
            idx = np.argwhere(lab == c)
            t = self._describe(emap, idx, solid, table_rgb, cams)
            if t.kind == "part" and table_rgb is not None and cams is not None:
                parts = _split_by_colour(emap, idx, t, table_rgb, cams["table"])
                if len(parts) > 1:  # touching parts of different colours: one thing per colour
                    found += [self._describe(emap, sub, solid, table_rgb, cams) for sub in parts if len(sub) >= MIN_VOXELS]
                    continue
            found.append(t)
        found = _absorb_into_boxes(found)
        _name_pale_boxes(found)
        ok = lambda t: t.sharp_mm <= SHARP_MM and t.sure >= SURE and t.layers >= MIN_LAYERS
        self.maybe = [t for t in found if not ok(t)]
        found = [t for t in found if ok(t)]
        self._keep_ids(found)
        self.known = found
        return found

    @staticmethod
    def _describe(emap, idx, solid, table_rgb, cams) -> Thing:
        pts = emap.centers(*idx.T)
        # size and height from the top face: seen from above, its edges sit where they are (sides are seen at
        # grazing angles and smear outwards: a 10 mm cube measured 16-22 mm across that way)
        sure = solid[tuple(idx.T)]
        core = pts[sure] if sure.sum() >= 6 else pts  # confirmed voxels only, when there are enough
        top_z = np.percentile(core[:, 2], 95)
        top = core[core[:, 2] > top_z - 0.003]
        (cx, cy), (w, h), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
        w, h = w + emap.v * 1000, h + emap.v * 1000  # voxel centres: the surface reaches half a voxel further
        long_, short = max(w, h), min(w, h)
        angle = (ang if w >= h else ang + 90) % 180
        t = Thing(0, np.array([cx, cy]) / 1000, (long_, short), angle,
                  float((np.median(top[:, 2]) - SURFACE_Z) * 1000 + emap.v * 500), len(idx), float(sure.mean()), pts=pts)
        t.kind = "box" if short > BOX_MM else "part"
        t.sharp_mm = float(np.median(emap.sharpest[tuple(idx.T)].astype(float)) * 1000)
        t.layers = len(np.unique(idx[sure, 2]))  # distinct heights of confirmed evidence
        if table_rgb is not None and cams is not None:
            t.rgb = _sample_colour(t, table_rgb, cams["table"])
            t.color = colour_name(t.rgb)
        return t

    def _keep_ids(self, found: list[Thing]) -> None:
        """Nearest earlier thing within MATCH_MM keeps its id (greedy, closest pairs first); the rest get new ids."""
        pairs = sorted(((np.linalg.norm(f.xy - k.xy), i, j) for i, f in enumerate(found) for j, k in enumerate(self.known)),
                       key=lambda p: p[0])
        used_f, used_k = set(), set()
        for d, i, j in pairs:
            if d * 1000 > MATCH_MM or i in used_f or j in used_k:
                continue
            found[i].id = self.known[j].id
            used_f.add(i); used_k.add(j)
        for i, f in enumerate(found):
            if i not in used_f:
                f.id = self.next_id
                self.next_id += 1


def _split_by_colour(emap, idx, t: Thing, img, cam) -> list[np.ndarray]:
    """A blob whose top shows two or more strong colours (touching parts: a red and a blue cube came out as one
    64 mm "purple part") split into one voxel set per colour: each voxel's column takes the colour seen above it."""
    pts = emap.centers(*idx.T)
    uv = project(cam, pts) * [img.shape[1] / cam["size"][0], img.shape[0] / cam["size"][1]]
    H, W = img.shape[:2]
    u, v = np.clip(uv[:, 0].astype(int), 0, W - 1), np.clip(uv[:, 1].astype(int), 0, H - 1)
    px = img[v, u].astype(np.uint8)
    hsv = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0]
    vivid = hsv[:, 1] > 90
    if vivid.sum() < 12:
        return [idx]
    names = np.array([colour_name(c) if ok else "" for c, ok in zip(px, vivid)])
    kinds, counts = np.unique(names[vivid], return_counts=True)
    big = kinds[(counts >= 6) & (counts >= 0.2 * vivid.sum())]
    if len(big) < 2:
        return [idx]
    # every voxel joins the colour of the nearest vivid voxel (in xy): sides and pale pixels follow their top
    from scipy.spatial import cKDTree
    keep = np.isin(names, big)
    tree = cKDTree(pts[keep, :2])
    _, j = tree.query(pts[:, :2])
    label = names[keep][j]
    return [idx[label == k] for k in big]


WALL_MM = 25.0  # a "part" this tall inside a box's footprint is a piece of the box's wall (a part in the well sits lower)


def _absorb_into_boxes(found: list[Thing]) -> list[Thing]:
    """Slivers of a box's walls that came out as separate things (noise breaks the rim) go back into the box."""
    boxes = [t for t in found if t.kind == "box"]
    keep = []
    for t in found:
        if t.kind == "part" and t.top_mm > WALL_MM:
            host = next((b for b in boxes if _inside(t.xy, b, margin_mm=8)), None)
            if host is not None:
                host.pts = np.vstack([host.pts, t.pts])
                host.voxels += t.voxels
                continue
        keep.append(t)
    return keep


def _name_pale_boxes(found: list[Thing]) -> None:
    """Colourless boxes named against each other, not by absolute brightness (shade turns white gray and gray black):
    of two or more, the brightest is white, the darkest black, any between gray."""
    pale = [t for t in found if t.kind == "box" and t.color in ("white", "gray", "black")]
    if len(pale) < 2:
        return
    pale.sort(key=lambda t: -sum(t.rgb))
    for k, t in enumerate(pale):
        t.color = "white" if k == 0 else "black" if k == len(pale) - 1 else "gray"


def _inside(xy, box: Thing, margin_mm: float = 0.0) -> bool:
    """Is xy within the box's footprint (+ margin)?"""
    a = np.radians(box.angle)
    d = (np.asarray(xy) - box.xy) * 1000
    along, across = abs(d @ [np.cos(a), np.sin(a)]), abs(d @ [-np.sin(a), np.cos(a)])
    return along <= box.size_mm[0] / 2 + margin_mm and across <= box.size_mm[1] / 2 + margin_mm


def _sample_colour(t: Thing, img: np.ndarray, cam: dict) -> tuple:
    """Median colour of the thing's top surface in the table camera's image (the top few mm, where the camera sees it
    head-on; sides are shaded). For a box, its rim."""
    z = t.pts[:, 2]
    top = t.pts[z > z.max() - 0.004]
    uv = project(cam, top)
    H, W = img.shape[:2]
    uv = uv * [W / cam["size"][0], H / cam["size"][1]]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not ok.any():
        return (128, 128, 128)
    u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
    # a 10 mm part is 3-4 px in a 640 x 360 table image: look at a small patch around each point, and if a good share
    # of it is strongly coloured (a part on a pale table), that's the part's colour; otherwise it's a pale part
    du, dv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    U = np.clip((u[:, None] + du.ravel()).ravel(), 0, W - 1)
    V = np.clip((v[:, None] + dv.ravel()).ravel(), 0, H - 1)
    px = img[V, U].astype(np.uint8)
    sat = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0, :, 1]
    vivid = px[sat > 90]
    if len(vivid) >= max(4, 0.08 * len(px)):
        return tuple(int(c) for c in np.median(vivid, axis=0))
    return tuple(int(c) for c in np.median(img[v, u], axis=0))


def score(found: list[Thing], truth: list[dict]) -> dict:
    """Against the sim's truth meshes: each true part matched to the nearest found thing within 12 mm (centre to
    centre, xy); found things matching nothing are phantoms. Boxes matched the same way, within 20 mm."""
    parts = [o for o in truth if not o["kind"].startswith("box")]
    boxes = [o for o in truth if o["kind"].startswith("box")]
    out = {"parts": len(parts), "found": 0, "missed": [], "phantoms": [], "pos_err_mm": [], "height_err_mm": [],
           "colour_right": 0, "boxes": len(boxes), "boxes_found": 0, "merged": 0}
    free = list(range(len(found)))
    for group, tol, is_box in ((parts, 12.0, False), (boxes, 20.0, True)):
        for o in group:
            m = o["mesh"]
            c = (m.bounds[0][:2] + m.bounds[1][:2]) / 2
            cand = [(np.linalg.norm(found[i].xy - c) * 1000, i) for i in free if (found[i].kind == "box") == is_box]
            if not cand or min(cand)[0] > tol:
                if not is_box:
                    out["missed"].append(o["name"])
                continue
            d, i = min(cand)
            free.remove(i)
            if is_box:
                out["boxes_found"] += 1
                continue
            out["found"] += 1
            out["pos_err_mm"].append(round(d, 1))
            out["height_err_mm"].append(round(found[i].top_mm - (m.bounds[1][2] - SURFACE_Z) * 1000, 1))
            out["colour_right"] += int(found[i].color == o["color"] or (o["color"] == "steel" and found[i].color in ("gray", "white")))
    # a leftover thing that covers a true part's centre is several parts merged into one, not a phantom
    for i in free:
        t = found[i]
        covers = sum(1 for o in parts if np.min(np.linalg.norm(t.pts[:, :2] - (o["mesh"].bounds[0][:2] + o["mesh"].bounds[1][:2]) / 2, axis=1)) < 0.004)
        if covers:
            out["merged"] += 1
        else:
            out["phantoms"].append(t.describe())
    return out
