"""Where the two streams agree and where they don't: the dorsal stream's 3D segments (lab.things: touching evidence in
the map) projected into the table camera's picture, against the ventral stream's SAM masks of the same picture.

Each mask goes to the segment that holds most of it; then every segment and mask falls in one of these cases:

    agree          1 segment, 1 mask                 name the segment from the mask's prompt
    split          1 segment, 2+ masks               the ventral stream sees several things there: break it up
    merged         1 mask over 2+ segments           SAM joined things that are physically apart: trust the gap
    no_mask        a segment no mask covers          SAM missed it, or it isn't an object: ask about a crop
    no_segment     a mask with no segment under it   depth missed it (flat, thin): look closer there

Shown on the table camera (cams/table/rgb): the masks filled, the segments outlined, so a disagreement is visible as
one outline holding two fills, or a fill with no outline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import rerun as rr
from scipy import ndimage
from scipy.spatial import cKDTree

from calibration.geometry import SURFACE_Z, project

OWN = 0.5        # a mask belongs to a segment holding at least this share of it
TOUCH = 0.2      # under this share in every segment: the mask has no segment under it
MIN_SHARE = 0.08  # a mask this small next to its segment is a detail of it (a stud, a shadow), not another thing
DUP_IOU = 0.7    # two masks this alike (from two prompts) are one


@dataclass
class Row:
    seg: int                      # the thing's id (lab.things)
    kind: str
    name: str
    masks: list = field(default_factory=list)  # indices into Agreement.masks
    case: str = "no_mask"


@dataclass
class Agreement:
    rows: list
    masks: list                   # [{"prompt", "area", "box", "mask", "seg", "case", "xy"}]
    image_wh: tuple

    def case_counts(self) -> dict:
        c = {}
        for r in self.rows:
            c[r.case] = c.get(r.case, 0) + 1
        for m in self.masks:
            if m["case"] in ("no_segment", "merged"):
                c[m["case"]] = c.get(m["case"], 0) + 1
        return c

    def text(self, things: dict) -> str:
        """For the LLM: one line per segment, then the masks with no segment."""
        out = []
        for r in self.rows:
            t = things[r.seg]
            labels = [self.masks[j]["prompt"] for j in r.masks]
            say = {"agree": f"agree: SAM sees one '{labels[0]}'" if labels else "agree",
                   "split": f"DISAGREE: SAM sees {len(labels)} things here ({', '.join(labels)}), depth sees one segment",
                   "no_mask": "no SAM mask on it (missed, or not an object)",
                   "merged": "one SAM mask spans this and another segment"}[r.case]
            out.append(f"#{t.id} {t.kind}, {t.color}, {t.size_mm[0]:.0f} x {t.size_mm[1]:.0f} mm, {t.top_mm:.0f} mm tall, "
                       f"at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) mm" + (f", labelled '{t.label}'" if getattr(t, "label", None) else "")
                       + f": {say}")
        for j, m in enumerate(self.masks):
            if m["case"] == "no_segment":
                out.append(f"mask m{j} '{m['prompt']}' near ({m['xy'][0] * 1000:.0f}, {m['xy'][1] * 1000:.0f}) mm: "
                           f"no depth segment under it (too flat to see yet, or not there)")
        return "\n".join(out) or "(nothing on the table yet)"


class TablePixels:
    """Pixels of the table camera's picture -> where that ray meets the table (x, y in m): a lookup from a 2 mm grid."""

    def __init__(self, cam: dict, image_wh):
        xs, ys = np.meshgrid(np.arange(0.10, 0.70, 0.002), np.arange(-0.50, 0.50, 0.002))
        self.xy = np.column_stack([xs.ravel(), ys.ravel()])
        uv = project(cam, np.column_stack([self.xy, np.full(len(self.xy), SURFACE_Z)]))
        self.tree = cKDTree(uv * [image_wh[0] / cam["size"][0], image_wh[1] / cam["size"][1]])

    def __call__(self, u: float, v: float) -> np.ndarray:
        return self.xy[self.tree.query([u, v])[1]]


def focus(things: list, cam: dict, image_wh, margin_m: float = 0.03, target_px: int = 1024) -> tuple:
    """The part of the picture to show SAM: around the parts (boxes left out: they're big and easy), with a margin,
    and the scale that makes it about target_px wide (a 10 mm part is ~8 px in the whole 1920 px frame)."""
    from calibration.geometry import SURFACE_Z
    pts = [t.pts for t in things if t.kind != "box"] or [t.pts for t in things]
    if not pts:
        return (0, 0, *image_wh), 1.0
    p = np.vstack(pts)
    lo, hi = p[:, :2].min(0) - margin_m, p[:, :2].max(0) + margin_m
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (SURFACE_Z, p[:, 2].max())])
    uv = project(cam, c) * [image_wh[0] / cam["size"][0], image_wh[1] / cam["size"][1]]
    x0, y0 = np.clip(uv.min(0), 0, None).astype(int)
    x1, y1 = np.minimum(uv.max(0), image_wh).astype(int)
    return (int(x0), int(y0), int(x1), int(y1)), float(np.clip(target_px / max(x1 - x0, y1 - y0, 1), 1, 6))


def silhouette(t, cam: dict, image_wh) -> np.ndarray:
    """Where segment t falls in the picture: its voxels projected, grown by a voxel's width, holes filled."""
    W, H = image_wh
    uv = project(cam, t.pts) * [W / cam["size"][0], H / cam["size"][1]]
    m = np.zeros((H, W), np.uint8)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m[v[ok], u[ok]] = 1
    # a 2 mm voxel at the table camera's distance is ~1-2 px: close the gaps between them, then fill
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return ndimage.binary_fill_holes(m)


def compare(things: list, masks: list, cam: dict, image_wh, table_px: TablePixels | None = None) -> Agreement:
    """things: lab.things.Thing (known and maybe); masks: lab.sam.segment results on the same picture (same size)."""
    masks = _dedupe(masks)
    sil = [silhouette(t, cam, image_wh) for t in things]
    areas = np.array([s.sum() for s in sil], float)
    rows = [Row(t.id, t.kind, t.name) for t in things]
    table_px = table_px or TablePixels(cam, image_wh)
    for j, m in enumerate(masks):
        a = max(m["mask"].sum(), 1)
        share = np.array([(s & m["mask"]).sum() / a for s in sil]) if sil else np.zeros(0)
        cover = np.array([(s & m["mask"]).sum() / max(areas[i], 1) for i, s in enumerate(sil)]) if sil else np.zeros(0)
        v, u = np.nonzero(m["mask"])
        m["xy"] = table_px(float(u.mean()), float(v.mean())) if len(u) else np.zeros(2)
        m["seg"], m["case"] = None, "no_segment"
        if (cover > OWN).sum() >= 2:                      # covers most of two segments: SAM joined them
            m["case"] = "merged"
            for i in np.nonzero(cover > OWN)[0]:
                rows[i].case = "merged"
            continue
        if len(share) and share.max() >= OWN:
            i = int(share.argmax())
            if things[i].kind == "box" and rows[i].masks:  # a box holds things: one mask is the box, the rest inside it
                continue
            if a < MIN_SHARE * areas[i]:
                m["case"] = "detail"
                continue
            m["seg"], m["case"] = int(things[i].id), "owned"
            rows[i].masks.append(j)
        elif len(share) and share.max() >= TOUCH:
            m["case"] = "partial"
    for r in rows:
        if r.case == "merged":
            continue
        r.case = {0: "no_mask", 1: "agree"}.get(len(r.masks), "split")
    return Agreement(rows, masks, tuple(image_wh))


def _dedupe(masks: list) -> list:
    out = []
    for m in sorted(masks, key=lambda m: -m["area"]):
        same = next((o for o in out if _iou(o["mask"], m["mask"]) > DUP_IOU), None)
        if same is None:
            out.append(dict(m))
        elif m["prompt"] not in same["prompt"].split(" / "):
            same["prompt"] += " / " + m["prompt"]
    return out


def _iou(a, b) -> float:
    i = (a & b).sum()
    return i / max((a | b).sum(), 1)


CASE_COLOUR = {"agree": (60, 200, 90), "split": (240, 70, 60), "merged": (240, 160, 30), "no_mask": (150, 150, 150),
               "no_segment": (60, 160, 255)}


def draw(ag: Agreement, things: list, cam: dict, entity: str = "cams/table/rgb") -> None:
    """On the table camera's picture: each mask filled (its prompt as the label), each segment outlined in the colour
    of its case."""
    W, H = ag.image_wh
    seg = np.zeros((H, W), np.uint16)
    info = [rr.AnnotationInfo(id=0, label="", color=(0, 0, 0, 0))]
    for j, m in enumerate(ag.masks):
        seg[m["mask"]] = j + 1
        col = CASE_COLOUR.get(m["case"], (200, 200, 200)) if m["case"] in ("no_segment", "merged") else (90, 140, 255)
        info.append(rr.AnnotationInfo(id=j + 1, label=f"m{j} {m['prompt']} ({m['case'].replace('_', ' ')})", color=(*col, 150)))
    rr.log(f"{entity}/sam", rr.AnnotationContext(info))
    rr.log(f"{entity}/sam", rr.SegmentationImage(seg, opacity=0.45))
    strips, cols, labels = [], [], []
    by_id = {t.id: t for t in things}
    for r in ag.rows:
        s = silhouette(by_id[r.seg], cam, ag.image_wh).astype(np.uint8)
        cs, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for k, c in enumerate(cs):
            p = c[:, 0, :].astype(float)
            strips.append(np.vstack([p, p[:1]]))
            cols.append(CASE_COLOUR[r.case])
            labels.append(f"#{r.seg} {r.case.replace('_', ' ')}" if k == 0 else "")
    if strips:
        rr.log(f"{entity}/depth_segments", rr.LineStrips2D(strips, colors=cols, labels=labels, radii=0.8))
    else:
        rr.log(f"{entity}/depth_segments", rr.Clear(recursive=False))
