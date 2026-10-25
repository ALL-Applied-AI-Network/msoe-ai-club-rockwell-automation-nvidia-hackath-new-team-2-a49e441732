"""The robot's mind, drawn so every dot can be explained: one layer per kind of belief, each its own entity (the left
tree in Rerun names them, in order, and its eye icons hide them), every point tagged with what it means (hover it), the
evidence points carrying their numbers (click one), and a legend with live counts.

    1 depth now      what the camera just measured, as points (cyan): raw input, before any judging
    2 table floor    per 4 mm column of the table: confirmed table, probably table, looked but unclear, never seen
    3 unknown space  a 1 cm lattice of the space above the table that no ray has passed through yet
    4 gaps           unknown space touching known-empty space: the edge of the known, what the next look aims at
    5 solid          confirmed solid above the table (several agreeing sharp readings), coloured by height
    6 maybe solid    some evidence of solid, not confirmed yet (dim)
    7 things         touching evidence grouped as one thing, with its size, colour and how sure: a box per thing
    plan, arm        the next look, the chosen grip; the arm as the robot believes it is

Nothing here comes from the simulator's truth: only what the robot itself has seen and worked out.
"""
from __future__ import annotations

import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z, flange_pose

DEPTH, NEVER, UNCLEAR, TABLE_MAYBE, TABLE, UNKNOWN, GAP, SOLID, LEANING, PART, BOX, MAYBE = range(1, 13)
CLASSES = {
    DEPTH: ("depth just measured by the camera (raw reading, not judged yet)", (0, 200, 255)),
    NEVER: ("table floor no camera has seen yet", (130, 60, 190)),
    UNCLEAR: ("table floor looked at, but no clear table reading yet (hidden, or only blurry readings)", (55, 58, 70)),
    TABLE_MAYBE: ("probably table: faint or far readings agree", (105, 112, 125)),
    TABLE: ("table, confirmed", (175, 182, 192)),
    UNKNOWN: ("space above the table no ray has passed through yet", (150, 110, 200)),
    GAP: ("gap: unknown space at the edge of what's known (the next look aims at these)", (235, 70, 200)),
    SOLID: ("solid, confirmed: several agreeing sharp readings (colour = height)", (255, 140, 40)),
    LEANING: ("maybe solid: some evidence, not confirmed yet", (150, 95, 45)),
    PART: ("a thing: touching evidence taken as one part", (60, 220, 120)),
    BOX: ("a thing: a box (wider than 40 mm)", (230, 230, 230)),
    MAYBE: ("maybe a thing: not sure enough to act on yet (see why in its label)", (140, 140, 140)),
}
# the layers, in the order the left tree lists them (Rerun sorts by name)
L_DEPTH, L_FLOOR, L_UNKNOWN, L_GAPS, L_SOLID, L_LEANING, L_THINGS = (
    "mind/1_depth_now", "mind/2_table_floor", "mind/3_unknown_space", "mind/4_gaps", "mind/5_solid",
    "mind/6_maybe_solid", "mind/7_things")
SWATCH = {DEPTH: "🟦", NEVER: "🟪", UNCLEAR: "⬛", TABLE_MAYBE: "◻️", TABLE: "⬜", UNKNOWN: "🟣", GAP: "🩷",
          SOLID: "🟧", LEANING: "🟫", PART: "🟩", BOX: "▫️", MAYBE: "⚪"}


def context() -> None:
    """What each class of point means: Rerun shows the label when a point is hovered."""
    rr.log("mind", rr.AnnotationContext([rr.AnnotationInfo(id=c, label=t, color=col) for c, (t, col) in CLASSES.items()]))


def depth_now(depth_mm, cam: dict, joints=None, which: str = "wrist", max_points: int = 25000) -> int:
    """The latest depth image as points in the robot's frame, cropped to the table area: what the camera said."""
    T = cam["T"] if cam["frame"] == "base" else flange_pose(joints) @ cam["T"]
    D = np.asarray(depth_mm, np.float32) / 1000.0
    H, W = D.shape
    s = max(1, int(np.ceil(np.sqrt(H * W / (max_points * 2)))))
    v, u = np.mgrid[0:H:s, 0:W:s]
    d = D[::s, ::s]
    ok = d > 0
    u, v, d = u[ok], v[ok], d[ok]
    pc = np.stack([(u - cam["cx"]) / cam["fx"] * d, (v - cam["cy"]) / cam["fy"] * d, d], -1)
    p = pc @ T[:3, :3].T + T[:3, 3]
    keep = (p[:, 0] > 0.1) & (p[:, 0] < 0.7) & (np.abs(p[:, 1]) < 0.5) & (p[:, 2] > SURFACE_Z - 0.02) & (p[:, 2] < SURFACE_Z + 0.15)
    p = p[keep][:max_points]
    rr.log(L_DEPTH, rr.Points3D(p, class_ids=np.full(len(p), DEPTH), radii=0.0007),
           rr.AnyValues(camera=which, distance_mm=np.round(d[keep][:max_points] * 1000).astype(np.float32)))
    return len(p)


def evidence(m) -> dict:
    """Layers 2-6 from the evidence map; returns the counts the legend shows."""
    nx, ny, nz = m.shape
    step = 2
    band = slice(max(0, m.k_table - 1), m.k_table + 2)
    floor_seen = (m.looks[:, :, band] > 0).any(2)
    occupied = ((m.solid() | m.leaning()) & m.clear_of_table()).any(2)
    cls = np.full((nx, ny), NEVER, np.uint16)
    cls[floor_seen] = UNCLEAR
    cls[m.table_found(sure=False)] = TABLE_MAYBE
    cls[m.table_found()] = TABLE
    counts = {c: int((cls[~occupied] == c).sum()) for c in (NEVER, UNCLEAR, TABLE_MAYBE, TABLE)}
    counts["occupied"] = int(occupied.sum())
    sub = cls[::step, ::step]
    keep = ~occupied[::step, ::step]
    i, j = np.nonzero(keep)
    pts = m.centers(i * step, j * step, np.full(len(i), m.k_table))
    pts[:, 2] = SURFACE_Z
    rr.log(L_FLOOR, rr.Points3D(pts, class_ids=sub[keep], radii=m.v * step * 0.45))

    lat = m.lattice_unknown()
    rr.log(L_UNKNOWN, rr.Points3D(lat, class_ids=np.full(len(lat), UNKNOWN), radii=0.0008))

    gaps = m.frontier_points()
    counts["gaps"] = len(gaps)
    if len(gaps) > 20000:
        gaps = gaps[np.linspace(0, len(gaps) - 1, 20000).astype(int)]
    rr.log(L_GAPS, rr.Points3D(gaps, class_ids=np.full(len(gaps), GAP), radii=m.v * 0.35))

    above = m.clear_of_table()
    for path, mask, c, dim in ((L_SOLID, m.solid() & above, SOLID, 1.0),
                               (L_LEANING, m.leaning() & ~m.solid() & above, LEANING, 0.55)):
        idx = np.nonzero(mask)
        p = m.centers(*idx)
        counts[c] = len(p)
        if not len(p):
            rr.log(path, rr.Clear(recursive=False))
            continue
        h_mm = (p[:, 2] - SURFACE_Z) * 1000
        h = np.clip(h_mm / 50, 0, 1)
        col = (np.stack([np.full_like(h, 255), 200 * (1 - h) + 40, np.full_like(h, 40)], 1) * dim).astype(np.uint8)
        rr.log(path, rr.Points3D(p, class_ids=np.full(len(p), c), colors=col, radii=m.v * (0.55 if c == SOLID else 0.45)),
               rr.AnyValues(height_mm=np.round(h_mm, 1).astype(np.float32),
                            confidence=np.round(m.L[idx], 2).astype(np.float32),
                            times_seen=m.looks[idx].astype(np.float32),
                            sharpest_reading_mm=np.round(m.sharpest[idx].astype(np.float32) * 1000, 1)))
    a = m.above_table()
    counts["known_pct"] = round(100 * float((~m.unknown() & a).sum()) / max(int(a.sum()), 1), 1)
    return counts


def why_maybe(t) -> str:
    from lab.things import MIN_LAYERS, SHARP_MM, SURE
    r = []
    if t.sharp_mm > SHARP_MM:
        r.append(f"only blurry readings ({t.sharp_mm:.0f} mm error; needs ≤ {SHARP_MM:.0f})")
    if t.sure < SURE:
        r.append(f"only {t.sure:.0%} of it confirmed (needs {SURE:.0%})")
    if t.layers < MIN_LAYERS:
        r.append(f"confirmed at {t.layers} height only (needs {MIN_LAYERS})")
    return "; ".join(r) or "not sure enough"


def things(known, maybe=()) -> None:
    """Layer 7: each group of touching evidence as one box, labelled with what the robot makes of it."""
    rr.log(L_THINGS, rr.Clear(recursive=True))
    for t, sure in [(t, True) for t in known] + [(t, False) for t in maybe]:
        h = max(t.top_mm, 2.0) / 1000
        dims = f"{t.size_mm[0]:.0f} × {t.size_mm[1]:.0f} mm, {t.top_mm:.0f} mm tall"
        if sure:
            c = BOX if t.kind == "box" else PART
            label = f"{t.name} · {dims}"
            col = list(t.rgb) if getattr(t, "rgb", None) is not None else list(CLASSES[c][1])
        else:
            c = MAYBE
            label = f"maybe? {dims} · {why_maybe(t)}"
            col = list(CLASSES[MAYBE][1])
        rr.log(f"{L_THINGS}/{'t' if sure else 'maybe'}{t.id if sure else id(t) % 10000}",
               rr.Boxes3D(centers=[[*t.xy, SURFACE_Z + h / 2]], half_sizes=[[t.size_mm[0] / 2000, t.size_mm[1] / 2000, h / 2]],
                          rotation_axis_angles=[rr.RotationAxisAngle([0, 0, 1], degrees=t.angle)],
                          colors=[col], labels=[label], class_ids=[c], radii=0.0006 if sure else 0.0003),
               rr.AnyValues(voxels=float(t.voxels), confirmed_share=round(float(t.sure), 2),
                            sharpest_mm=round(float(t.sharp_mm), 1), heights_confirmed=float(t.layers)))


def legend(counts: dict, looks: int, known=(), maybe=(), job: str = "") -> None:
    """The key to the mind view, with live numbers: what each colour is, and how much of it there is right now."""
    floor = sum(counts.get(c, 0) for c in (NEVER, UNCLEAR, TABLE_MAYBE, TABLE)) or 1
    pct = lambda c: f"{100 * counts.get(c, 0) / floor:.0f}%"
    parts = [t for t in known if t.kind == "part"]
    boxes = [t for t in known if t.kind == "box"]
    row = lambda c, extra="": f"| {SWATCH[c]} | {CLASSES[c][0]} | {extra} |"
    md = [f"### What the robot believes{(' · ' + job) if job else ''}",
          f"After **{looks} look{'s' if looks != 1 else ''}**: {counts.get('known_pct', 0)}% of the space above the table "
          f"has evidence; **{len(parts)} parts, {len(boxes)} boxes**" + (f", {len(maybe)} maybe" if maybe else "") + ".",
          "", "| | what it is | now |", "|---|---|---|",
          row(DEPTH, "last look only"),
          "| | **the table floor** (every 4 mm) | |",
          row(TABLE, pct(TABLE)), row(TABLE_MAYBE, pct(TABLE_MAYBE)), row(UNCLEAR, pct(UNCLEAR)), row(NEVER, pct(NEVER)),
          "| | **the space above it** | |",
          row(UNKNOWN, f"{100 - counts.get('known_pct', 0):.0f}% of it"),
          row(GAP, f"{counts.get('gaps', 0):,} voxels"),
          row(SOLID, f"{counts.get(SOLID, 0):,} voxels"), row(LEANING, f"{counts.get(LEANING, 0):,} voxels"),
          "| | **what it makes of it** | |",
          row(PART, str(len(parts))), row(BOX, str(len(boxes))), row(MAYBE, str(len(maybe))),
          "", "Empty space it has seen through is drawn as nothing. A voxel is 2 mm.",
          "", "**Hover** a point: what it is. **Click** a solid point: its height, confidence, how many looks saw it, "
              "and its sharpest reading. The **eye icons** in the left tree hide a layer."]
    if known or maybe:
        md += ["", "| thing | size (mm) | tall | sure | why |", "|---|---|---|---|---|"]
        md += [f"| {t.name} | {t.size_mm[0]:.0f} × {t.size_mm[1]:.0f} | {t.top_mm:.0f} | {t.sure:.0%} | "
               f"touching evidence, sharpest reading {t.sharp_mm:.1f} mm |" for t in known]
        md += [f"| maybe at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) | {t.size_mm[0]:.0f} × {t.size_mm[1]:.0f} | "
               f"{t.top_mm:.0f} | {t.sure:.0%} | {why_maybe(t)} |" for t in maybe]
    rr.log("legend", rr.TextDocument("\n".join(md), media_type=rr.MediaType.MARKDOWN))
