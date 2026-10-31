"""Look: one still shot (wrist colour + wrist depth, the arm not moving) -> the things in it, measured.

    seen = measure(Shot(colour, depth, joints), ["pen", "cube", "cup"])

For each word SAM finds disjoint masks in the colour picture (lab/sam: its own GPU, ~0.4 s). The depth picture's points
are placed with the same pose and their heights taken from the table this shot sees (lab/stills_depth.table_relative:
errors in where the camera was cancel, the table and the thing share them). Each mask then gets the depth points that
fall inside it (projected into the colour picture), its edge trimmed (depth smears down at a small thing's edges):
    real      enough of them stand above the table: a thing, not a sticker or a printed picture
    top_mm    how tall (the 90th percentile of its points): good for things big enough to fill the depth picture
              (cups, pens); for a small block it's rough, and only `real` is relied on
    centre, angle, size   from the mask's outline, laid on the plane at its measured top (or the table, if not real)
Nothing here moves the arm or needs timing: a still shot's pose is the joints it came with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, flange_pose, load, project
from lab.stills_depth import table_relative

REAL_MM = 4.0       # a point this far above the table is part of a thing
REAL_SHARE = 0.3    # this share of a mask's points must be: else it's flat (a sticker, a shadow, a printed picture)
EDGE_PX = 3         # trimmed off a mask's edge before reading its depth


@dataclass
class Shot:
    colour: np.ndarray      # RGB, the wrist colour camera's size (1280 x 720)
    depth: np.ndarray       # uint16 mm, the wrist depth camera's (848 x 480), 0 = none
    joints: np.ndarray      # J1-J6 (deg) when it was taken, the arm still
    id: str = ""


@dataclass
class Seen:
    word: str
    centre: np.ndarray      # base frame xy (m)
    angle_deg: float        # of the outline's long side
    size_mm: tuple          # long, short side of the outline
    top_mm: float | None    # height above the table (None: too few depth points)
    real: bool              # stands up off the table
    points: int             # depth points inside the mask
    mask: np.ndarray = field(repr=False, default=None)
    shot: str = ""
    score: float = 0.0


def _cams():
    c = load()
    return c["wrist"], c["wrist_depth"]


def heights(shot: Shot, rng=None):
    """The shot's depth points (base m) and each one's height above this shot's table (mm)."""
    rng = rng or np.random.default_rng(0)
    _, cd = _cams()
    p = depth_to_base(cd, shot.depth, stride=2, joints=shot.joints)  # (half resolution: plenty for heights)
    if len(p) < 500:
        return p, None
    return p, table_relative(p, rng)


def measure(shot: Shot, words: list[str], sam_fn=None, min_area: int = 60, most: int = 40) -> list[Seen]:
    from lab import sam
    sam_fn = sam_fn or sam.segment_many
    cw, _ = _cams()
    camd = dict(cw, T=flange_pose(shot.joints[:6]) @ cw["T"])
    p, h = heights(shot)
    uv = project(camd, p) if h is not None else np.zeros((0, 2))
    H, W = shot.colour.shape[:2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) if len(uv) else np.zeros(0, bool)
    ui, vi = uv[ok].astype(int).T if ok.any() else (np.zeros(0, int), np.zeros(0, int))
    hp = h[ok] if h is not None else np.zeros(0)
    found = sam_fn(shot.colour, words, scale=1.0, timeout=30.0)  # (the wrist picture as it is: 1280 x 720)
    found = sorted(found, key=lambda d: -d.get("area", 0))[:most]
    out = []
    for d in found:
        m = d["mask"]
        if m.sum() < min_area:
            continue
        core = cv2.erode(m.astype(np.uint8), np.ones((2 * EDGE_PX + 1, 2 * EDGE_PX + 1), np.uint8)).astype(bool)
        inside = core[vi, ui] if len(ui) else np.zeros(0, bool)
        hs = hp[inside]
        real = bool(len(hs) >= 5 and (hs > REAL_MM).mean() >= REAL_SHARE)
        top = float(np.percentile(hs, 90)) if len(hs) >= 5 else None
        # the outline, laid on the plane at its top (or on the table): its centre, angle and size in the robot's frame
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cnt, key=cv2.contourArea).reshape(-1, 2).astype(float)
        z = SURFACE_Z + (top / 1000 if (real and top) else 0.0)
        xy = _on_plane(camd, pts, z)
        (cx, cy), (a, b), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
        if a < b:
            a, b, ang = b, a, ang + 90
        out.append(Seen(d["prompt"], np.array([cx, cy]) / 1000, float(ang % 180), (round(a, 1), round(b, 1)),
                        None if top is None else round(top, 1), real, int(inside.sum()), None, shot.id,
                        float(d.get("score", 0.0))))
    return out


def _on_plane(cam, uv, z):
    from lab.scene3d import rays_to_plane
    return rays_to_plane(dict(cam, size=tuple(cam["size"])), np.asarray(uv, float), z)[:, :2]
