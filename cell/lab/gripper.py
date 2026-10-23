"""The real gripper: SMC MHZL2-16D2 with the cell's custom aluminium fingers.

Geometry and friction copied from the organizers' Isaac Sim starter (fanuc_er4ia_sim: scene_config.py and
gripper_geometry.py, "v0.5" measurements from bench photos and the SMC STEP file), so the lab and the Isaac twin agree.

Each finger is a 76 mm blade, 12 mm deep along the closing direction and 6 mm thick across it. Its inner face is flat,
except near the tip where a half-round recess (sized for an AA battery) leaves a hooked lip: the lips stand 2 mm back
from the face, so the tips are about 16 mm apart when open and 4 mm when closed, while the flat faces go 12 -> 0 mm.
The last 32 mm carry soft contact tips (friction 1.5 static / 1.3 sliding).

Finger frame used here (metres): z up the blade from the tip (0) to its root (0.076); "out" = distance from the inner
face away from the other finger (0 .. 0.012); "across" = -0.003 .. 0.003.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

LENGTH = 0.076          # FINGER_LENGTH
DEPTH = 0.012           # FINGER_WIDTH: along the closing direction
THICK = 0.006           # FINGER_THICK: across
SCALLOP_R = 0.00725     # AA recess radius
SCALLOP_ABOVE_TIP = 0.00697
TIP_BEVEL = 0.014
TIP_WIDTH = 0.0075
EDGE_BREAK = 0.0006
STROKE = 0.006          # per jaw: flat faces 12 mm apart open, 0 closed
INNER_TIP = math.sqrt(SCALLOP_R ** 2 - SCALLOP_ABOVE_TIP ** 2)  # ~2.0 mm: how far the hook lip sits back from the face
SOFT_TIP = 0.032        # soft tips on the last 32 mm
FRICTION_SOFT = 1.5
FRICTION_METAL = 0.8
TIP_CLEARANCE_MM = 2.5  # the twin keeps the tips 2.5 mm off the mat when picking (at 1 mm they dragged on it)


def profile(arc_segments: int = 16) -> list[tuple[float, float]]:
    """The blade outline in (along from the root, out from the inner face), metres. Same as the twin's finger_profile()."""
    L, W, R = LENGTH, DEPTH, SCALLOP_R
    center = L * (1.0 - SCALLOP_ABOVE_TIP / L)
    pts = [(0.0, 0.0), (center - R, 0.0)]
    theta_end = math.acos(max(-1.0, min(1.0, (center - L) / R))) if center + R > L else math.pi
    pts += [(center - R * math.cos(theta_end * k / arc_segments), R * math.sin(theta_end * k / arc_segments))
            for k in range(1, arc_segments + 1)]
    if center + R <= L:
        pts += [(L - EDGE_BREAK, 0.0), (L, EDGE_BREAK)]
    pts += [(L, TIP_WIDTH), (L - TIP_BEVEL, W), (0.0, W)]
    return pts


def _outer(a: float) -> float:
    """Outer edge (out from the face) at distance a along from the root: straight, then the tip bevel."""
    return DEPTH if a <= LENGTH - TIP_BEVEL else DEPTH - (a - (LENGTH - TIP_BEVEL)) / TIP_BEVEL * (DEPTH - TIP_WIDTH)


def pieces(arc_segments: int = 16) -> list[tuple[np.ndarray, float]]:
    """The blade as convex chunks for physics: [(8 vertices (z, out, across) as (N, 3) in the finger frame, friction)].
    The recess makes the blade non-convex, so it is sliced along its length between the recess and the outer edge."""
    prof = profile(arc_segments)
    inner = [p for p in prof if p[0] >= prof[1][0] - 1e-12 and p[0] <= LENGTH and p[1] <= TIP_WIDTH + 1e-9][:arc_segments + 1]
    slabs = [(0.0, LENGTH - SOFT_TIP, 0.0, 0.0), (LENGTH - SOFT_TIP, inner[0][0], 0.0, 0.0)]
    slabs += [(a0, a1, b0, b1) for (a0, b0), (a1, b1) in zip(inner, inner[1:])]
    out = []
    for a0, a1, b0, b1 in slabs:
        quad = [(a0, b0), (a1, b1), (a1, _outer(a1)), (a0, _outer(a0))]
        v = np.array([[LENGTH - a, b, s * THICK / 2] for a, b in quad for s in (-1, 1)])  # z up from the tip
        out.append((v, FRICTION_SOFT if a1 > LENGTH - SOFT_TIP + 1e-9 else FRICTION_METAL))
    return out


def finger_mesh(side: int) -> trimesh.Trimesh:
    """One finger as a mesh in the hand frame with the jaw fully open: side -1 sits at negative x, +1 at positive x.
    Hand frame: origin at the fingertip centre, z up the fingers, x = closing direction."""
    parts = []
    for v, _ in pieces():
        x = side * (STROKE + v[:, 1])
        parts.append(trimesh.Trimesh(np.column_stack([x, v[:, 2], v[:, 0]])).convex_hull)
    return trimesh.util.concatenate(parts)


def piece_vertices_hand(side: int) -> list[tuple[np.ndarray, float]]:
    """Physics chunks in the finger body's own frame (the body sits at x = side * STROKE when open)."""
    return [(np.column_stack([side * v[:, 1], v[:, 2], v[:, 0]]), f) for v, f in pieces()]


BODY = (0.047, 0.0236, 0.0527)   # SMC MHZL2-16D2 body: along closing, across, height (from the STEP)
SPACER_R, SPACER_LEN = 0.023, 0.0723  # round spacer above it, up to the faceplate


def finger_body_mesh(side: int) -> trimesh.Trimesh:
    """One finger in its own body frame (the body sits at x = side * STROKE when the jaw is open)."""
    return trimesh.util.concatenate([trimesh.Trimesh(v).convex_hull for v, _ in piece_vertices_hand(side)])


def body_meshes() -> list[tuple[trimesh.Trimesh, tuple]]:
    """Gripper body and spacer in the hand frame, with display colours."""
    b = trimesh.creation.box(BODY)
    b.apply_translation([0, 0, LENGTH + BODY[2] / 2])
    c = trimesh.creation.cylinder(SPACER_R, SPACER_LEN, sections=32)
    c.apply_translation([0, 0, LENGTH + BODY[2] + SPACER_LEN / 2])
    return [(b, (70, 70, 75)), (c, (45, 45, 50))]
