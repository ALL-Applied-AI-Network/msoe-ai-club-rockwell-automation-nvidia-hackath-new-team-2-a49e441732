"""The bag of objects: every kind of part the lab can drop on the table.

Each kind is built from simple shapes (box, cylinder, capsule, sphere, convex shell), so one definition gives both the
physics body (MuJoCo) and the mesh the cameras see (trimesh). Sizes are in millimetres.

Add a kind: write a function returning Parts and register it in KINDS. Real CAD (STL/OBJ) can come in as a "convex" part
for convex shapes; non-convex CAD needs splitting into convex pieces first (CoACD).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

COLORS = {
    "red": (200, 35, 35), "yellow": (235, 200, 30), "blue": (35, 80, 205), "green": (40, 160, 70),
    "orange": (235, 120, 25), "purple": (130, 60, 170), "white": (240, 240, 235), "black": (35, 35, 38),
    "gray": (150, 150, 155), "steel": (165, 168, 175), "gold": (212, 175, 55), "wood": (215, 185, 140),
    "olive": (95, 105, 50), "tan": (200, 170, 120),
}
DENSITY = {"plastic": 1100.0, "steel": 7850.0, "wood": 600.0, "ceramic": 2000.0, "brass": 8500.0}


@dataclass
class Part:
    kind: str                      # box | cylinder | capsule | sphere | convex
    size: tuple = ()               # box: (x, y, z); cylinder: (radius, height); capsule: (radius, length); sphere: (radius,)
    pos: tuple = (0.0, 0.0, 0.0)   # mm, in the object's frame
    rot: tuple = (0.0, 0.0, 0.0)   # degrees about x, then y, then z (fixed axes)
    vertices: np.ndarray | None = None  # convex: (N, 3) mm


@dataclass
class Kind:
    name: str
    parts: list[Part]
    color: str = "gray"
    material: str = "plastic"
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def rot_matrix(rot_deg) -> np.ndarray:
    rx, ry, rz = np.radians(rot_deg)
    return trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]


def part_mesh(p: Part) -> trimesh.Trimesh:
    s = np.asarray(p.size, float) / 1000.0
    if p.kind == "box":
        m = trimesh.creation.box(s)
    elif p.kind == "cylinder":
        m = trimesh.creation.cylinder(radius=s[0], height=s[1], sections=24)
    elif p.kind == "capsule":
        m = trimesh.creation.capsule(height=s[1], radius=s[0], count=[12, 12])
    elif p.kind == "sphere":
        m = trimesh.creation.icosphere(subdivisions=2, radius=s[0])
    elif p.kind == "convex":
        m = trimesh.Trimesh(np.asarray(p.vertices, float) / 1000.0).convex_hull
    else:
        raise ValueError(p.kind)
    T = np.eye(4)
    T[:3, :3] = rot_matrix(p.rot)
    T[:3, 3] = np.asarray(p.pos, float) / 1000.0
    m.apply_transform(T)
    return m


def kind_mesh(k: Kind) -> trimesh.Trimesh:
    """The whole object in its own frame (metres)."""
    return trimesh.util.concatenate([part_mesh(p) for p in k.parts])


# ---------------------------------------------------------------- the kinds

def cube(edge: float) -> list[Part]:
    return [Part("box", (edge, edge, edge))]


def screw_m6x30() -> list[Part]:
    """Lying along x: a 6 mm shank, 30 mm long, with an 11 mm head at +x."""
    return [Part("cylinder", (3, 30), (0, 0, 0), (0, 90, 0)), Part("cylinder", (5.5, 4), (17, 0, 0), (0, 90, 0))]


def jack() -> list[Part]:
    legs = [Part("capsule", (1.8, 20), rot=r) for r in [(0, 90, 0), (90, 0, 0), (0, 0, 0)]]
    tips = [Part("sphere", (3,), pos) for pos in [(12, 0, 0), (-12, 0, 0), (0, 0, 12), (0, 0, -12)]]
    return legs + tips


def ring(major: float = 9.0, minor: float = 1.5, n: int = 12) -> list[Part]:
    """A torus as a loop of short capsules."""
    seg = 2 * np.pi * major / n
    return [Part("capsule", (minor, seg), (major * np.cos(a), major * np.sin(a), 0), (90, 0, np.degrees(a)))
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def die(sides: int, size: float) -> list[Part]:
    if sides == 6:
        return cube(size)
    if sides == 4:
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    elif sides == 8:
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    elif sides == 20:
        v = np.asarray(trimesh.creation.icosahedron().vertices, float)
    else:
        raise ValueError(f"no d{sides}")
    v = v / np.abs(v).max() * size / 2
    return [Part("convex", vertices=v)]


def lego(studs_x: int, studs_y: int) -> list[Part]:
    w, l, h = 7.8 * studs_y, 7.8 * studs_x, 9.6
    parts = [Part("box", (l, w, h))]
    for i in range(studs_x):
        for j in range(studs_y):
            parts.append(Part("cylinder", (2.4, 1.8), ((i - (studs_x - 1) / 2) * 8, (j - (studs_y - 1) / 2) * 8, h / 2 + 0.9)))
    return parts


def soldier() -> list[Part]:
    return [Part("box", (16, 16, 2.5), (0, 0, 1.25)), Part("capsule", (3.5, 14), (0, 0, 12)),
            Part("sphere", (4,), (0, 0, 25)), Part("capsule", (1.4, 12), (5, 0, 14), (0, 30, 0)),
            Part("capsule", (1.0, 22), (-5, 0, 22), (0, -8, 0))]  # base, body, head, arm, rifle


def cup(r_in: float = 18, wall: float = 2, h: float = 40, n: int = 16, brim: float = 6) -> list[Part]:
    """Open cup as wall panels + floor + a brim ring (convex pieces only)."""
    parts = [Part("cylinder", (r_in + wall, 2), (0, 0, 1))]
    seg = 2 * np.pi * (r_in + wall / 2) / n * 1.08
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        c = r_in + wall / 2
        parts.append(Part("box", (wall, seg, h), (c * np.cos(a), c * np.sin(a), h / 2), (0, 0, np.degrees(a))))
        cb = r_in + wall + brim / 2
        segb = 2 * np.pi * cb / n * 1.08
        parts.append(Part("box", (brim, segb, 2), (cb * np.cos(a), cb * np.sin(a), h - 1), (0, 0, np.degrees(a))))
    return parts


def bin_box(w: float = 50, h: float = 55, wall: float = 2) -> list[Part]:
    return [Part("box", (w, w, wall), (0, 0, wall / 2)),
            Part("box", (wall, w, h), (w / 2 - wall / 2, 0, h / 2)), Part("box", (wall, w, h), (-w / 2 + wall / 2, 0, h / 2)),
            Part("box", (w - 2 * wall, wall, h), (0, w / 2 - wall / 2, h / 2)), Part("box", (w - 2 * wall, wall, h), (0, -w / 2 + wall / 2, h / 2))]


KINDS: dict[str, Kind] = {k.name: k for k in [
    Kind("cube_10", cube(10), "red", notes="the cell's small cube; fits the 12 mm gripper", tags=["cube", "grippable"]),
    Kind("cube_16", cube(16), "blue", notes="too wide to grip face-to-face: push it", tags=["cube"]),
    Kind("screw_m6x30", screw_m6x30(), "steel", "steel", tags=["fastener", "grippable"]),
    Kind("jack", jack(), "gold", "steel", notes="thin legs: grippable", tags=["toy", "grippable"]),
    Kind("ring", ring(), "gold", "brass", notes="thin band: grip the band", tags=["jewelry", "grippable"]),
    Kind("d4", die(4, 16), "purple", tags=["die"]),
    Kind("d6", die(6, 12), "white", notes="a small 12 mm die", tags=["die"]),
    Kind("d8", die(8, 16), "green", tags=["die"]),
    Kind("d20", die(20, 20), "orange", tags=["die"]),
    Kind("lego_1x2", lego(2, 1), "red", notes="7.8 mm wide: grippable across", tags=["lego", "grippable"]),
    Kind("lego_2x4", lego(4, 2), "yellow", notes="15.6 mm wide: push it", tags=["lego"]),
    Kind("soldier", soldier(), "olive", notes="grip the rifle or an arm", tags=["toy", "grippable"]),
    Kind("dowel", [Part("cylinder", (4, 30), rot=(0, 90, 0))], "wood", "wood", notes="rolls", tags=["rolls", "grippable"]),
    Kind("marker", [Part("cylinder", (7, 120), rot=(0, 90, 0))], "black", notes="14 mm: rolls, push it", tags=["rolls"]),
    Kind("tongue_depressor", [Part("box", (150, 18, 2))], "wood", "wood", notes="flat: push to an edge", tags=["flat"]),
    Kind("cup", cup(), "white", "ceramic", notes="container; grip the brim", tags=["container"]),
    Kind("bin", bin_box(), "yellow", notes="the cell's 50x50x55 mm bin", tags=["container"]),
    Kind("box_60", bin_box(60, 45, 10), "white", notes="a thick 3D-printed box: 10 mm walls, 40 mm well (sort into it)",
         tags=["container"]),
    Kind("cup_51", bin_box(50.8, 50.8, 10), "blue", notes="the lab's coloured cups (measured Oct 26): a 2 in cube, "
         "10 mm walls, a ~31 mm well", tags=["container"]),
]}
