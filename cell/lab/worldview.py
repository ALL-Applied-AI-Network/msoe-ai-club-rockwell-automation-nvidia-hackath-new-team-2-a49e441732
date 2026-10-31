"""See the robot's world in 3D: the carved voxels (the raw evidence) and every thing it believes in.

Solids are filled boxes - a camera confirmed them there. Ghosts are red wireframes - predictions a move
made, not yet checked. Watching a carry here shows the split Joel asked for: the solid stays on the
table while the arm moves, the ghost waits at the destination, and only the confirming look snaps them
together.

    from lab import worldview
    worldview.start(run_dir)                 # rr.init + save worldview.rrd + the table
    worldview.voxels(solid, axes)            # one carve's surviving voxels
    worldview.things(w)                      # the World, boxes + labels (call again after every change)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z
from lab.carve import GRID_X, GRID_Y
from lab.world import World

SOLID = (90, 160, 255)
GHOST = (255, 80, 80)


def start(run: Path, name: str = "worldview") -> None:
    rr.init(name, spawn=False)
    rr.save(str(Path(run) / "worldview.rrd"))
    rr.log("world/table", rr.Boxes3D(
        centers=[[(GRID_X[0] + GRID_X[1]) / 2, (GRID_Y[0] + GRID_Y[1]) / 2, SURFACE_Z - 0.002]],
        half_sizes=[[(GRID_X[1] - GRID_X[0]) / 2, (GRID_Y[1] - GRID_Y[0]) / 2, 0.002]],
        colors=[(235, 235, 235)], fill_mode="solid"), static=True)


def voxels(solid: np.ndarray, axes: tuple) -> None:
    """The surviving voxels of one carve, coloured by height."""
    xs, ys, zs = axes
    ix, iy, iz = np.nonzero(solid)
    pts = np.stack([xs[ix], ys[iy], zs[iz]], -1)
    h = np.clip((pts[:, 2] - SURFACE_Z) / 0.06, 0, 1)
    col = np.stack([255 * h, 80 + 100 * (1 - h), 255 * (1 - h)], -1).astype(np.uint8)
    rr.log("world/voxels", rr.Points3D(pts, colors=col, radii=float(xs[1] - xs[0]) / 2))


def things(w: World) -> None:
    """Every thing the robot believes in, where it believes it: solids filled, ghosts as red wireframes."""
    for state, path, colour, fill in (("solid", "world/solids", SOLID, "solid"),
                                      ("ghost", "world/ghosts", GHOST, "densewireframe")):
        ts = [t for t in w.things if t.state == state]
        if not ts:
            rr.log(path, rr.Clear(recursive=True))
            continue
        rr.log(path, rr.Boxes3D(
            centers=[[t.x_mm / 1000.0, t.y_mm / 1000.0, SURFACE_Z + t.top_mm / 2000.0] for t in ts],
            half_sizes=[[t.size_mm[0] / 2000.0, t.size_mm[1] / 2000.0, t.top_mm / 2000.0] for t in ts],
            quaternions=[rr.Quaternion(xyzw=[0.0, 0.0, float(np.sin(t.yaw / 2)), float(np.cos(t.yaw / 2))])
                         for t in ts],
            labels=[f"{t.name} #{t.id}" + (" (ghost)" if t.state == "ghost" else "") for t in ts],
            colors=[colour] * len(ts), fill_mode=fill))
