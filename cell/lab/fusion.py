"""Depth fusion: many depth images (as base-frame points) -> one picture of the table.

height_map: many views of a still scene -> one top-down picture.
carve_merge: after something moved, fold a fresh local look into the existing map: wherever the new look saw the table
(or anything), the old points there are dropped, so moved objects don't leave ghosts; everywhere else the map is kept.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z

# the part of the table we map: comfortably covers the reachable area in front of the robot
X_RANGE, Y_RANGE, CELL = (0.15, 0.65), (-0.45, 0.45), 0.001


def height_map(points: np.ndarray, x_range=X_RANGE, y_range=Y_RANGE, cell=CELL) -> np.ndarray:
    """Top-down height above the table (mm) on a 1 mm grid: the highest point per cell, NaN where nothing was seen.
    Rows run along +x (away from the robot), columns along +y (to the robot's left)."""
    nx, ny = int(np.ceil((x_range[1] - x_range[0]) / cell)), int(np.ceil((y_range[1] - y_range[0]) / cell))
    ix = ((points[:, 0] - x_range[0]) / cell).astype(int)
    iy = ((points[:, 1] - y_range[0]) / cell).astype(int)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    hm = np.full(nx * ny, -np.inf)
    np.maximum.at(hm, ix[keep] * ny + iy[keep], (points[keep, 2] - SURFACE_Z) * 1000)
    hm[np.isinf(hm)] = np.nan
    return hm.reshape(nx, ny)


def _cells(points: np.ndarray, x_range=X_RANGE, y_range=Y_RANGE, cell=CELL) -> tuple[np.ndarray, tuple[int, int]]:
    nx, ny = int(np.ceil((x_range[1] - x_range[0]) / cell)), int(np.ceil((y_range[1] - y_range[0]) / cell))
    ix = np.clip(((points[:, 0] - x_range[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((points[:, 1] - y_range[0]) / cell).astype(int), 0, ny - 1)
    return ix * ny + iy, (nx, ny)


def carve_merge(old: np.ndarray, new: np.ndarray, grow: int = 2) -> np.ndarray:
    """Old points minus everything in the cells the new look covered (grown by `grow` mm to close gaps), plus the new points."""
    from scipy import ndimage
    if len(old) == 0:
        return new
    idx, (nx, ny) = _cells(new)
    seen = np.zeros(nx * ny, bool)
    seen[idx] = True
    seen = ndimage.binary_dilation(seen.reshape(nx, ny), iterations=grow).ravel()
    keep = ~seen[_cells(old)[0]]
    return np.concatenate([old[keep], new])
