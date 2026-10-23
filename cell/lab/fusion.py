"""Depth fusion: many depth images (as base-frame points) -> one picture of the table.

Only the height map so far. TSDF fusion and per-object (moving) fusion come next.
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
