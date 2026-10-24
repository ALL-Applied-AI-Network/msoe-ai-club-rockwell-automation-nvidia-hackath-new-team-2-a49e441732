"""Make simulated depth as bad as the real sensors' (the sims' perfect depth hid every perception problem until the
real cell, Oct 24).

    d = realistic_depth(depth_mm, rng)        # uint16 mm, 0 = no reading, as the cells serve it

What the real wrist and table depth did, roughly (to calibrate on the recorded real points: lab.evidence's SIG0/SIG2
and the numbers here should agree):
    - error along the ray that grows with distance (SIG0 + SIG2 * d^2)
    - flying pixels: at a depth edge, readings smeared between the near and the far surface
    - holes: no reading at many edge pixels, and some random ones
    - specks: a few readings far off (the phantom "things" were these)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SIG0, SIG2 = 0.0015, 0.010   # m: same model as lab.evidence assumes
EDGE_MM = 6.0                # a jump this big between neighbouring pixels is a depth edge
FLY = 0.6                    # share of edge pixels that read somewhere between the two surfaces
EDGE_HOLE = 0.25             # share of edge pixels with no reading
DROP = 0.02                  # random pixels with no reading
SPECK = 0.003                # random pixels read 5-40 mm off


def realistic_depth(depth_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d0 = depth_mm.astype(np.float32)
    valid = d0 > 0
    z = d0 / 1000.0
    d = d0 + rng.standard_normal(d0.shape).astype(np.float32) * (SIG0 + SIG2 * z * z) * 1000.0
    gy, gx = np.gradient(np.where(valid, d0, np.nan))
    edge = np.nan_to_num(np.hypot(gx, gy), nan=EDGE_MM + 1) > EDGE_MM
    edge = ndimage.binary_dilation(edge) & valid
    near = ndimage.minimum_filter(np.where(valid, d0, 1e6), 3)
    far = ndimage.maximum_filter(d0, 3)
    r = rng.random(d0.shape)
    fly = edge & (r < FLY)
    d[fly] = (near + (far - near) * rng.random(d0.shape))[fly]
    d[edge & (r > 1 - EDGE_HOLE)] = 0
    d[rng.random(d0.shape) < DROP] = 0
    sp = (rng.random(d0.shape) < SPECK) & valid
    d[sp] += (rng.choice([-1, 1], sp.sum()) * rng.uniform(5, 40, sp.sum())).astype(np.float32)
    d[~valid] = 0
    return np.clip(d, 0, 65535).astype(np.uint16)
