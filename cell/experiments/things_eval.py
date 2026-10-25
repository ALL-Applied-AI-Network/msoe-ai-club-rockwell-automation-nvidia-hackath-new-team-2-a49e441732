"""Map a scene in the MuJoCo cell (noisy depth), read the things off the evidence map, score them against the truth.

    uv run python -m experiments.things_eval [trial ...]        # default: sort_cubes six_cubes_pile sort_mixed, seeds 0-2
"""
from __future__ import annotations

import sys
import time

import numpy as np

from lab import primitives as P
from lab.minds import MINDS
from lab.things import Things, score
from lab.trial import build, load_trial


def run(trial: str, seed: int, verbose: bool = False) -> dict:
    cell = build(load_trial(trial), seed)
    cell.noise = True
    m = MINDS["map"]()
    m.reset()
    obs = cell.observe()
    table_rgb = obs["images"]["table"]
    t0 = time.time()
    for _ in range(40):
        ch = m.act(obs)
        if m.done:
            break
        P.send(cell, ch)
        obs = cell.observe()
    things = Things().update(m.map, table_rgb, m.cams)
    s = score(things, cell.object_meshes())
    s |= {"trial": trial, "seed": seed, "looks": m.looks, "robot_s": round(cell.t), "wall_s": round(time.time() - t0)}
    if verbose:
        for t in things:
            print("    ", t.describe())
    return s


def line(s: dict) -> str:
    pe, he = np.array(s["pos_err_mm"] or [np.nan]), np.array(s["height_err_mm"] or [np.nan])
    return (f"{s['trial']:15s} s{s['seed']}: parts found {s['found']}/{s['parts']}, colour right {s['colour_right']}/{s['found']}, "
            f"boxes {s['boxes_found']}/{s['boxes']}, merged {s['merged']}, phantoms {len(s['phantoms'])}; "
            f"position error median {np.nanmedian(pe):.1f} mm (max {np.nanmax(pe):.1f}), height error median {np.nanmedian(he):+.1f} mm"
            f"  [{s['looks']} looks, {s['robot_s']} s]"
            + (f"\n      missed: {s['missed']}" if s["missed"] else "") + (f"\n      phantoms: {s['phantoms']}" if s["phantoms"] else ""))


if __name__ == "__main__":
    trials = sys.argv[1:] or ["sort_cubes", "six_cubes_pile", "sort_mixed"]
    for tr in trials:
        for seed in (0, 1, 2):
            print(line(run(tr, seed)), flush=True)
