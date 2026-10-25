"""Sort trials in the MuJoCo cell (noisy depth), scored against the truth: parts in the box of their colour.

    uv run python -m experiments.sort_eval [trial] [seeds...]
"""
from __future__ import annotations

import sys
import time

from lab import armcheck, primitives as P
from lab.minds import MINDS
from lab.trial import build, load_trial, objects_only, sort_check


def run(trial: str, seed: int, max_steps: int = 120) -> dict:
    cell = build(load_trial(trial), seed)
    cell.noise = True
    m = MINDS["sort"]()
    m.reset()
    obs = cell.observe()
    t0 = time.time()
    flips = 0
    for _ in range(max_steps):
        ch = m.act(obs)
        if m.done:
            break
        P.send(cell, ch)
        obs = cell.observe()
    truth = cell.object_meshes()
    sc = sort_check(truth, "color")
    right = sum(1 for b, names in sc["contents"].items() for n in names if n.split()[0] == b.split()[0])
    grouped = sc["in_boxes"] == len(objects_only(truth)) and not sc["mixed_boxes"] and not sc["split_groups"]
    return {"trial": trial, "seed": seed, "parts": len(objects_only(truth)), "right_box": right, "in_boxes": sc["in_boxes"],
            "on_table": sc["on_table"], "grouped": grouped, "mixed": sc["mixed_boxes"], "split": sc["split_groups"], "picks": m.picks, "robot_s": round(cell.t), "wall_s": round(time.time() - t0),
            "moves_failing_check": flips, "report": m.report}


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "sort_cubes"
    for seed in [int(a) for a in sys.argv[2:]] or [0, 1, 2]:
        r = run(trial, seed)
        print(f"{trial} s{seed}: {r['in_boxes']}/{r['parts']} in boxes, {r['right_box']} in a box of their own colour; "
              f"sorted by colour {'YES' if r['grouped'] else 'no'} (mixed {r['mixed']}, split {r['split']}); picks {r['picks']}; "
              f"{r['robot_s']} s robot, {r['wall_s']} s wall"
              + (f"\n    on the table: {r['on_table']}" if r["on_table"] else ""), flush=True)
