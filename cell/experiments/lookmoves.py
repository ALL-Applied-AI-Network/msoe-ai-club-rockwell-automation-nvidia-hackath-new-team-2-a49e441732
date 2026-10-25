"""How often do the mapper's look moves make the arm whip round or hit itself? Runs the mapper in the MuJoCo cell and
puts every move it sends through lab.armcheck (wrist flips, the table, and the arm against itself).

    uv run python -m experiments.lookmoves [trial] [seed]
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from lab import armcheck
from lab import primitives as P
from lab.minds import MINDS
from lab.trial import build, load_trial


def main(trial: str = "sort_mixed", seed: int = 0) -> None:
    cell = build(load_trial(trial), seed)
    cell.noise = True
    m = MINDS["map"]()
    m.reset()
    obs = cell.observe()
    kinds, moves = Counter(), 0
    for _ in range(60):
        chunk = m.act(obs)
        if getattr(m, "done", False):
            break
        q0 = np.asarray(obs["joints"][:6], float)
        if isinstance(chunk, P.JointRows):
            rep = armcheck.check_joints(np.asarray(chunk)[:, :6], 0.0)
        else:
            rep = armcheck.check(chunk, q0)
        moves += 1
        if not rep.ok:
            kinds[rep.problem.split(": ", 1)[-1].split(" (")[0]] += 1
            print(f"  move {moves}: {rep.problem}")
        P.send(cell, chunk)
        obs = cell.observe()
    bad = sum(kinds.values())
    print(f"{trial} seed {seed}: {moves} moves, {bad} with a problem")
    for k, n in kinds.most_common():
        print(f"  {n:3d}  {k}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["sort_mixed"]), *(int(a) for a in sys.argv[2:3]))
