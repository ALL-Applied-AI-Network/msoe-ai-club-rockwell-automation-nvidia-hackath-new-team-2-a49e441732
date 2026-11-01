"""Staging a trial's table for the simulator: shared by the app and the overnight suite."""
from __future__ import annotations

import copy

import numpy as np


def staged(trial_name: str, say=print):
    """Build a trial's scene with everything INSIDE the arm's envelope from birth: fixtures
    (bins) get their declared spots clamped in the spec, and randomly-spread objects re-roll
    their layout seed until all land inside. (Editing positions after the physics settles does
    nothing: Isaac imposes the SETTLED body poses, which is how a clamped bolt stayed at 492.)"""
    from lab import ladder as LD
    from lab.trial import build as build_scene
    t = copy.deepcopy(LD.load_trial(trial_name))
    moved = 0
    for f in t.get("fixtures", []):
        x, y = float(f["at"][0]), float(f["at"][1])
        r = float(np.hypot(x, y)) * 1000
        if r > 440:
            f["at"] = [x * 435.0 / r, y * 435.0 / r]
            moved += 1
    sim = None
    for seed in [t.get("seed", 0), *range(1, 24)]:
        sim = build_scene(t, seed)
        if all(float(np.hypot(i.pos[0], i.pos[1])) * 1000 <= 445 for i in sim.instances):
            break
    else:
        say("staging: no all-inside layout found in 24 rolls; using the last one")
    if moved:
        say(f"staging: pulled {moved} fixture(s) inside the reach ring")
    return sim
