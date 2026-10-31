"""V7 alive in the sim cell: watch the robot build its 3D world from SAM alone, predict, then act.

    uv run python -m lab.v7run              # spawns a live Rerun viewer

The "step" timeline tells the story:
    looks     each posed still carves the world tighter (voxels + blue solid boxes)
    predict   BEFORE the arm moves: the planned fingertip path (green) and the ghost (red wireframe)
    act       the sim arm does it
    confirm   the look where it should be: the ghost hardens into the thing, or dies
It ends by printing the robot's world against the sim's ground truth, thing by thing. The sim cameras
render through the real calibration, so this tests the machinery; the accuracy numbers that count are
the offline ones on real recorded stills (lab.carve_check).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z, UF0_Z_MM, load
from calibration.kinematics import tcp_uf0
from lab import armcheck
from lab import moves as MV
from lab import sam, vlm, worldview
from lab.carve import View, carve, name_objects, objects, on_table
from lab.simcell import SimCell, spread_layout
from lab.world import World
from lab.recorder import RUNS

NAME_WORDS = """List every kind of thing sitting on the table in this photo, as JSON:
{"words": ["cube", "screw"]}
One or two plain words per kind, the words a person would use to find one in a picture.
Leave out the robot, the gripper and the table itself."""


def llm_words(table_img: np.ndarray) -> list[str]:
    """Nothing hard-coded: the LLM looks at the table and names what SAM should find."""
    r = vlm.ask_image(table_img, NAME_WORDS)
    ws = [w.strip() for w in (r or {}).get("words", []) if isinstance(w, str) and w.strip()][:8]
    return ws or ["object"]
STOPS_FROM = Path("/Users/a12242/yaad/team-repo/cell/runs/20261030-131742_datarun_real/stops")
STOP_IDS = ["000", "001", "004", "007"]   # a high look, two scan lines, one real 40-degree tilt
DOWN = (-180.0, 0.0, 0.0)


def fly(cell: SimCell, target_row: np.ndarray) -> None:
    """Lift, go over, settle on the stop pose. The floating gripper follows with the cell's lag."""
    cur = np.array([*cell.tip() * 1000, 0, 0, 0], float)
    cur[2] += UF0_Z_MM
    up = cur.copy(); up[2] = max(cur[2], target_row[2], SURFACE_Z * 1000 + UF0_Z_MM + 250)
    rows = [list(up[:3]) + list(target_row[3:6]) + [0.0],
            list(target_row[:6]) + [0.0]]
    cell.play([r for r in rows for _ in range(12)])  # repeats: the lagged arm arrives and settles


def one_look(cell: SimCell, cams: dict, views: list[View], first: bool, words: list[str]) -> tuple:
    obs = cell.observe()
    jnts = np.asarray(obs["joints"], float)
    shots = [("wrist", jnts)] + ([("table", None), ("side", None)] if first else [])
    naming = None
    for cam_name, j in shots:
        img = obs["images"][cam_name]
        found = sam.segment_many(img, words, scale=2.0)
        found = [d for d in found if on_table(d["mask"], cams[cam_name], j)]
        union = np.zeros(img.shape[:2], bool)
        per_word: dict[str, np.ndarray] = {}
        for d in found:
            union |= d["mask"]
            per_word[d["prompt"]] = per_word.get(d["prompt"], np.zeros(img.shape[:2], bool)) | d["mask"]
        views.append(View(mask=union, cam=cams[cam_name], joints=j))
        if cam_name == "wrist" and per_word:
            naming = (per_word, views[-1])
        print(f"  {cam_name}: {len(found)} masks", flush=True)
    return naming


def confirm_look(cell: SimCell, cams: dict, place_m, word: str) -> tuple | None:
    """One top-down still over the destination: is it there? -> (x_mm, y_mm) or None."""
    from calibration.geometry import pixel_to_base
    cur = tcp_uf0(cell.observe()["joints"])
    cell.play([r for r in (MV.travel(cur, place_m, 0, 0, 300.0) + MV.hold(MV.uf0(*place_m, 300.0), 0, 0, 10))])
    obs = cell.observe()
    img, jnts = obs["images"]["wrist"], np.asarray(obs["joints"], float)
    for d in sorted(sam.segment_many(img, [word], scale=2.0), key=lambda d: -d["area"]):
        vs, us = np.nonzero(d["mask"])
        w, h = cams["wrist"]["size"]; mh, mw = d["mask"].shape
        p = pixel_to_base(cams["wrist"], (us.mean() * w / mw, vs.mean() * h / mh), joints=jnts)
        if np.hypot(p[0] - place_m[0], p[1] - place_m[1]) < 0.035:
            return (p[0] * 1000, p[1] * 1000)
    return None


def row_points(rows) -> np.ndarray:
    a = np.atleast_2d(np.asarray(rows, float))
    return np.stack([a[:, 0] / 1000, a[:, 1] / 1000, (a[:, 2] - UF0_Z_MM) / 1000], -1)


def main(headless: bool = False) -> None:
    run = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_v7sim"
    run.mkdir(parents=True)
    rr.init("v7", spawn=not headless)
    rr.save(str(run / "v7.rrd"))
    rr.log("world/table", rr.Boxes3D(centers=[[0.49, 0.0, SURFACE_Z - 0.002]],
                                     half_sizes=[[0.29, 0.36, 0.002]],
                                     colors=[(235, 235, 235)], fill_mode="solid"), static=True)

    rng = np.random.default_rng(3)
    cell = SimCell(spread_layout([("cube_10", "red"), ("cube_10", "blue"), ("cube_10", "yellow"),
                                  ("screw_m6x30", "gray")], rng), "v7 demo", seed=3)
    cell.settle()
    cams = load()
    words = llm_words(cell.observe()["images"]["table"])
    print(f"the LLM's words for SAM: {words}", flush=True)
    w = World()
    views: list[View] = []
    step = 0

    # ---- look: the world model builds, stop by stop ----
    naming = None
    for sid in STOP_IDS:
        d = np.load(STOPS_FROM / f"{sid}.npz", allow_pickle=True)
        target = np.asarray(tcp_uf0(d["joints"]), float)
        try:
            fly(cell, target)
        except ValueError as e:
            print(f"stop {sid} skipped: {e}", flush=True)
            continue
        n = one_look(cell, cams, views, first=(sid == STOP_IDS[0]), words=words)
        naming = n or naming
        solid, axes = carve(views)
        obs_list = objects(solid, axes)
        if naming:
            name_objects(obs_list, axes, *naming)
        w.look(obs_list)
        rr.set_time("step", sequence=step); step += 1
        worldview.voxels(solid, axes)
        worldview.things(w)
        print(f"stop {sid}: {len(obs_list)} objects in the world", flush=True)

    print("\n" + w.mirror() + "\n", flush=True)

    # back to a plain down-looking pose: a pick must not start from a tilted look (the wrist-flip
    # guard refuses the 44-degree snap, rightly)
    fly(cell, np.array([430.0, 0.0, SURFACE_Z * 1000 + UF0_Z_MM + 160, *DOWN], float))

    # ---- act: pick each cube and place it in a line near the base, predict first ----
    lane_y = -0.22
    for k, t in enumerate([t for t in sorted(w.things, key=lambda t: t.id)
                           if t.state == "solid" and max(t.size_mm) < 25][:2]):
        place = (0.33 + 0.06 * k, lane_y)
        obs = cell.observe()
        cur, q = tcp_uf0(obs["joints"]), np.asarray(obs["joints"], float)
        jaw = float(np.degrees(t.yaw) + 90.0)
        rows = MV.pick_top(cur, (t.x_mm / 1000, t.y_mm / 1000), jaw, place, 120.0, None)

        # predict: self-collision + the world's own things as obstacles + the ghost, all BEFORE moving
        rep = armcheck.check(rows, q, obstacles=w.boxes(ignore=(t.id,)))
        g = w.plan_move(t.id, place[0] * 1000, place[1] * 1000)
        rr.set_time("step", sequence=step); step += 1
        rr.log("world/plan/path", rr.LineStrips3D([row_points(rows)], colors=[(60, 200, 90)], radii=0.0012))
        worldview.things(w)
        print(f"predict: {t.name} #{t.id} -> ({place[0]*1000:.0f}, {place[1]*1000:.0f}); "
              f"motion check: {'ok' if rep.ok else rep.problem}", flush=True)
        if not rep.ok:
            w.confirm(t.id, found=None)
            continue

        cell.play(rows)

        found = confirm_look(cell, cams, place, t.name if t.name != "?" else words[0])
        ok = w.confirm(t.id, found)
        rr.set_time("step", sequence=step); step += 1
        rr.log("world/plan/path", rr.Clear(recursive=True))
        worldview.things(w)
        print(f"confirm: {'at (%.0f, %.0f)' % found if found else 'NOT where predicted'}", flush=True)

    # ---- the honest ending: the robot's world against the sim's truth ----
    truth = cell.body_positions()
    print("\nworld vs truth (mm):", flush=True)
    for t in w.things:
        if t.state != "solid":
            continue
        best = min(truth.values(), key=lambda p: np.hypot(p[0] * 1000 - t.x_mm, p[1] * 1000 - t.y_mm))
        err = np.hypot(best[0] * 1000 - t.x_mm, best[1] * 1000 - t.y_mm)
        print(f"  {t.name} #{t.id}: world ({t.x_mm:6.0f}, {t.y_mm:6.0f})  truth ({best[0]*1000:6.0f}, "
              f"{best[1]*1000:6.0f})  off {err:5.1f}", flush=True)
    print(f"\nsaved {run / 'v7.rrd'}", flush=True)


if __name__ == "__main__":
    import sys
    main(headless="--headless" in sys.argv)
