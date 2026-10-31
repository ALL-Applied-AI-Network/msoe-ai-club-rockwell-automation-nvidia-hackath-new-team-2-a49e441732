"""V7 alive in the sim cell: the agent loop, with everything it thinks visible in a live Rerun viewer.

    uv run python -m lab.v7run ["instruction"] [--headless]

The 3D view holds the robot's head: carved voxels, blue solids (evidence), red wireframe ghosts
(predictions), the green fingertip path before every move. The text log is the agent's stack trace,
one collapsible entity per layer of its mind:
    agent/see     every SAM call: word, masks found
    agent/think   every LLM prompt and raw reply
    agent/world   every state event: new / predict / confirm / miss
    agent/plan    every motion verdict (self-collision, world collision, wrist flips)
    agent/do      what was actually sent to the arm
    agent/check   what the confirming look found
    agent/mind    the mirror the LLM reasons over, updating in place
The loop is the `while True:`: look once, then decide -> predict -> act -> confirm until the LLM says
done. The sim cameras render through the real calibration, so this tests the machinery; the accuracy
numbers that count are the offline ones on real recorded stills (lab.carve_check).
"""
from __future__ import annotations

import json
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
    worldview.trace("think", "naming the table: " + NAME_WORDS.split("\n")[0])
    r = vlm.ask_image(table_img, NAME_WORDS) or {}
    worldview.trace("think", "reply: " + (r.get("text") or "(nothing)")[:300])
    d = r.get("json") or {}
    ws = [w.strip() for w in d.get("words", []) if isinstance(w, str) and w.strip()][:8]
    return ws or ["object"]


DECIDE = """You are the planner of a robot arm. Its world model (mm, base frame, x away from the base,
y left; the photo is the table camera, for context only - trust the model's numbers):
{mirror}

Instruction: {instruction}

Done so far: {done}

Answer ONE next change of the world as JSON:
  {{"say": "<one short line of what you're doing and why>", "move": {{"thing": <id>, "to": [x_mm, y_mm]}}}}
or, when the instruction is satisfied (or nothing useful is left):
  {{"say": "<one line>", "done": true}}
Rules: only move solid things. Destinations on clear table: at least 40 mm from everything else,
x between 260 and 500, y between -280 and 280. A thing that failed to confirm is where the model says."""


def decide(w: World, instruction: str, done: list[str], table_img: np.ndarray) -> dict:
    p = DECIDE.format(mirror=w.mirror(), instruction=instruction, done="; ".join(done) or "nothing yet")
    worldview.trace("think", "deciding: " + p.replace("\n", " ⏎ ")[:500])
    r = vlm.ask_image(table_img, p, think=True) or {}
    if r.get("thinking"):
        worldview.trace("think", "thinking: " + r["thinking"][:400])
    worldview.trace("think", "reply: " + (r.get("text") or "(nothing)")[:400])
    return r.get("json") or {}
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
        worldview.trace("see", f"{cam_name}: {len(found)} masks ("
                        + ", ".join(sorted({f['prompt'] for f in found})) + ")")
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


def main(headless: bool = False,
         instruction: str = "line the cubes up in a row near the base, and leave the screw alone") -> None:
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
        worldview.mind(w)
        print(f"stop {sid}: {len(obs_list)} objects in the world", flush=True)

    print("\n" + w.mirror() + "\n", flush=True)

    # back to a plain down-looking pose: a pick must not start from a tilted look (the wrist-flip
    # guard refuses the 44-degree snap, rightly)
    fly(cell, np.array([430.0, 0.0, SURFACE_Z * 1000 + UF0_Z_MM + 160, *DOWN], float))

    # ---- the loop: the LLM decides, the world predicts, the arm acts, the look confirms ----
    said = len(w.events)
    done: list[str] = []
    for cycle in range(10):
        r = decide(w, instruction, done, cell.observe()["images"]["table"])
        if r.get("say"):
            worldview.trace("say", r["say"])
            print("robot:", r["say"], flush=True)
        m = r.get("move")
        if r.get("done") or not isinstance(m, dict):
            break
        try:
            t = w[int(m["thing"])]
            to = (float(m["to"][0]), float(m["to"][1]))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            worldview.trace("plan", f"bad move {m}: {e}", "WARN")
            done.append(f"your move {m} made no sense")
            continue
        place = (to[0] / 1000.0, to[1] / 1000.0)
        obs = cell.observe()
        cur, q = tcp_uf0(obs["joints"]), np.asarray(obs["joints"], float)
        jaw = float(np.degrees(t.yaw) + 90.0)
        rows = MV.pick_top(cur, (t.x_mm / 1000, t.y_mm / 1000), jaw, place, 120.0, None)

        # predict BEFORE moving: ghost at the destination, green path, both collision checks
        rep = armcheck.check(rows, q, obstacles=w.boxes(ignore=(t.id,)))
        w.plan_move(t.id, *to)
        rr.set_time("step", sequence=step); step += 1
        rr.log("world/plan/path", rr.LineStrips3D([row_points(rows)], colors=[(60, 200, 90)], radii=0.0012))
        worldview.things(w)
        worldview.mind(w)
        worldview.trace("plan", f"{t.name} #{t.id} -> ({to[0]:.0f}, {to[1]:.0f}): "
                        + ("ok" if rep.ok else rep.problem), "INFO" if rep.ok else "WARN")
        if not rep.ok:
            w.confirm(t.id, found=None)
            done.append(f"{t.name} #{t.id} -> ({to[0]:.0f}, {to[1]:.0f}) refused: {rep.problem}")
        else:
            worldview.trace("do", f"{len(rows)} rows: pick {t.name} #{t.id}, carry, set down, confirm")
            cell.play(rows)
            found = confirm_look(cell, cams, place, t.name if t.name != "?" else words[0])
            ok = w.confirm(t.id, found)
            rr.set_time("step", sequence=step); step += 1
            rr.log("world/plan/path", rr.Clear(recursive=True))
            worldview.things(w)
            worldview.mind(w)
            worldview.trace("check", f"at ({found[0]:.0f}, {found[1]:.0f})" if found else "NOT where predicted",
                            "INFO" if found else "WARN")
            done.append(f"{t.name} #{t.id} -> ({to[0]:.0f}, {to[1]:.0f}): "
                        + ("confirmed" if ok else "NOT there, still at its old spot"))
        for e in w.events[said:]:
            worldview.trace("world", e)
        said = len(w.events)

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
    words_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if words_args:
        main(headless="--headless" in sys.argv, instruction=words_args[0])
    else:
        main(headless="--headless" in sys.argv)
