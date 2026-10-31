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

# SAM finds, it does not name: one generic prompt catches everything, the LLM names blobs after.
WORDS = ["object"]

DECIDE = """You are the mind of a robot arm. SAM found these things (your world model: mm, base frame,
x away from the base, y left; the photo is the table camera, use it to recognize what each blob is):
{mirror}

Instruction: {instruction}

Done so far: {done}

Answer ONE action as JSON, nothing else:
  {{"say": "<one short line>", "do": {{"name": {{"thing": <id>, "name": "<what it clearly is, e.g. red cube 1>"}}}}}}
  {{"say": "<one short line>", "do": {{"look": {{"at": [x_mm, y_mm]}}}}}}   fresh close outlines there: finds what was missed, tightens a one-glance blob
  {{"say": "<one short line>", "do": {{"move": {{"thing": <id>, "to": [x_mm, y_mm]}}}}}}   pick it up, set it down there
  {{"say": "<one short line>", "do": {{"into": {{"thing": <id>, "container": <id>}}}}}}   drop it into an open-top thing (a bin, a cup): fly above it, let go, gravity does the rest
  {{"say": "<one short line>", "done": true}}   the instruction is satisfied, or nothing useful is left

You are curious: name blobs you recognize in the photo before moving them; look closer when unsure.
Only move things that exist in the model. Destinations on clear table, 40 mm from everything else,
x 260-500, y -280-280. If the model is empty, say the eyes have failed and stop: do not invent things."""


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


def run(cell, instruction: str, headless: bool = False, tag: str = "v7") -> None:
    """The V7 loop on any cell that speaks observe/play/tip: the MuJoCo sim, the Isaac twin through
    the cell API (lab/realcell.py), and one day the real arm - same loop, same trace."""
    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    out.mkdir(parents=True)
    rr.init("v7", spawn=not headless)
    rr.save(str(out / "v7.rrd"))
    rr.log("world/table", rr.Boxes3D(centers=[[0.49, 0.0, SURFACE_Z - 0.002]],
                                     half_sizes=[[0.29, 0.36, 0.002]],
                                     colors=[(235, 235, 235)], fill_mode="solid"), static=True)

    cams = load()
    words = WORDS
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
        one_look(cell, cams, views, first=(sid == STOP_IDS[0]), words=words)
        solid, axes, support = carve(views)
        obs_list = objects(solid, axes, support=support)
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
        do = r.get("do") or {}
        if r.get("done") or not do:
            break
        if "name" in do:
            try:
                w.rename(int(do["name"]["thing"]), str(do["name"]["name"])[:40])
                done.append(f"named #{do['name']['thing']} {do['name']['name']!r}")
            except (KeyError, TypeError, ValueError) as e:
                worldview.trace("plan", f"bad name {do['name']}: {e}", "WARN")
                done.append(f"your name call {do['name']} made no sense")
            for e2 in w.events[said:]:
                worldview.trace("world", e2)
            said = len(w.events)
            worldview.mind(w)
            continue
        if "look" in do:
            at = do["look"].get("at") or [430, 0]
            fly(cell, np.array([float(at[0]), float(at[1]),
                                SURFACE_Z * 1000 + UF0_Z_MM + 250, *DOWN], float))
            fresh: list[View] = []
            one_look(cell, cams, fresh, first=True, words=words)
            solid, axes, support = carve(fresh)
            obs_list = objects(solid, axes, support=support)
            seen_ids = {t2.id for t2 in w.look(obs_list)}
            # the friend turned around: a solid AT the looked-at spot that this look didn't find is gone.
            # But a look that saw NOTHING ANYWHERE is a blind instrument, not evidence of absence
            # (SAM returns zero masks on a sparse close-up): it deletes nothing.
            if obs_list:
                for t2 in [t2 for t2 in w.things if t2.state == "solid" and t2.id not in seen_ids
                           and t2.dist_mm(float(at[0]), float(at[1])) < 60.0]:
                    w.forget(t2.id)
            rr.set_time("step", sequence=step); step += 1
            worldview.voxels(solid, axes)
            worldview.things(w)
            worldview.mind(w)
            for e2 in w.events[said:]:
                worldview.trace("world", e2)
            said = len(w.events)
            done.append(f"looked at ({at[0]:.0f}, {at[1]:.0f})")
            continue
        if "into" in do:  # Joel: no cavity geometry - the LLM knows it's open on top, gravity finds the
            try:          # inside, and a glance validates
                t, c = w[int(do["into"]["thing"])], w[int(do["into"]["container"])]
            except (KeyError, TypeError, ValueError) as e:
                worldview.trace("plan", f"bad into {do['into']}: {e}", "WARN")
                done.append(f"your into call {do['into']} made no sense")
                continue
            obs = cell.observe()
            cur, q = tcp_uf0(obs["joints"]), np.asarray(obs["joints"], float)
            jaw = float(np.degrees(t.yaw) + 90.0)
            place = (c.x_mm / 1000.0, c.y_mm / 1000.0)
            rows = MV.pick_top(cur, (t.x_mm / 1000, t.y_mm / 1000), jaw, place,
                               max(120.0, c.top_mm + 60.0), drop_mm=c.top_mm + 12.0)
            rep = armcheck.check(rows, q, obstacles=w.boxes(ignore=(t.id,)))
            w.plan_move(t.id, c.x_mm, c.y_mm)
            rr.set_time("step", sequence=step); step += 1
            rr.log("world/plan/path", rr.LineStrips3D([row_points(rows)], colors=[(60, 200, 90)], radii=0.0012))
            worldview.things(w)
            worldview.trace("plan", f"{t.name} #{t.id} -> into {c.name} #{c.id}: "
                            + ("ok" if rep.ok else rep.problem), "INFO" if rep.ok else "WARN")
            if not rep.ok:
                w.confirm(t.id, found=None)
                done.append(f"{t.name} #{t.id} into {c.name} #{c.id} refused: {rep.problem}")
            else:
                worldview.trace("do", f"{len(rows)} rows: drop {t.name} #{t.id} into {c.name} #{c.id}")
                cell.play(rows)
                cur2 = tcp_uf0(cell.observe()["joints"])
                cell.play(MV.travel(cur2, place, 0, 0, 300.0) + MV.hold(MV.uf0(*place, 300.0), 0, 0, 10))
                shot = cell.observe()["images"]["wrist"]
                g = vlm.ask_image(shot, f"Looking straight down at the {c.name}. Is the {t.name} now inside "
                                  'it? Answer JSON: {"inside": true or false}') or {}
                inside = bool((g.get("json") or {}).get("inside"))
                worldview.trace("check", f"glance: {t.name} inside {c.name}? {inside}",
                                "INFO" if inside else "WARN")
                ok = w.confirm(t.id, (c.x_mm, c.y_mm) if inside else None)
                if inside:
                    w[t.id].note = f"in {c.name} #{c.id}"
                rr.set_time("step", sequence=step); step += 1
                rr.log("world/plan/path", rr.Clear(recursive=True))
                worldview.things(w)
                worldview.mind(w)
                done.append(f"{t.name} #{t.id} into {c.name} #{c.id}: " + ("confirmed inside" if ok else "NOT seen inside"))
            for e2 in w.events[said:]:
                worldview.trace("world", e2)
            said = len(w.events)
            continue
        m = do.get("move")
        if not isinstance(m, dict):
            worldview.trace("plan", f"unknown action {do}", "WARN")
            done.append(f"your action {do} made no sense")
            continue
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

    # ---- the honest ending: the robot's world against the sim's truth (sim only: Isaac and the
    # real cell keep their truth to themselves) ----
    if not hasattr(cell, "body_positions"):
        print(f"\nsaved {out / 'v7.rrd'}", flush=True)
        return
    truth = cell.body_positions()
    print("\nworld vs truth (mm):", flush=True)
    for t in w.things:
        if t.state != "solid":
            continue
        best = min(truth.values(), key=lambda p: np.hypot(p[0] * 1000 - t.x_mm, p[1] * 1000 - t.y_mm))
        err = np.hypot(best[0] * 1000 - t.x_mm, best[1] * 1000 - t.y_mm)
        print(f"  {t.name} #{t.id}: world ({t.x_mm:6.0f}, {t.y_mm:6.0f})  truth ({best[0]*1000:6.0f}, "
              f"{best[1]*1000:6.0f})  off {err:5.1f}", flush=True)
    print(f"\nsaved {out / 'v7.rrd'}", flush=True)


def main(headless: bool = False,
         instruction: str = "line the cubes up in a row near the base, and leave the screw alone",
         things: str = "cube_10:red,cube_10:blue,cube_10:yellow,screw_m6x30:gray") -> None:
    rng = np.random.default_rng(3)
    order = [tuple(t.strip().split(":")) for t in things.split(",") if t.strip()]
    cell = SimCell(spread_layout(order, rng), "v7", seed=3)
    cell.settle()
    run(cell, instruction, headless, tag="v7sim")


if __name__ == "__main__":
    import sys
    kw: dict = {"headless": "--headless" in sys.argv}
    plain = [a for a in sys.argv[1:] if not a.startswith("--")]
    if plain:
        kw["instruction"] = plain[0]
    for a in sys.argv[1:]:
        if a.startswith("--things="):
            kw["things"] = a.split("=", 1)[1]
    main(**kw)
