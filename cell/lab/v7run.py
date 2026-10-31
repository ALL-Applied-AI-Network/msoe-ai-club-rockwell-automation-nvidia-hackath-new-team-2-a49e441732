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
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose, load, project
from calibration.kinematics import tcp_uf0
from lab import armcheck
from lab import moves as MV
from lab import sam, vlm, worldview
from lab.carve import View, carve, name_objects, objects, on_table
from lab.simcell import SimCell, spread_layout
from lab.world import World
from lab.recorder import RUNS

# SAM finds, it does not name: generic thingness prompts catch everything, the LLM names blobs after.
# Two words because SAM's "object" concept is salience-hungry: 19 masks on the cluttered real table,
# ZERO on a sparse scene of three things, where "small object" finds all of them. Their masks union.
WORDS = ["object", "small object"]

# The deciding brain. Measured on real crops (croptest, Oct 31): Nemotron called 3 of 9 solid cubes
# flat stickers/tape (the dangerous direction) and needs ~90 s to think; gpt-6.1-terra got 10/10 with
# tape correctly flat, in 1.4-2.7 s at low effort. Default to it whenever the OpenRouter key is around;
# V7_VLM in the environment overrides either way (any VLM_<NAME> name or a full "openrouter/..." id).
BRAIN = os.getenv("V7_VLM") or ("openrouter/openai/gpt-6.1-terra" if os.getenv("OPENROUTER_API_KEY") else None)

DECIDE = """You are the mind of a robot arm. SAM found these things (your world model: mm, base frame,
x away from the base, y left; the photo is the table camera with each blob's id painted at its spot -
read the ids off the photo, never guess which blob is which):
{mirror}

Instruction: {instruction}

Done so far: {done}

Answer ONE action as JSON, nothing else:
  {{"say": "<one short line>", "do": {{"name": {{"thing": <id>, "name": "<what it clearly is, e.g. red cube 1>"}}}}}}
  {{"say": "<one short line>", "do": {{"look": {{"at": [x_mm, y_mm]}}}}}}   fresh close outlines there: finds what was missed, tightens a one-glance blob
  {{"say": "<one short line>", "do": {{"inspect": {{"thing": <id>}}}}}}   fly over it and study a close-up crop: tells you what it really is (use when the table photo is too far away to tell)
  {{"say": "<one short line>", "do": {{"move": {{"thing": <id>, "to": [x_mm, y_mm]}}}}}}   pick it up, set it down there
  {{"say": "<one short line>", "do": {{"into": {{"thing": <id>, "container": <id>}}}}}}   drop it into an open-top thing (a bin, a cup): fly above it, let go, gravity does the rest
  {{"say": "<one short line>", "do": {{"forget": {{"thing": <id>}}}}}}   the photo shows this blob is not a real object (tape, a shadow, a mark on the table, a double of another blob)
  {{"say": "<one short line>", "done": true}}   the instruction is satisfied, or nothing useful is left

You are curious and decisive: figure out the scene, then do the task. The ids painted on the photo
are the model's ids. move and into accept only things you have named (the code refuses an unnamed
blob). The table is x 260-500, y -280-280. Every answer carries a "do" or "done"."""

INSPECT = ('This is a close-up crop of a robot workcell table. The thing in question is at the CENTER '
           'of the crop; anything near the edges is a neighbor, not it. If there is nothing at the '
           'center, say so. What is the center thing? Is it a flat marking on the table surface (tape, '
           'shadow, stain, line), a solid object, or nothing? Answer JSON only: '
           '{"what": "<2-4 words, or nothing>", "color": "<word>", "flat": true or false}')


def marked(img: np.ndarray, w: World, cam: dict) -> np.ndarray:
    """Set-of-marks: the table photo with every solid thing's id painted at its projected spot, so
    photo -> id is read off the picture, not guessed (the grounding gap: the agent once moved junk
    blob #5 believing it was the red cube)."""
    out = img.copy()
    ih, iw = out.shape[:2]
    cw, ch = cam["size"]
    for t in w.things:
        if t.state != "solid":
            continue
        u, v = project(cam, [[t.x_mm / 1000, t.y_mm / 1000, SURFACE_Z + t.top_mm / 1000]])[0]
        u, v = int(u * iw / cw), int(v * ih / ch)
        if not (0 <= u < iw and 0 <= v < ih):
            continue
        cv2.circle(out, (u, v), 3, (255, 255, 255), -1)
        cv2.circle(out, (u, v), 4, (0, 0, 0), 1)
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(out, f"#{t.id}", (u + 6, v - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th)
    return out


def decide(w: World, instruction: str, done: list[str], table_img: np.ndarray) -> dict:
    p = DECIDE.format(mirror=w.mirror(), instruction=instruction, done="; ".join(done) or "nothing yet")
    worldview.trace("think", "deciding: " + p.replace("\n", " ⏎ ")[:500])
    r = vlm.ask_image(table_img, p, think=True, model=BRAIN) or {}
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


def show_cam(name: str, img: np.ndarray, found: list[dict]) -> None:
    """What this camera sent to SAM and what SAM answered: the frame with every mask outlined and
    the word that found it - the eyes, inspectable, next to the world they built."""
    vis = img.copy()
    for d in found:
        cs, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (60, 255, 80), 2)
        bx = d.get("box")
        if bx is not None:
            cv2.putText(vis, d.get("prompt", ""), (int(bx[0]), max(14, int(bx[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 80), 1)
    rr.log(f"cams/{name}", rr.Image(vis).compress(jpeg_quality=70))


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
        show_cam(cam_name, img, found)
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


def grasp(w: World, t, cur, place, travel_mm: float, drop_mm: float | None) -> tuple[list, str]:
    """Slide-in first (fingers down beside it, slide until it sits between them - the grip that works
    on 10 mm cubes and screws on the real table), top pick only when no side is clear."""
    jaw = float(np.degrees(t.yaw) + 90.0)
    at = (t.x_mm / 1000.0, t.y_mm / 1000.0)

    def clear(xy) -> bool:
        return all(th.dist_mm(xy[0] * 1000, xy[1] * 1000) > max(th.size_mm) / 2 + 20
                   for th in w.things if th.id != t.id and th.of != t.id)

    rows, why = MV.push_grasp(cur, at, jaw, max(t.size_mm), place, travel_mm, drop_mm, clear)
    if rows:
        return rows, "slide-in grasp"
    return MV.pick_top(cur, at, jaw, place, travel_mm, drop_mm), f"top pick ({why})"


def row_points(rows) -> np.ndarray:
    a = np.atleast_2d(np.asarray(rows, float))
    return np.stack([a[:, 0] / 1000, a[:, 1] / 1000, (a[:, 2] - UF0_Z_MM) / 1000], -1)


def run(cell, instruction: str, headless: bool = False, tag: str = "v7", attach: bool = False) -> None:
    """The V7 loop on any cell that speaks observe/play/tip: the MuJoCo sim, the Isaac twin through
    the cell API (lab/realcell.py), and one day the real arm - same loop, same trace.
    attach=True (the app): log into the recording that already feeds the page's embedded viewer,
    replacing its layout with the V7 one, instead of opening a window of our own."""
    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    out.mkdir(parents=True)
    if attach:
        worldview.attach()
    else:
        rr.init("v7", spawn=not headless)
        rr.save(str(out / "v7.rrd"))
        worldview.blueprint()
        worldview.table()

    cams = load()
    worldview.trace("think", f"brain: {vlm.model_id(BRAIN) or 'default'}")
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
    for cycle in range(20):
        r = decide(w, instruction, done, marked(cell.observe()["images"]["table"], w, cams["table"]))
        if r.get("say"):
            worldview.trace("say", r["say"])
            print("robot:", r["say"], flush=True)
        do = r.get("do") or {}
        if r.get("done"):
            break
        if not do:  # it narrated instead of acting: remind it, don't end the run
            done.append("you said something but did nothing: every answer carries a do or done")
            continue
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
        if "forget" in do:  # the ventral stream dismissing a false blob: tape, shadow, a double
            try:
                w.forget(int(do["forget"]["thing"]))
                done.append(f"forgot blob #{do['forget']['thing']}")
            except (KeyError, TypeError, ValueError) as e:
                worldview.trace("plan", f"bad forget {do['forget']}: {e}", "WARN")
            worldview.things(w)
            worldview.mind(w)
            for e2 in w.events[said:]:
                worldview.trace("world", e2)
            said = len(w.events)
            continue
        if "inspect" in do:  # Joel: crop what SAM found, send it up close, ask "what's this?" -
            try:             # the ventral stream foveating before it names
                t = w[int(do["inspect"]["thing"])]
            except (KeyError, TypeError, ValueError) as e:
                worldview.trace("plan", f"bad inspect {do['inspect']}: {e}", "WARN")
                done.append(f"your inspect call {do['inspect']} made no sense")
                continue
            fly(cell, np.array([t.x_mm, t.y_mm,
                                SURFACE_Z * 1000 + UF0_Z_MM + max(160.0, t.top_mm + 120.0), *DOWN], float))
            obs = cell.observe()
            img, jnts = obs["images"]["wrist"], np.asarray(obs["joints"], float)
            Tf = np.linalg.inv(flange_pose(jnts))   # wrist cam: project() wants flange-frame points
            c3 = np.array([t.x_mm / 1000, t.y_mm / 1000, SURFACE_Z + t.top_mm / 2000])
            r3 = c3 + [max(t.size_mm) / 2000 + 0.012, 0, 0]
            uv = project(cams["wrist"], [Tf[:3, :3] @ c3 + Tf[:3, 3], Tf[:3, :3] @ r3 + Tf[:3, 3]])
            ih, iw = img.shape[:2]
            cw = cams["wrist"]["size"][0]
            u, v = uv[0][0] * iw / cw, uv[0][1] * iw / cw
            rad = int(max(float(np.hypot(*(uv[1] - uv[0]))) * iw / cw * 1.8, 60))
            crop = img[max(0, int(v) - rad):int(v) + rad, max(0, int(u) - rad):int(u) + rad]
            if crop.size and min(crop.shape[:2]) < 300:   # blow tiny crops up so the model sees pixels
                s = 300 / min(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            if crop.size:
                rr.log("cams/closeup", rr.Image(crop).compress(jpeg_quality=80))
            a = (vlm.ask_image(crop, INSPECT, model=BRAIN, think=False) or {}) if crop.size else {}
            j = a.get("json") or {}
            what, color, flat = j.get("what", "?"), j.get("color", "?"), j.get("flat")
            worldview.trace("check", f"inspect #{t.id}: {what}, {color}, flat={flat}")
            print(f"inspect #{t.id}: {what}, {color}, flat={flat}", flush=True)
            if j and str(what).lower().startswith("nothing"):
                # the close look AT its spot saw nothing there: that is the contradiction that deletes
                # (friend-walks-away) - evidence logic lives in code, not in the prompt
                w.forget(t.id)
                done.append(f"inspected #{t.id}: nothing there - forgotten")
            else:
                t.note = f"close-up: {what}, {color}" + (", flat on the table" if flat else "")
                done.append(f"inspected #{t.id}: {what} ({color}), flat={flat}")
            worldview.things(w)
            worldview.mind(w)
            for e2 in w.events[said:]:
                worldview.trace("world", e2)
            said = len(w.events)
            continue
        if "into" in do:  # Joel: no cavity geometry - the LLM knows it's open on top, gravity finds the
            try:          # inside, and a glance validates
                t, c = w[int(do["into"]["thing"])], w[int(do["into"]["container"])]
            except (KeyError, TypeError, ValueError) as e:
                worldview.trace("plan", f"bad into {do['into']}: {e}", "WARN")
                print(f"bad into {do['into']}: {e}", flush=True)
                done.append(f"your into call {do['into']} made no sense")
                continue
            unnamed = [x for x in (t, c) if x.name == "blob"]
            if unnamed:  # the photo<->mirror join: acting on a blob it never named is how #5 got moved
                msg = "still unnamed: " + ", ".join(f"#{x.id}" for x in unnamed) + " - inspect or name first"
                worldview.trace("plan", "into refused, " + msg, "WARN")
                done.append("into refused, " + msg)
                continue
            obs = cell.observe()
            cur, q = tcp_uf0(obs["joints"]), np.asarray(obs["joints"], float)
            place = (c.x_mm / 1000.0, c.y_mm / 1000.0)
            rows, how = grasp(w, t, cur, place, max(120.0, c.top_mm + 60.0), drop_mm=c.top_mm + 12.0)
            worldview.trace("plan", f"grip: {how}")
            rep = armcheck.check(rows, q, obstacles=w.boxes(ignore=(t.id,)))
            w.plan_move(t.id, c.x_mm, c.y_mm)
            rr.set_time("step", sequence=step); step += 1
            rr.log("world/plan/path", rr.LineStrips3D([row_points(rows)], colors=[(60, 200, 90)], radii=0.0012))
            worldview.things(w)
            worldview.trace("plan", f"{t.name} #{t.id} -> into {c.name} #{c.id}: "
                            + ("ok" if rep.ok else rep.problem), "INFO" if rep.ok else "WARN")
            print(f"into: {t.name} #{t.id} -> {c.name} #{c.id}: {'ok' if rep.ok else rep.problem}", flush=True)
            if not rep.ok:
                w.confirm(t.id, found=None)
                done.append(f"{t.name} #{t.id} into {c.name} #{c.id} refused: {rep.problem}")
            else:
                worldview.trace("do", f"{len(rows)} rows: drop {t.name} #{t.id} into {c.name} #{c.id}")
                cell.play(rows)
                cur2 = tcp_uf0(cell.observe()["joints"])
                cell.play(MV.travel(cur2, place, 0, 0, 300.0) + MV.hold(MV.uf0(*place, 300.0), 0, 0, 10))
                shot = cell.observe()["images"]["wrist"]
                rr.log("cams/closeup", rr.Image(shot).compress(jpeg_quality=80))
                g = vlm.ask_image(shot, f"Looking straight down at the {c.name}. Is the {t.name} now inside "
                                  'it? Answer JSON: {"inside": true or false}', model=BRAIN, think=False) or {}
                inside = bool((g.get("json") or {}).get("inside"))
                worldview.trace("check", f"glance: {t.name} inside {c.name}? {inside}",
                                "INFO" if inside else "WARN")
                print(f"glance: {t.name} inside {c.name}? {inside}", flush=True)
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
        if t.name == "blob":
            worldview.trace("plan", f"move refused, #{t.id} still unnamed", "WARN")
            done.append(f"move refused, #{t.id} still unnamed - inspect or name first")
            continue
        place = (to[0] / 1000.0, to[1] / 1000.0)
        obs = cell.observe()
        cur, q = tcp_uf0(obs["joints"]), np.asarray(obs["joints"], float)
        rows, how = grasp(w, t, cur, place, 120.0, None)
        worldview.trace("plan", f"grip: {how}")

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
