"""The task ladder in the sim: simple tasks first, harder ones after, each one end to end and scored by the sim's truth.

    uv run python -m lab.ladder                      # every lab/trials/ladder*.yaml, in order
    uv run python -m lab.ladder ladder03 ladder09    # some of them (a prefix of the name is enough)
    uv run python -m lab.ladder --list

Each task: reset the table; a table-camera picture (before the arm comes over); the fast scan (lab/depth_check --fast:
safe height, sweep, the evidence grid on the GPU box, rigid objects); the LLM gets the picture with each found
object's number drawn on it, the list of objects (size, height, colour, which are containers) and the instruction, and
answers with moves: object -> a container, or object -> a named pile (a pile of one is "somewhere else"). The arm does
them one after the other (lab/handling.put: every height from the measured shapes). Then the sim's truth says whether
the goal in the trial's YAML holds:
    into: any | {color: c}   in a box (of that colour): inside its walls, below its rim
    stays: true              moved less than STAY_MM
    moved: true              moved more than MOVED_MM, still on the table
    pile: true               all of them on the table, within PILE_MM of their middle
    piles_by: kind           each kind in its own pile, the piles apart
    bins_by: kind            every one in a box, one kind per box
Writes runs/<stamp>_ladder/<task>/ (before.jpg, after.jpg, plan.json, result.json) and runs/<stamp>_ladder/sheet.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import requests

from calibration.geometry import SURFACE_Z, load, project
from lab import handling as HD
from lab import objects as OB
from lab import primitives as P
from lab import vlm
from lab.depth_bench import Poly, sim_objects
from lab.depth_live import DepthLink, tunnel
from lab.pick_check import joints_for
from lab.recorder import RUNS
from lab.trial import TRIALS, build, load_trial

MODEL = "openrouter/openai/gpt-6-luna"
STAY_MM, MOVED_MM, PILE_MM = 10.0, 30.0, 45.0
SPACE_MM = 32.0  # a set-down spot this far from any matter or anything already put down (fingers open 20 mm + 6 each)
SPOTS = ((0.27, 0.49), (-0.17, 0.07))  # where piles may go (x, y m): the table minus the box row and the far corner

PROMPT = """You control a robot arm at a table. The picture is the table camera's view; each thing the depth scan
found has its number drawn on it. The things found (sizes from the depth scan; colour is the mean of its pixels, RGB):
{objects}

The instruction: "{instruction}"

Decide what to move. Answer in JSON only:
{{
  "see": "one sentence: what is on the table (name each numbered thing: block, bolt, pen, lego brick, die, bin...)",
  "moves": [{{"object": 3, "into": 7}}, {{"object": 4, "pile": "yellow pile"}}],
  "leave": [5],
  "why": "one sentence"
}}
"into" is the number of a container (only containers take things). "pile" is a name: things given the same pile name
are set down together in a free spot (a pile of one thing is how to move something "somewhere else"). Leave out
anything the instruction doesn't ask you to move. Move each thing at most once, in the order to do it."""


def table_picture(c) -> np.ndarray:
    return c.observe(["table"], (1280, 720), "jpeg")["images"]["table"]


def labelled(img: np.ndarray, objs) -> np.ndarray:
    """The table picture cropped to the work area, each found object's number drawn at its top."""
    cam = load()["table"]
    w, h = cam["size"]
    s = img.shape[1] / w
    im = img.copy()
    for o in objs:
        u, v = project(cam, [[*o.centre, SURFACE_Z + o.top_mm / 1000]])[0] * s
        col = (255, 140, 0) if o.container else (255, 255, 255)
        cv2.putText(im, str(o.id), (int(u) - 6, int(v) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, str(o.id), (int(u) - 6, int(v) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    (x0, x1), (y0, y1) = (0.20, 0.56), (-0.34, 0.14)
    uv = project(cam, [[x, y, SURFACE_Z] for x in (x0, x1) for y in (y0, y1)]) * s
    a, b = np.clip(uv.min(0).astype(int), 0, None), uv.max(0).astype(int)
    return im[a[1]:b[1], a[0]:b[0]]


def describe(objs) -> str:
    out = []
    for o in objs:
        kind = f"CONTAINER (open inside, rim {o.rim_mm:.0f} mm up)" if o.container else "solid"
        out.append(f"#{o.id}: {kind}; {o.size_mm[0]:.0f} x {o.size_mm[1]:.0f} mm, {o.top_mm:.0f} mm tall; "
                   f"colour RGB {o.rgb}; at ({o.centre[0] * 1000:.0f}, {o.centre[1] * 1000:.0f}) mm")
    return "\n".join(out)


def free_spot(h: np.ndarray, cols: dict, taken: list, near=None) -> np.ndarray | None:
    """A spot SPACE_MM clear of all matter in the grid and of everything put down so far: the one nearest `near`
    (a pile's middle), or with no `near` the one with the most room around it."""
    from scipy.ndimage import distance_transform_edt
    vox = cols["vox"]
    clear = distance_transform_edt(~np.isfinite(h) | (h < OB.ABOVE_MM)) * vox
    X, Y = h.shape
    xs, ys = cols["x0"] + (np.arange(X) + 0.5) * vox, cols["y0"] + (np.arange(Y) + 0.5) * vox
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    (a, b), (c, d) = SPOTS
    ok = (gx > a) & (gx < b) & (gy > c) & (gy < d) & (clear >= SPACE_MM / 1000)
    room = clear.copy()
    for t in taken:
        dt = np.hypot(gx - t[0], gy - t[1])
        ok &= dt >= SPACE_MM / 1000
        room = np.minimum(room, dt)
    if not ok.any():
        return None
    score = -np.hypot(gx - near[0], gy - near[1]) if near is not None else room
    i = np.unravel_index(np.argmax(np.where(ok, score, -np.inf)), h.shape)
    return np.array([gx[i], gy[i]])


def wait_idle(c) -> None:
    t0 = time.time()
    while time.time() - t0 < 300:
        time.sleep(1.0)
        o = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
        if o["queued"] == 0 and o["status"] == "idle":
            return


def where(c) -> tuple:
    o = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
    return o["tcp_pose"], np.asarray(o["joints"], float)[:6]


# ---- the truth check -------------------------------------------------------------------------------------------

def parse_name(n: str) -> dict:
    color, kind, num = n.split(" ")
    return {"color": color, "kind": kind, "id": num.lstrip("#")}


def check(goals: list, truth: dict, before: dict, after: dict) -> list:
    """Each goal line -> (ok, what it says). truth: name -> mesh (boxes); before/after: id -> pose."""
    boxes = {n: m for n, m in truth.items() if "box" in n}
    things = [n for n in truth if "box" not in n]

    def box_of(n):
        p = np.asarray(after[parse_name(n)["id"]][:3])
        for bn, m in boxes.items():
            v = m.vertices[:, :2]
            poly = Poly(v[np.argsort(np.arctan2(*(v - v.mean(0)).T[::-1]))])
            if poly.contains_points(p[None, :2])[0] and p[2] < m.vertices[:, 2].max():
                return bn
        return None

    def moved_mm(n):
        i = parse_name(n)["id"]
        return float(np.linalg.norm(np.asarray(after[i][:2]) - np.asarray(before[i][:2])) * 1000)

    out = []
    for g in goals:
        sel = [n for n in things if all(parse_name(n)[k] == v for k, v in g["what"].items())]
        label = ", ".join(f"{k} {v}" for k, v in g["what"].items()) or "everything"
        if "into" in g:
            want = g["into"]
            bad = [n for n in sel if not (b := box_of(n)) or (want != "any" and parse_name(b)["color"] != want["color"])]
            out.append((not bad, f"{label} into {'a bin' if want == 'any' else want['color'] + ' bin'}: "
                                 f"{len(sel) - len(bad)}/{len(sel)}" + (f" (not: {', '.join(bad)})" if bad else "")))
        elif g.get("stays"):
            bad = [f"{n} moved {moved_mm(n):.0f} mm" for n in sel if moved_mm(n) > STAY_MM or box_of(n)]
            out.append((not bad, f"{label} left alone: " + ("yes" if not bad else "; ".join(bad))))
        elif g.get("moved"):
            bad = [f"{n} moved {moved_mm(n):.0f} mm" for n in sel if moved_mm(n) < MOVED_MM or box_of(n)]
            out.append((not bad, f"{label} moved elsewhere: " + ("yes, " + ", ".join(f"{moved_mm(n):.0f} mm" for n in sel)
                                                                 if not bad else "; ".join(bad))))
        elif g.get("pile") or g.get("piles_by"):
            groups = {}
            for n in sel:
                groups.setdefault(parse_name(n)[g["piles_by"]] if g.get("piles_by") else "pile", []).append(n)
            mids, ok, notes = {}, True, []
            for k, ns in groups.items():
                xy = np.array([after[parse_name(n)["id"]][:2] for n in ns])
                mid = xy.mean(0)
                spread = float(np.linalg.norm(xy - mid, axis=1).max() * 1000)
                inbox = [n for n in ns if box_of(n)]
                moved = all(moved_mm(n) > 1 for n in ns) or len(ns) > 1
                good = spread <= PILE_MM and not inbox
                ok &= good
                mids[k] = mid
                notes.append(f"{k}: {len(ns)} within {spread:.0f} mm" + (" (some in a bin)" if inbox else ""))
            ks = list(mids)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    d = np.linalg.norm(mids[ks[i]] - mids[ks[j]]) * 1000
                    if d < 2 * PILE_MM:
                        ok = False
                        notes.append(f"{ks[i]} and {ks[j]} piles only {d:.0f} mm apart")
            out.append((ok, f"{label} in {'a pile' if g.get('pile') else 'piles by ' + g['piles_by']}: " + "; ".join(notes)))
        elif g.get("bins_by"):
            per = {}
            loose = [n for n in sel if not box_of(n)]
            for n in sel:
                if (b := box_of(n)):
                    per.setdefault(b, set()).add(parse_name(n)[g["bins_by"]])
            mixed = {b: s for b, s in per.items() if len(s) > 1}
            split = len({k for s in per.values() for k in s}) != sum(len(s) for s in per.values())
            ok = not loose and not mixed and not split
            out.append((ok, f"{label} into bins by {g['bins_by']}: " +
                        "; ".join(f"{b.split(' #')[0]} #{parse_name(b)['id']} has {sorted(s)}" for b, s in per.items()) +
                        (f"; loose: {loose}" if loose else "")))
    return out


# ---- one task ---------------------------------------------------------------------------------------------------

def run(task: str, out, say=print, rerun_to: str | None = None) -> dict:
    from cell_client import CellClient
    from lab.isaaccell import PORT, IsaacCell
    url = f"http://127.0.0.1:{PORT}"
    api = url + "/api/v1/"
    c = CellClient(url, token="sim")
    t = load_trial(task)
    out.mkdir(parents=True)
    t0 = time.time()
    requests.post(api + "sim/honest", json={"on": False}, timeout=30)
    IsaacCell(build(t, 0))
    before_img = table_picture(c)
    say(f"\n=== {task}: \"{t['instruction']}\"")
    say("  scanning (safe height, sweep, depth on the GPU box)...")
    r = subprocess.run([sys.executable, "-m", "lab.depth_check", "--fast", "--no-reset", "--trial", task],
                       capture_output=True, text=True, env={**os.environ, **({"RERUN_TO": rerun_to} if rerun_to else {})})
    scan_log = r.stdout + r.stderr
    (out / "scan.txt").write_text("\n".join(l for l in scan_log.splitlines() if "token" not in l.lower()))
    if r.returncode:
        say("  the scan failed: " + scan_log.strip().splitlines()[-1])
        return {"task": task, "ok": False, "why": "scan failed", "seconds": time.time() - t0}
    for l in scan_log.splitlines():
        if l.startswith(("tallest", "rigid objects", "seen ")):
            say("  " + l)
    t_scan = time.time() - t0

    link = DepthLink()
    g, cols = link.grid(), link.columns()
    objs = OB.find(cols, g["xyz"], g["rgb"])
    truth = sim_objects(task)
    before = requests.get(api + "sim/truth", timeout=30).json()["poses"]
    pic = labelled(before_img, objs)
    cv2.imwrite(str(out / "before.jpg"), cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
    say(f"  {len(objs)} things found ({sum(o.container for o in objs)} containers); asking the LLM...")

    t1 = time.time()
    ans = vlm.ask_image(pic, PROMPT.format(objects=describe(objs), instruction=t["instruction"]), model=MODEL,
                        think=True, timeout=120)
    plan = (ans or {}).get("json") or {}
    t_llm = time.time() - t1
    (out / "plan.json").write_text(json.dumps({"answer": plan, "text": (ans or {}).get("text"), "seconds": t_llm,
                                               "objects": describe(objs)}, indent=1))
    say(f"  LLM ({t_llm:.0f} s): {plan.get('see')}")
    say(f"    moves {plan.get('moves')}; leave {plan.get('leave')}; why: {plan.get('why')}")

    by_id = {o.id: o for o in objs}
    h = HD.column_heights(cols)
    taken, piles, done = [], {}, []
    for m in plan.get("moves") or []:
        o = by_id.get(m.get("object"))
        if o is None or o.container:
            say(f"  skip {m}: no such solid object")
            continue
        dest, place = None, None
        if m.get("into") is not None:
            dest = by_id.get(m["into"])
            if dest is None or not dest.container:
                say(f"  skip {m}: #{m['into']} is not a container")
                continue
        else:
            name = str(m.get("pile") or "elsewhere")
            members = piles.setdefault(name, [])
            place = free_spot(h, cols, taken, np.mean(members, 0) if members else None)
            if place is None:
                say(f"  skip {m}: no free spot left for pile {name!r}")
                continue
        cur, q = where(c)
        plan_m = HD.put(o, dest, objs, cols, cur, place)
        if not plan_m.ok:
            say(f"  #{o.id}: no plan ({plan_m.why})")
            continue
        try:
            jrows = joints_for(HD.rows(plan_m, cur), q)
        except ValueError as e:
            say(f"  #{o.id}: out of reach ({e})")
            continue
        if place is not None:
            taken.append(place)
            piles[name].append(place)
        say(f"  #{o.id} -> " + (f"bin #{dest.id}" if dest else f"pile {name!r} at ({place[0] * 1000:.0f}, "
                                                                 f"{place[1] * 1000:.0f}) mm") +
            f": grip {plan_m.grip_mm:.1f} mm, carry {plan_m.carry_mm:.0f} mm, release {plan_m.release_mm:.0f} mm")
        for k in range(0, len(jrows), 100):
            c.act(jrows[k:k + 100], P.DT)
        wait_idle(c)
        done.append(m)
    time.sleep(2.0)
    after = requests.get(api + "sim/truth", timeout=30).json()["poses"]
    cv2.imwrite(str(out / "after.jpg"), cv2.cvtColor(labelled(table_picture(c), objs), cv2.COLOR_RGB2BGR))
    lines = check(t["goal"], truth, before, after)
    ok = all(x for x, _ in lines)
    for x, s in lines:
        say(f"  {'PASS' if x else 'FAIL'}  {s}")
    res = {"task": task, "instruction": t["instruction"], "ok": ok, "checks": [{"ok": x, "says": s} for x, s in lines],
           "llm": plan, "moves_done": done, "seconds": round(time.time() - t0), "scan_s": round(t_scan),
           "llm_s": round(t_llm)}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    say(f"  {'PASSED' if ok else 'FAILED'} in {res['seconds']} s (scan {res['scan_s']} s, LLM {res['llm_s']} s)")
    return res


def sheet(results: list, root) -> None:
    rows = []
    for r in results:
        d = root / r["task"]
        ims = [cv2.imread(str(d / f)) for f in ("before.jpg", "after.jpg")]
        if any(i is None for i in ims):
            continue
        ims = [cv2.resize(i, (480, int(i.shape[0] * 480 / i.shape[1]))) for i in ims]
        row = np.hstack([ims[0], np.full((ims[0].shape[0], 6, 3), 40, np.uint8), ims[1]])
        bar = np.full((34, row.shape[1], 3), (40, 140, 40) if r["ok"] else (40, 40, 180), np.uint8)
        cv2.putText(bar, f"{r['task']}: {'PASS' if r['ok'] else 'FAIL'}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        rows.append(np.vstack([bar, row]))
    if rows:
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(str(root / "sheet.jpg"), np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = sorted(f.stem for f in TRIALS.glob("ladder*.yaml"))
    if a.list:
        for n in names:
            print(f"{n:24} {load_trial(n)['instruction']}")
        return
    if a.tasks:
        names = [n for n in names if any(n.startswith(p) for p in a.tasks)]
    from lab.isaaccell import tunnel as isaac_tunnel
    isaac_tunnel()
    tunnel()
    root = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_ladder"
    results = []
    for n in names:
        try:
            results.append(run(n, root / n))
        except Exception as e:  # one task failing doesn't stop the ladder
            print(f"  {n} crashed: {type(e).__name__}: {e}")
            results.append({"task": n, "ok": False, "why": f"{type(e).__name__}: {e}"})
        (root / "results.json").write_text(json.dumps(results, indent=1, default=float))
        sheet(results, root)
    print("\n" + "\n".join(f"{'PASS' if r['ok'] else 'FAIL'}  {r['task']}" for r in results))
    print(f"saved {root}")


if __name__ == "__main__":
    main()
