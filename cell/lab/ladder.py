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
BIN_MIN_MM, BIN_TOP_MM, WALL_MM = 40.0, 25.0, 12.0  # bin-sized, and how thick its walls are taken to be
SWEEP_MM_S = 250.0  # the opening sweep: one quick pass over the table, then pick
REAL_SWEEP_MM_S = 150.0  # the real cell: slower, sharper (250 smeared a 15 mm cube to 23 mm, Oct 30), delay measured
DELAY_MS = 35.0  # the wrist pictures' delay in the sim (measured 32-38 ms by the twitch, Oct 30): no twitch per task
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
  "names": {{"1": "short name, e.g. red block, blue bin, bolt"}},
  "say": "what you tell the person watching, first person, plain words, one or two sentences: what you were asked and what you're going to do about it",
  "moves": [{{"object": 3, "into": 7}}, {{"object": 4, "pile": "yellow pile"}}],
  "leave": [5],
  "why": "one sentence"
}}
"into" is the number of a container (only containers take things). "pile" is a name: things given the same pile name
are set down together in a free spot (a pile of one thing is how to move something "somewhere else"). Leave out
anything the instruction doesn't ask you to move. Move each thing at most once, in the order to do it."""


def table_picture(c) -> np.ndarray:
    return c.observe(["table"], (1280, 720), "jpeg")["images"]["table"]


FIRST = """You control a robot arm at a table. The instruction: "{instruction}".
The TOP picture is the table camera looking down at the work area; below it, left: the side camera, right: the camera
on the robot's wrist. A depth scan is running at the same time and will measure every thing's exact place and size; you
say what things are and what to do. Answer in JSON only:
{{
  "say": "what you tell the person watching, first person, plain words, one or two sentences: what you see and what you'll do",
  "things": [{{"name": "red block", "at": [0.42, 0.61]}}, {{"name": "white bin", "at": [0.20, 0.30]}}],
  "moves": [{{"thing": "red block", "into": "white bin"}}, {{"thing": "yellow block", "pile": "yellow pile"}}],
  "why": "one sentence"
}}
"things": every thing on the table that matters, each with a unique name and "at" = where its middle is in the TOP
picture, as fractions of the picture's width and height (0,0 top left). "into" names a container from "things";
"pile" names a pile (things with the same pile name are set down together in a free spot; a pile of one thing is how to
move something "somewhere else"). Leave out anything the instruction doesn't ask to move; move each thing at most once,
in the order to do it."""


def crop_box(img: np.ndarray):
    """The work area in the table picture: (corner (u, v), far corner (u, v), picture px per calibration px)."""
    cam = load()["table"]
    s = img.shape[1] / cam["size"][0]
    (x0, x1), (y0, y1) = (0.20, 0.56), (-0.34, 0.14)
    uv = project(cam, [[x, y, SURFACE_Z] for x in (x0, x1) for y in (y0, y1)]) * s
    return np.clip(uv.min(0).astype(int), 0, None), uv.max(0).astype(int), s


def first_look(obs) -> np.ndarray:
    """The pictures the language model gets at second 0: the table camera's work area on top, side and wrist below."""
    tab = obs["images"]["table"]
    a, b, _ = crop_box(tab)
    top = cv2.resize(tab[a[1]:b[1], a[0]:b[0]], (1024, int(1024 * (b[1] - a[1]) / (b[0] - a[0]))))
    small = [cv2.resize(obs["images"][k], (512, 288)) for k in ("side", "wrist")]
    for im, k in zip(small, ("side", "wrist")):
        cv2.putText(im, k, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([top, np.hstack(small)])


def match(first: dict, objs, img: np.ndarray, near: float = 0.08) -> tuple[dict, list]:
    """The first look's named things -> the scan's objects: the object whose middle (projected into the top picture)
    is nearest the point the language model gave, containers for "into", solids for what's moved. Returns the plan in
    object numbers ({"moves": [...], "names": {...}}) and what couldn't be matched."""
    cam = load()["table"]
    a, b, s = crop_box(img)
    def frac(o):
        u, v = project(cam, [[*o.centre, SURFACE_Z + o.top_mm / 1000]])[0] * s
        return np.array([(u - a[0]) / (b[0] - a[0]), (v - a[1]) / (b[1] - a[1])])
    at = {t_["name"]: np.asarray(t_.get("at", [np.nan, np.nan]), float) for t_ in first.get("things") or []}
    names, moves, missing = {}, [], []
    def find(name, want_container):
        if name not in at or not np.isfinite(at[name]).all():
            return None
        # a bin the depth map didn't see as open (a plain inside gives it nothing to match) is still a bin when the
        # language model says so and it's bin-sized: open it up, the opening its outline less the walls
        cands = [o for o in objs if o.container == want_container or
                 (want_container and min(o.size_mm) >= BIN_MIN_MM and o.top_mm >= BIN_TOP_MM)]
        if not cands:
            return None
        d = [np.linalg.norm(frac(o) - at[name]) for o in cands]
        i = int(np.argmin(d))
        if d[i] > near:
            return None
        o = cands[i]
        if want_container and not o.container:
            c_ = o.footprint.mean(0)
            r_ = np.linalg.norm(o.footprint - c_, axis=1, keepdims=True)
            o.container, o.rim_mm = True, o.top_mm
            o.opening = c_ + (o.footprint - c_) * np.clip(1 - WALL_MM / 1000 / np.maximum(r_, 1e-6), 0.1, 1)
        return o
    for m in first.get("moves") or []:
        o = find(m.get("thing"), False)
        if o is None:
            missing.append(m.get("thing"))
            continue
        names[o.id] = m["thing"]
        if m.get("into"):
            d = find(m["into"], True)
            if d is None:
                missing.append(m["into"])
                continue
            names[d.id] = m["into"]
            moves.append({"object": o.id, "into": d.id})
        else:
            moves.append({"object": o.id, "pile": m.get("pile") or "elsewhere"})
    return {"moves": moves, "names": names, "why": first.get("why"), "see": first.get("say")}, missing


HOVER_ABOVE_MM = 25.0  # the fingertips this far over the target's top while the wrist camera takes its still picture
SNAP_MIN_MM = 7.0      # narrower than this in the still picture: only part of it seen
SNAP_FAR_MM = 25.0     # the picture's object further than this from where depth put it: not trusted


def queued_of(c) -> int:
    return c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})["queued"]


def snap(c, o, plan, out, say=print, cols=None):
    """Hover and snap: the wrist camera straight over the target, the arm still, a picture; the object found by
    colour near where depth put it, its outline (edge at half contrast) on the plane of its top -> its centre, angle
    and width. Then a second look with the camera dead over that centre (no side faces in view), which is the one
    used. No timing guess: nothing moves while a picture is taken. Returns (plan with the grip re-aimed, why) or
    (None, why)."""
    import dataclasses
    from calibration.geometry import UF0_Z_MM, flange_pose, project
    from calibration.kinematics import DOWN, tool_ik, tool_pose, wpr_of
    from lab import scan as SC
    from lab.scene3d import cameras, rays_to_plane
    cam, T_fc, _ = cameras(None)
    tm = lambda p_: np.array([p_[0], p_[1], p_[2] - UF0_Z_MM]) / 1000
    z_top = SURFACE_Z + o.top_mm / 1000

    def hover(target):  # the camera over target (base m): fingertips HOVER_ABOVE_MM over its top, clear of neighbours
        cur, q = where(c)
        tip_xy = np.asarray(target) - SC.cam_offset(q, T_fc)
        near = HD.path_max(cols, HD.column_heights(cols), tip_xy, tip_xy, None, half_mm=30) if cols is not None else 0
        tip_h = max(o.top_mm + HOVER_ABOVE_MM, near + 15.0)
        goal = P.uf0(*tip_xy, tip_h)
        Tf = tool_pose(q)
        qg = tool_ik(tm(goal), DOWN, q, near=True)
        rows = P.joint_move(q, Tf[:3, 3], wpr_of(Tf[:3, :3]), qg, tm(goal), DOWN,
                            height_mm=max(min(plan.carry_mm, tip_h + 30), tip_h))
        if rows is None:
            return False
        SC.feed(c, [list(r) for r in rows], lambda: queued_of(c))
        wait_idle(c)
        time.sleep(0.4)  # still
        return True

    def look(target, k_):
        ob = c.observe(["wrist"], tuple(cam["size"]), "png")
        img, j = ob["images"]["wrist"], np.asarray(ob["joints"], float)
        camd = dict(cam, T=flange_pose(j[:6]) @ T_fc)
        u0, v0 = project(camd, [[*target, z_top]])[0]
        r_px = float(np.hypot(*(project(camd, [[target[0] + 0.03, target[1], z_top]])[0] - (u0, v0)))) + 5  # 30 mm
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
        med = np.median(lab.reshape(-1, 3), 0)  # the table (most of the picture)
        d = np.sqrt(((lab[..., 0] - med[0]) * 0.5) ** 2 + (lab[..., 1] - med[1]) ** 2 + (lab[..., 2] - med[2]) ** 2)
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        win = np.hypot(xx - u0, yy - v0) < r_px
        shot = img.copy()
        cv2.circle(shot, (int(u0), int(v0)), int(r_px), (255, 255, 0), 1)
        save = lambda: cv2.imwrite(str(out / f"snap_{o.id}_{k_}.jpg"), cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))

        def blob(thr):
            m = cv2.morphologyEx(((d > thr) & win).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab_, st, cen = cv2.connectedComponentsWithStats(m)
            if n <= 1:
                return None
            i = 1 + int(np.argmin([np.hypot(*(cen[i_] - (u0, v0))) - 0.001 * st[i_, cv2.CC_STAT_AREA] for i_ in range(1, n)]))
            return lab_ == i
        b = blob(22.0)
        if b is None:
            save()
            return None
        core = cv2.erode(b.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if core.any():  # the edge where the colour is halfway between table and object (not the blurred rim)
            b2 = blob(max(22.0, 0.5 * float(np.median(d[core]))))
            b = b2 if b2 is not None else b
        cnt, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnt, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2).astype(float)
        c_px = pts.mean(0)
        pts = pts + 0.5 * np.sign(c_px - pts)  # boundary pixels' centres sit half a pixel outside the edge
        xyz = rays_to_plane(camd, pts, z_top)
        (cx, cy), (w, l), ang = cv2.minAreaRect((xyz[:, :2] * 1000).astype(np.float32))
        cv2.drawContours(shot, [cnt], -1, (0, 255, 0), 1)
        save()
        return np.array([cx, cy]) / 1000, (w, l), ang

    if not hover((plan.grip.x, plan.grip.y)):
        return None, "no way to hover over it from here"
    m1 = look((plan.grip.x, plan.grip.y), 1)
    if m1 is None:
        return None, "nothing stands out from the table where depth put it"
    shift = float(np.linalg.norm(m1[0] - (plan.grip.x, plan.grip.y)) * 1000)
    if shift > SNAP_FAR_MM:
        return None, f"what I see is {shift:.0f} mm from where depth put it: not sure it's the same thing"
    m = m1  # the second look, the camera dead over it: the top face only
    if hover(m1[0]):
        m2 = look(m1[0], 2)
        if m2 is not None and np.linalg.norm(m2[0] - m1[0]) * 1000 < 10:
            m = m2
    (cxy, (w, l), ang) = m
    shift = float(np.linalg.norm(cxy - (plan.grip.x, plan.grip.y)) * 1000)
    say(f"  snap #{o.id}: look 1 {m1[1][0]:.1f} x {m1[1][1]:.1f} mm; " + ("look 2 " if m is not m1 else "(no look 2) ") +
        f"{w:.1f} x {l:.1f} mm at ({cxy[0] * 1000:.0f}, {cxy[1] * 1000:.0f}) mm, angle {ang:.0f} deg, "
        f"{shift:.1f} mm from the depth estimate")
    if min(w, l) < SNAP_MIN_MM:  # a sliver of it: the look didn't see the whole thing
        return None, f"I only see {min(w, l):.0f} mm of it: not a clear enough look"
    jaw_room = 20.0 - 2 * 2.0  # the open gap less the clearance each side (lab/grasp.GAP, CLEAR)
    if min(w, l) > jaw_room:
        return None, f"it's {min(w, l):.1f} mm across: too wide for the jaws ({jaw_room:.0f} mm with room to spare)"
    # the jaws close across a side: the side angle or 90 deg from it, whichever is nearer the planned yaw
    cands = [ang % 180, (ang + 90) % 180]
    yaw = min(cands, key=lambda a_: abs(((a_ - plan.grip.yaw) + 90) % 180 - 90))
    g = dataclasses.replace(plan.grip, x=float(cxy[0]), y=float(cxy[1]), yaw=float(yaw))
    return dataclasses.replace(plan, grip=g), (f"{min(w, l):.1f} mm across, moved the aim {shift:.1f} mm, "
                                               f"jaws at {yaw:.0f} deg")


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


def wait_idle(c, each=None) -> None:
    t0, zeros = time.time(), False
    while time.time() - t0 < 120:  # (a pick and place is well under a minute)
        time.sleep(1.0)
        o = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
        if each:
            each(o["queued"])
        if o["queued"] == 0:  # (the real cell's status says "running" all trial long) twice in a row = settled
            if zeros:
                return
            zeros = True
        else:
            zeros = False


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

def _things(objs) -> str:
    """The found objects in a sentence: 'two small solids (about 10 mm) and one open container (45 mm)'."""
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    n = lambda k: words[k] if k < len(words) else str(k)
    solids = [o for o in objs if not o.container]
    boxes = [o for o in objs if o.container]
    parts = []
    if solids:
        tops = sorted(o.top_mm for o in solids)
        parts.append(f"{n(len(solids))} solid thing{'s' * (len(solids) != 1)} "
                     f"({tops[0]:.0f} mm tall" + (f" to {tops[-1]:.0f} mm)" if tops[-1] - tops[0] > 2 else ")"))
    if boxes:
        parts.append(f"{n(len(boxes))} open container{'s' * (len(boxes) != 1)} (rim{'s' * (len(boxes) != 1)} "
                     f"{', '.join(f'{b.rim_mm:.0f}' for b in boxes)} mm up)")
    return " and ".join(parts) or "nothing"


GLANCE = """You control a robot arm at a table; the picture is the table camera's view. You've been asked: "{instruction}".
Before the depth scan finishes, glance at the picture and say in one or two plain first-person sentences what you see on
the table and roughly what you'll do. JSON only: {{"say": "..."}}"""


def run(task: str, out, say=print, rerun_to: str | None = None, tell=None, state=None, real=None) -> dict:
    """say: detail lines (a log); tell: what the robot would say to a person watching, as things change;
    state(lane, text): live status, lanes "arm" (what it's doing), "where" (depth: the scan, the 3D map),
    "what" (the language model: what things are, the plan)."""
    tell = tell or (lambda m: say("  " + m))
    state = state or (lambda lane, text: None)
    from cell_client import CellClient
    from lab.isaaccell import PORT, IsaacCell
    if real:  # (url, token, ca) of the real cell, a trial already running: no reset, no sim truth
        url, token_, ca_ = real
        c = CellClient(url, token_, ca_)
    else:
        url = f"http://127.0.0.1:{PORT}"
        c = CellClient(url, token="sim")
    api = url + "/api/v1/"
    t = load_trial(task)
    out.mkdir(parents=True)
    t0 = time.time()
    state("where", "nothing yet")
    state("what", "nothing yet")
    t_ = time.time()
    if not real:
        state("arm", "switching the sim's timing mode off for the reset")
        requests.post(api + "sim/honest", json={"on": False}, timeout=30)
        t_ = time.time()
        state("arm", "resetting the table (the simulator rebuilds the scene)")
        IsaacCell(build(t, 0))
    t_reset = time.time() - t_
    state("arm", "taking the first pictures (table, side, wrist, table depth)")
    t_go = time.time()  # second 0: from here everything runs at once
    obs0 = c.observe(["table", "side", "wrist"], (1280, 720), "jpeg", depth=["table"])
    np.savez(out / "first.npz", depth=obs0["depth"]["table"], joints=np.asarray(obs0["joints"], float),
             table=cv2.resize(obs0["images"]["table"], (960, 540)), side=cv2.resize(obs0["images"]["side"], (960, 540)))
    say(f"  timing: honest-off {t_ - t0:.1f} s, reset {t_reset:.1f} s, first pictures {time.time() - t_ - t_reset:.1f} s")
    before_img = obs0["images"]["table"]
    look = first_look(obs0)
    cv2.imwrite(str(out / "first_look.jpg"), cv2.cvtColor(look, cv2.COLOR_RGB2BGR))
    first = {}

    def plan_first():  # second 0: the language model gets every camera and the instruction, while the scan runs
        t_ = time.time()
        state("what", "looking at the table, side and wrist pictures and deciding what to do...")
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(vlm.ask_image, look, FIRST.format(instruction=t["instruction"]), model=MODEL,
                                think=True, timeout=120)
                while not fut.done():
                    time.sleep(1.0)
                    state("what", f"deciding what to do from the first pictures ({time.time() - t_:.0f} s)")
                a_ = fut.result()
            first.update((a_ or {}).get("json") or {})
            first["_s"] = time.time() - t_
            if first.get("say"):
                tell(first["say"])
            mv = "; ".join(f"{m.get('thing')} -> {m.get('into') or 'pile ' + str(m.get('pile'))}" for m in first.get("moves") or [])
            state("what", f"plan ready in {first['_s']:.0f} s: " + (mv or "nothing to move") +
                  " (waiting for the scan to say exactly where)")
        except Exception as e:
            state("what", f"the first look failed: {e}")
    import threading
    th_first = threading.Thread(target=plan_first, daemon=True)
    th_first.start()
    say(f"  LLM asked at {time.time() - t_go:.1f} s")
    say(f"\n=== {task}: \"{t['instruction']}\"")
    tell("Looking at everything at once: the language model is reading the pictures while the arm sweeps the "
         "table and the GPU box builds the 3D map.")
    env = {**os.environ, **({"RERUN_TO": rerun_to} if rerun_to else {})}
    if real:  # the real cell: quick start, a 4 s wrist twitch to measure its picture delay, a 150 mm/s sweep
        env |= {"CELL_URL": url, "CELL_TOKEN": token_, **({"CELL_CA": ca_} if ca_ else {})}
    cmd = (["--first", str(out / "first.npz"), "--speed", str(REAL_SWEEP_MM_S), "--short-twitch"] if real else
           ["--first", str(out / "first.npz"), "--speed", str(SWEEP_MM_S), "--delay-ms", str(DELAY_MS)])
    p = subprocess.Popen([sys.executable, "-u", "-m", "lab.depth_check", "--no-reset", "--trial", task, *cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines_, timing = [], {}
    import queue as _q
    inbox = _q.Queue()
    def _read():
        for l_ in p.stdout:
            inbox.put(l_)
    threading.Thread(target=_read, daemon=True).start()

    def scan_lines():  # its lines until it exits (a background process holding the pipe open can't stall this)
        quiet = 0.0
        while True:
            try:
                yield inbox.get(timeout=1.0)
                quiet = 0.0
            except _q.Empty:
                if p.poll() is not None:
                    return
                quiet += 1.0
                if quiet >= 60:  # no heartbeat for a minute: the scan is hung
                    p.kill()
                    yield "the scan said nothing for 60 s: stopped it"
                    return
    for l in scan_lines():
        l = l.rstrip()
        if "token" in l.lower():
            continue
        if l.startswith("HB\t"):
            _, where_, arm_ = (l.split("\t") + ["", ""])[:3]
            if arm_.startswith("moving") and "t_move" not in timing:
                timing["t_move"] = round(time.time() - t_go, 1)
                say(f"  arm moving at {timing['t_move']} s")
            state("where", where_)
            state("arm", arm_)
            continue
        lines_.append(l)
        if l.startswith("tallest thing"):
            import re
            m = re.search(r"tallest thing on the table (\d+) mm.*fly at (\d+) mm", l)
            if m:
                tell(f"The tallest thing on the table is {m[1]} mm, so I'll sweep the wrist camera over everything with "
                     f"the fingertips {m[2]} mm up. Nothing on the table can be hit at that height.")
        elif l.startswith("picture delay"):
            ms = l.split()[2]
            tell("Sweeping the wrist camera over the table now, building the 3D map on the GPU box as I go: the dots "
                 "on the right are where I'm sure there's something.")
        elif l.startswith(("tallest", "the scan:", "motion:", "rigid objects", "seen ")) or "Error" in l:
            say("  " + l)
    p.wait()
    scan_log = "\n".join(lines_)
    (out / "scan.txt").write_text(scan_log)
    if p.returncode:
        tell("The scan failed: " + (lines_[-1] if lines_ else "no output") + ". I'm stopping this task.")
        return {"task": task, "ok": False, "why": "scan failed: " + (lines_[-1] if lines_ else ""), "seconds": time.time() - t0}
    t_scan = time.time() - t0

    link = DepthLink()
    g, cols = link.grid(), link.columns()
    objs = OB.find(cols, g["xyz"], g["rgb"])
    truth = {} if real else sim_objects(task)
    before = {} if real else requests.get(api + "sim/truth", timeout=30).json()["poses"]
    pic = labelled(before_img, objs)
    cv2.imwrite(str(out / "before.jpg"), cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
    state("where", f"map done: {len(objs)} things ({sum(o.container for o in objs)} open containers)")
    tell(f"Scan done. The depth map has {_things(objs)}.")
    t1 = time.time()
    while th_first.is_alive():  # (usually done long before the scan)
        state("arm", "still, waiting for the language model's plan")
        time.sleep(1.0)
    plan, missing = match(first, objs, before_img) if first else ({}, ["(no first plan)"])
    ans = {"text": json.dumps(first)}
    if missing or not first:
        tell("I can't line up " + (", ".join(map(str, missing)) or "the plan") + " with what the depth scan found, "
             "so I'm asking again with the found things numbered on the picture.")
        state("what", "re-planning with the numbered picture...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(vlm.ask_image, pic, PROMPT.format(objects=describe(objs), instruction=t["instruction"]),
                            model=MODEL, think=True, timeout=120)
            while not fut.done():
                time.sleep(1.0)
                state("what", f"re-planning with the numbered picture ({time.time() - t1:.0f} s)")
            ans = fut.result()
        plan = (ans or {}).get("json") or {}
        plan.pop("say", None)
    else:
        tell("Matched the plan to the depth map: " + "; ".join(
            f"{plan['names'][m['object']]} is #{m['object']}" for m in plan["moves"]) + ".")
    t_llm = first.get("_s", 0.0) + (time.time() - t1)
    (out / "plan.json").write_text(json.dumps({"answer": plan, "text": (ans or {}).get("text"), "seconds": t_llm,
                                               "objects": describe(objs)}, indent=1))
    say(f"  LLM ({t_llm:.0f} s): {plan.get('see')}")
    say(f"    moves {plan.get('moves')}; leave {plan.get('leave')}; why: {plan.get('why')}")
    state("what", "plan: " + (plan.get("why") or plan.get("see") or "no answer"))
    names = {int(k): v for k, v in (plan.get("names") or {}).items() if str(k).isdigit()}
    nm = lambda i: f"the {names[i]}" if i in names else f"#{i}"
    if plan.get("say"):
        tell(plan["say"])
    elif not plan:
        tell("I couldn't get a plan from the language model, so I'm not moving anything.")
    if plan and not plan.get("moves"):
        tell("Nothing needs moving.")

    by_id = {o.id: o for o in objs}
    h = HD.column_heights(cols)
    taken, piles, done = [], {}, []
    for m in plan.get("moves") or []:
        o = by_id.get(m.get("object"))
        if o is None or o.container:
            tell(f"(The plan named #{m.get('object')}, but that's not something I can pick up. Skipping it.)")
            continue
        dest, place = None, None
        if m.get("into") is not None:
            dest = by_id.get(m["into"])
            if dest is None or not dest.container:
                tell(f"(The plan said to put it in #{m['into']}, but I don't see an open container there. Skipping.)")
                continue
        else:
            name = str(m.get("pile") or "elsewhere")
            members = piles.setdefault(name, [])
            place = free_spot(h, cols, taken, np.mean(members, 0) if members else None)
            if place is None:
                tell("I can't find a free spot on the table to set it down. Skipping it.")
                continue
        cur, q = where(c)
        plan_m = HD.put(o, dest, objs, cols, cur, place)
        if not plan_m.ok:
            tell(f"I can't work out how to pick up {nm(o.id)}: {plan_m.why}.")
            continue
        state("arm", f"hovering over {nm(o.id)} for a still picture")
        try:
            snapped, why_ = snap(c, o, plan_m, out, say, cols=cols)
        except Exception as e:
            snapped, why_ = None, f"the close look failed ({type(e).__name__}: {e})"
        if snapped is None:
            tell(f"I looked closely at {nm(o.id)} and I'm not going down: {why_}.")
            continue
        tell(f"Close look at {nm(o.id)}: {why_}. Going down.")
        plan_m = snapped
        cur, q = where(c)
        try:
            jrows = joints_for(HD.rows(plan_m, cur), q)
        except ValueError as e:
            tell(f"{nm(o.id).capitalize()} is out of the arm's reach from here. Skipping it.")
            continue
        if place is not None:
            taken.append(place)
            piles[name].append(place)
        tell(f"Picking up {nm(o.id)} (#{o.id}) and " + (f"dropping it in {nm(dest.id)} (#{dest.id})" if dest else
             f"setting it down in a free spot" + (" next to the others" if len(piles[name]) > 1 else "")) +
             f": fingers {plan_m.grip_mm:.0f} mm off the table to grab it, carried {plan_m.carry_mm:.0f} mm up so it "
             f"clears everything on the way.")
        say(f"  #{o.id} -> " + (f"bin #{dest.id}" if dest else f"pile {name!r} at ({place[0] * 1000:.0f}, "
                                                                 f"{place[1] * 1000:.0f}) mm") +
            f": grip {plan_m.grip_mm:.1f} mm, carry {plan_m.carry_mm:.0f} mm, release {plan_m.release_mm:.0f} mm")
        if "t_pick" not in timing:
            timing["t_pick"] = round(time.time() - t_go, 1)
            say(f"  first pick sent at {timing['t_pick']} s")
        what_ = f"{nm(o.id)} -> " + (nm(dest.id) if dest else "a free spot")
        from lab import scan as SC
        SC.feed(c, jrows, lambda: c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36,
                                                                         "format": "jpeg"})["queued"],
                each=lambda q: state("arm", f"moving {what_} ({q / 10:.0f} s of motion queued)"))
        wait_idle(c, lambda q: state("arm", f"moving {what_} ({q / 10:.0f} s of motion left)"))
        state("arm", f"done: {what_}")
        done.append(m)
    state("arm", "still: checking the result")
    time.sleep(2.0)
    after = {} if real else requests.get(api + "sim/truth", timeout=30).json()["poses"]
    cv2.imwrite(str(out / "after.jpg"), cv2.cvtColor(labelled(table_picture(c), objs), cv2.COLOR_RGB2BGR))
    lines = [] if real else check(t["goal"], truth, before, after)
    ok = all(x for x, _ in lines)
    for x, s_ in lines:
        say(f"  {'PASS' if x else 'FAIL'}  {s_}")
    if real:
        tell("Done. Look at the table: there's no simulator to check it here.")
    else:
        tell(("Done, and the simulator agrees: " if ok else "Done, but the simulator says it's not right: ") +
             "; ".join(s_ for _, s_ in lines) + ".")
    res = {"task": task, "instruction": t["instruction"], "ok": ok, "checks": [{"ok": x, "says": s} for x, s in lines],
           "llm": plan, "moves_done": done, "timing": timing, "seconds": round(time.time() - t0), "scan_s": round(t_scan),
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
