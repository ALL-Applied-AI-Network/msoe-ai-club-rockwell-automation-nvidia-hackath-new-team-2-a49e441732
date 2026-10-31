"""The look-and-act flow (task YAML `mode: v2`): no video sweep. What each piece does:

    1. first look (second 0): every camera to the language model with the instruction -> the things that matter, each
       with a SAM word, a kind (small / long / large / container / zone) and a point; the moves (into / to a zone /
       push to); runs while the arm already heads for the first look station
    2. look stations: the wrist camera over each thing the plan touches (its point), the arm still, one shot (wrist
       colour + depth together) -> lab/look.measure: SAM masks + depth measured from the shot's own table. Small
       things get a second, closer look ("is something really here", and a sharper outline)
    3. the move by kind (lab/moves): small -> push_grasp; long / large -> pick_top (the pen path); push_to -> push.
       Jaws close across the narrow side; grip low; the travel height clears every thing seen (the target too)
    4. every move is checked whole before it's sent (lab/moves.joints: reach and the padded self-collision check)
Each switch is a constant here: CLOSE_LOOK (the second look at small things), MULTI_VIEW (every look's masks of a
thing merged, not just the nearest), so a piece can be turned off to see what it buys.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, load, pixel_to_base
from lab import look as LK
from lab import moves as MV
from lab import primitives as P

CLOSE_LOOK = True
MULTI_VIEW = True
LOOK_TIP_MM = 100.0      # fingertips this high for a look station (the wrist camera ~230 mm over the table)
CLOSE_TIP_MM = 60.0      # and for the second look at a small thing (the camera ~190 mm up)
NEAR_M = 0.06            # a mask this close to where the language model pointed is that thing
SMALL_MM = 25.0          # a thing whose long side is shorter than this is small (push-grasp)
JAW_ROOM_MM = 18.0       # the open jaws take a narrow side up to this
CLEAR_MM = 15.0          # the travel height clears the tallest thing on the way by this
WORDS = {"small": ["cube", "block"], "long": ["stick", "cylinder"], "container": ["cup", "box"],
         "large": ["box", "object"]}  # generic shapes with the language model's word (SAM missed a plain "pen" in sim)

PLAN = """You control a robot arm at a table. The instruction: "{instruction}".
The TOP picture is the table camera looking down at the work area; below it, left: the side camera, right: the camera
on the robot's wrist. Answer in JSON only:
{{
  "say": "what you tell the person watching, first person, one or two plain sentences: what you see, what you'll do",
  "things": [{{"name": "red block", "word": "cube", "kind": "small", "at": [0.42, 0.61]}},
             {{"name": "blue bin", "word": "bin", "kind": "container", "at": [0.20, 0.30]}}],
  "moves": [{{"thing": "red block", "into": "blue bin"}},
            {{"thing": "black pen", "to": [0.70, 0.40]}},
            {{"thing": "white cup", "push_to": [0.50, 0.50]}}],
  "why": "one sentence"
}}
things: every thing that matters, a unique name each; "word": one or two words to find it in a picture (cube, pen,
marker, bolt, battery, screwdriver, cup, bin...); "kind": small (under ~25 mm), long (pens, markers, batteries,
bolts, screwdrivers), large (bigger, not open), container (cups, bins, trays you can drop things in); "at": its middle
in the TOP picture as fractions of the picture's width and height (0,0 top left).
moves, in the order to do them: "into": a container's name; "to": a spot on the table in the TOP picture's fractions
(for "somewhere else", a row, a group or a zone: pick spots on clear table, about 6 cm apart); "push_to": slide a
large thing or a container to a spot (it isn't lifted). Leave out what the instruction doesn't ask to move."""


def where(c):
    o = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
    return o["tcp_pose"], np.asarray(o["joints"], float)


def queued(c):
    return c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})["queued"]


def wait_still(c, timeout: float = 120.0):
    t0, zeros = time.time(), 0
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        zeros = zeros + 1 if queued(c) == 0 else 0
        if zeros >= 2:
            return


def still_shot(c, shot_id: str, timeout: float = 8.0):
    """Wrist colour + depth once the arm has stopped (two readings, same joints, nothing queued)."""
    prev = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        ob = c.observe(["wrist"], (1280, 720), "jpeg", 90, depth=["wrist"])
        j = np.asarray(ob["joints"], float)
        if prev is not None and ob["queued"] == 0 and np.abs(j[:6] - prev[:6]).max() < 0.02:
            return LK.Shot(ob["images"]["wrist"], ob["depth"]["wrist"], j, shot_id)
        prev = j
        time.sleep(0.12)
    return None


def to_table(frac, before_img, crop_box):
    """A point the language model gave (fractions of the top picture) -> the table (base m)."""
    a0, b0, sc = crop_box(before_img)
    uv = np.array([[a0[0] + frac[0] * (b0[0] - a0[0]), a0[1] + frac[1] * (b0[1] - a0[1])]]) / sc
    return pixel_to_base(load()["table"], uv, SURFACE_Z)[0][:2]


def go_look(c, xy, tip_mm, say):
    """The wrist camera over xy (base m), fingertips tip_mm up, every row checked. False if there's no safe way."""
    from lab import scan as SC
    from lab.scene3d import cameras
    _, T_fc, _ = cameras(None)
    from calibration.kinematics import DOWN, tool_ik
    from calibration.geometry import UF0_Z_MM
    cur, q = where(c)
    # the camera's offset from the fingertips with the fingers straight down and unturned (as the look will be): from
    # where the arm is now it would be turned with the wrist (after a pick), and the look would aim the wrong way
    ref = P.uf0(*np.asarray(xy), tip_mm)
    try:
        q_ref = tool_ik(np.array([ref[0], ref[1], ref[2] - UF0_Z_MM]) / 1000, DOWN, q[:6])
    except ValueError:
        q_ref = q[:6]
    tip_xy = np.asarray(xy) - SC.cam_offset(q_ref, T_fc)
    rows, why = SC.move_to(q[:6], P.uf0(*tip_xy, tip_mm), lifts=(max(tip_mm, 120.0), 200.0))
    if rows is None:
        say(f"  no safe way to look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm: {why}")
        return False
    if not SC.feed(c, rows, lambda: queued(c)):
        return False
    wait_still(c)
    return True


VIEW = {"rec": None}  # the Rerun stream this task logs its looks to (the page's viewer), if any


def show(shot, seen, near, title):
    """One look in the viewer: the still picture, every mask's outline (the one taken in green), what was measured."""
    rec = VIEW["rec"]
    if rec is None:
        return
    import rerun as rr
    rec.set_time("look", sequence=VIEW.setdefault("n", 0))
    VIEW["n"] += 1
    rec.log("look/picture", rr.Image(shot.colour))
    chosen = {id(s) for s in near}  # (by identity: comparing the records compares their arrays, which raises)
    others = [s.outline for s in seen if s.outline is not None and id(s) not in chosen]
    taken = [s.outline for s in near if s.outline is not None]
    rec.log("look/picture/others", rr.LineStrips2D([np.vstack([o, o[:1]]) for o in others], colors=[[255, 200, 0]] * len(others),
                                                   radii=1.5) if others else rr.Clear(recursive=False))
    rec.log("look/picture/taken", rr.LineStrips2D([np.vstack([o, o[:1]]) for o in taken], colors=[[40, 230, 90]] * len(taken),
                                                  radii=2.5, labels=[f"{s.word} {s.size_mm[0]:.0f}x{s.size_mm[1]:.0f} mm"
                                                                     f" {('%.0f mm tall' % s.top_mm) if s.top_mm else ''}"
                                                                     f" {'real' if s.real else 'flat?'}" for s in near])
            if taken else rr.Clear(recursive=False))
    rec.log("look/what", rr.TextDocument(f"{title}\n\n" + "\n".join(
        f"{'-> ' if id(s) in chosen else '   '}{s.word}: {s.size_mm[0]:.0f} x {s.size_mm[1]:.0f} mm, "
        f"top {s.top_mm} mm, {'real' if s.real else 'flat'} ({s.points} depth points)" for s in seen)))


def look_at(c, thing, xy, out, tell, say, k):
    """Look stations for one thing: over its point, a shot; small things a second, closer one. -> the Seen, or None."""
    words = list(dict.fromkeys([thing.get("word") or thing["name"]] + WORDS.get(thing.get("kind"), [])))
    seen_all = []
    for tip, tag in ((LOOK_TIP_MM, "look"), (CLOSE_TIP_MM, "close")):
        if tag == "close" and not (CLOSE_LOOK and thing.get("kind") == "small"):
            break
        if not go_look(c, xy, tip, say):
            break
        shot = still_shot(c, f"{k}_{tag}")
        if shot is None:
            say("  the arm never settled for a still shot")
            break
        cv2.imwrite(str(out / f"look_{k}_{tag}.jpg"), cv2.cvtColor(shot.colour, cv2.COLOR_RGB2BGR))
        seen = LK.measure(shot, words)
        near = [s for s in seen if np.linalg.norm(s.centre - xy) < NEAR_M]
        show(shot, seen, near, f"{tag} at the {thing['name']}")
        say(f"  {tag} at {thing['name']}: {len(seen)} masks, {len(near)} near the point: " +
            "; ".join(f"{s.word} {s.size_mm[0]:.0f}x{s.size_mm[1]:.0f} mm top {s.top_mm} real {s.real}" for s in near[:3]))
        if near:
            best = min(near, key=lambda s: np.linalg.norm(s.centre - xy))
            seen_all.append(best)
            xy = best.centre  # look closer at where it really is
    if not seen_all:
        return None
    s = seen_all[-1]  # the closest look's outline and position
    if MULTI_VIEW and len(seen_all) > 1:
        tops = [v.top_mm for v in seen_all if v.top_mm is not None]
        s.top_mm = float(np.median(tops)) if tops else s.top_mm
        s.real = any(v.real for v in seen_all)
    return s


def run_v2(c, t, out, before_img, look_pic, crop_box, say, tell, state, real=None, rerun_to=None) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from lab import vlm
    from lab.ladder import MODEL
    t0 = time.time()
    VIEW["rec"] = None
    try:  # the looks go to the page's viewer (and a file with the run)
        import rerun as rr
        import rerun.blueprint as rrb
        rec = rr.RecordingStream(f"look_{t.get('name', 'task')}")
        sinks = [rr.FileSink(str(out / "looks.rrd"))] + ([rr.GrpcSink(rerun_to)] if rerun_to else [])
        rec.set_sinks(*sinks, default_blueprint=rrb.Blueprint(
            rrb.Horizontal(rrb.Spatial2DView(origin="look/picture", name="what the wrist camera sees (still)"),
                           rrb.TextDocumentView(origin="look/what", name="what it measured"), column_shares=[3, 1]),
            rrb.BlueprintPanel(state="hidden"), rrb.SelectionPanel(state="hidden"), rrb.TimePanel(state="collapsed")))
        VIEW["rec"] = rec
    except Exception as e:
        say(f"  (no viewer for the looks: {e})")
    state("what", "reading the pictures and planning...")
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(vlm.ask_image, look_pic, PLAN.format(instruction=t["instruction"]), model=MODEL, think=True,
                        timeout=120)
        while not fut.done():
            time.sleep(1.0)
            state("what", f"planning from the first pictures ({time.time() - t0:.0f} s)")
        ans = fut.result()
    plan = (ans or {}).get("json") or {}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    if plan.get("say"):
        tell(plan["say"])
    things = {th["name"]: th for th in plan.get("things") or []}
    xy_of = {n: to_table(th["at"], before_img, crop_box) for n, th in things.items() if th.get("at")}
    for n, th in things.items():
        if n in xy_of:
            th["table_mm"] = [round(float(v) * 1000) for v in xy_of[n]]  # (saved with the plan, as plain numbers)
    moves_ = plan.get("moves") or []
    state("what", "plan: " + "; ".join(f"{m.get('thing')} -> {m.get('into') or m.get('to') or m.get('push_to')}"
                                       for m in moves_))
    seen: dict = {}
    done = []
    for k, m in enumerate(moves_):
        th = things.get(m.get("thing"))
        if th is None or th["name"] not in xy_of:
            tell(f"(The plan names {m.get('thing')!r} without a place: skipping it.)")
            continue
        state("arm", f"looking at the {th['name']}")
        s = look_at(c, th, xy_of[th["name"]], out, tell, say, k)
        if s is None:
            tell(f"I looked where the {th['name']} should be and didn't find it. Skipping it.")
            continue
        seen[th["name"]] = s
        if th.get("kind") == "small" and not s.real:
            tell(f"What looks like the {th['name']} is flat (a sticker or a picture?): not picking it up.")
            continue
        # where it goes
        drop = None
        if m.get("into"):
            dest = things.get(m["into"])
            if dest is None or dest["name"] not in xy_of:
                tell(f"(I don't know where the {m['into']} is: skipping.)")
                continue
            ds = seen.get(dest["name"]) or look_at(c, dest, xy_of[dest["name"]], out, tell, say, f"{k}d")
            if ds is not None:
                seen[dest["name"]] = ds
            place = ds.centre if ds is not None else xy_of[dest["name"]]
            rim = (ds.top_mm if ds is not None and ds.top_mm else 50.0)
            drop = max(rim, 30.0) + 10.0
        elif m.get("to"):
            place = to_table(m["to"], before_img, crop_box)
        elif m.get("push_to"):
            place = to_table(m["push_to"], before_img, crop_box)
        else:
            continue
        # how high to fly: over every thing seen so far, the target and the destination included
        tops = [v.top_mm for v in seen.values() if v.top_mm is not None] + [s.top_mm or 15.0, drop or 0.0]
        travel_mm = max(60.0, max(tops) + MV.GRIP_MM + CLEAR_MM + (s.top_mm or 15.0))
        cur, q = where(c)
        jaw = s.angle_deg + 90.0  # across the narrow side
        if m.get("push_to"):
            rows = MV.push(cur, s.centre, max(s.size_mm) / 2, place, min(max((s.top_mm or 30) * 0.4, 8.0), 20.0), travel_mm)
            how = "pushing"
        elif max(s.size_mm) < SMALL_MM:
            def clear(xy):
                return all(np.linalg.norm(v.centre - xy) > 0.015 for n, v in seen.items() if n != th["name"])
            rows, why = MV.push_grasp(cur, s.centre, jaw, max(s.size_mm), place, travel_mm, drop, clear)
            if not rows:
                tell(f"I can't push-grasp the {th['name']}: {why}.")
                continue
            how = "sliding in beside and grabbing"
        else:
            if min(s.size_mm) > JAW_ROOM_MM:
                tell(f"The {th['name']} is {min(s.size_mm):.0f} mm across its narrow side: too wide for the jaws.")
                continue
            rows = MV.pick_top(cur, s.centre, jaw, place, travel_mm, drop)
            how = "picking up"
        jrows, why = MV.joints(rows, q)
        if jrows is None:
            tell(f"I'm not moving the {th['name']}: {why}.")
            continue
        tell(f"{how.capitalize()} the {th['name']} ({s.size_mm[0]:.0f} x {s.size_mm[1]:.0f} mm, "
             f"{'%.0f mm tall' % s.top_mm if s.top_mm else 'height unknown'}), flying {travel_mm:.0f} mm up.")
        from lab import scan as SC
        state("arm", f"{how} the {th['name']}")
        if not SC.feed(c, jrows, lambda: queued(c), each=lambda qn: state("arm", f"{how} the {th['name']} ({qn / 10:.0f} s left)")):
            tell(f"I stopped part-way with the {th['name']}.")
            continue
        wait_still(c)
        done.append(m)
    return {"plan": plan, "moves_done": done, "seconds": round(time.time() - t0)}
