"""Step 3: live depth. The honest Isaac cell, the pullers, and the depth service on the GPU box, all at once.

    uv run python -m lab.depth_check [--no-reset] [--speed 60]

The arm goes over the table, twitches (the picture delay, lab/pull.py), then scans: straight lines both ways over
every object at SPEED mm/s. While it moves, every picture goes to the depth service (isaac/depth_server.py) with its
pose, and the grid comes back once a second into Rerun (live: it opens itself), next to the objects' true outlines.
At the end the grid is scored against the sim's objects (lab/depth_bench.py): measured vs true top of each object,
and voxels claiming matter where there is only air. Passes with every object seen, tops within 2 mm, little false
matter. Writes runs/<stamp>_depth/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import requests
import rerun as rr

from calibration.geometry import SURFACE_Z, UF0_Z_MM, flange_pose
from calibration.kinematics import DOWN, tool_ik, tool_pose, wpr_of
from lab import primitives as P
from lab.depth_bench import bench, sheet, sim_objects
from lab.depth_live import DepthLink, Forward
from lab.honest_check import HONEST
from lab.pull import Pull, twitch
from lab.recorder import RUNS
from lab.scene3d import cameras

SETTLE_S = 1.0  # the queue empty this long = the arm has stopped (it trails the queue by ~0.8 s on the real cell)
CAM_MM = 150  # fingertips above the table (camera ~280 mm up)
LINES = [((0.26, -0.30), (0.50, -0.30)), ((0.50, -0.20), (0.26, -0.20)), ((0.26, -0.10), (0.50, -0.10)),
         ((0.50, 0.00), (0.26, 0.00)), ((0.26, 0.08), (0.50, 0.08)), ((0.30, 0.08), (0.30, -0.30)),
         ((0.40, -0.30), (0.40, 0.08)), ((0.48, 0.08), (0.48, -0.30))]


def tip_m(p):
    return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="sort_cubes")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="mm/s along the scan lines (60; --fast: 150)")
    ap.add_argument("--first", default=None,
                    help="quick start: an .npz with the table depth picture (and joints, table/side pictures) taken at "
                         "second 0 by the caller: the safe height from it, then straight into the sweep")
    ap.add_argument("--delay-ms", type=float, default=None,
                    help="the picture delay, known already (no wrist twitch to measure it)")
    ap.add_argument("--fast", action="store_true",
                    help="step 4: the safe height from the table camera, lines over the whole table (lab/scan.py)")
    a = ap.parse_args()

    import os
    real = bool(os.getenv("CELL_URL"))  # the real cell (lab/ladder.py passes its address and token): no sim anything
    token, ca = (os.getenv("CELL_TOKEN", ""), os.getenv("CELL_CA") or None) if real else ("sim", None)
    if real:
        url = os.environ["CELL_URL"].rstrip("/")
        objects = {}
    else:
        from lab.isaaccell import PORT, IsaacCell, tunnel
        from lab.trial import build, load_trial
        tunnel()
        url = f"http://127.0.0.1:{PORT}"
        api = url + "/api/v1/"
        if not a.no_reset:
            IsaacCell(build(load_trial(a.trial), 0))
        objects = sim_objects(a.trial)
        requests.post(api + "sim/honest", json=HONEST, timeout=30)
    cam, T_fc, _ = cameras(None)
    link = DepthLink.start({"fx": cam["fx"], "k1": cam["k1"], "size": list(cam["size"])}, restart=a.first is None)

    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_depth"
    out.mkdir(parents=True)
    rr.init("depth_live")
    import os
    if os.getenv("RERUN_TO"):  # the app's viewer (lab/app.py serves one): log there, no window of its own
        rr.set_sinks(rr.GrpcSink(os.environ["RERUN_TO"]), rr.FileSink(str(out / "depth.rrd")))
    else:
        try:
            rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(out / "depth.rrd")))
            rr.spawn(connect=False)
        except Exception:
            rr.spawn()
    import rerun.blueprint as rrb
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(rrb.Spatial3DView(origin="world", name="what it knows: depth evidence, found objects"),
                       rrb.Vertical(rrb.Spatial2DView(origin="cams/wrist", name="wrist camera (live)"),
                                    rrb.Spatial2DView(origin="cams/table", name="table camera"),
                                    rrb.Spatial2DView(origin="cams/side", name="side camera")),
                       column_shares=[3, 1]),
        rrb.BlueprintPanel(state="hidden"), rrb.SelectionPanel(state="hidden"), rrb.TimePanel(state="collapsed")))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for name, m in objects.items():
        e = m.edges_unique
        rr.log(f"world/truth/{name.replace(' ', '_').replace('#', '')}",
               rr.LineStrips3D(m.vertices[e], colors=[[160, 160, 160]], radii=0.0003), static=True)
    rr.log("world/table", rr.Boxes3D(centers=[[0.42, -0.1, SURFACE_Z - 0.001]], half_sizes=[[0.2, 0.3, 0.001]],
                                     colors=[[235, 235, 235]]), static=True)

    from lab import scan
    from cell_client import CellClient
    c = CellClient(url, token, ca)
    first = np.load(a.first) if a.first else None
    try:
        fix = {k: first[k] for k in ("table", "side") if k in first} if first is not None else \
            c.observe(["table", "side"], (960, 540), "jpeg")["images"]
        for k, im in fix.items():
            rr.log(f"cams/{k}", rr.Image(im), static=True)
    except Exception as e:
        print(f"(no table/side pictures: {e})")
    speed = a.speed or (150.0 if a.fast else 60.0)
    if a.fast:
        tip_mm, segs = None, None  # the safe height comes from two table-depth pictures, below
    else:
        tip_mm, segs = CAM_MM, LINES
    pull = Pull(url, token, ca, inflight=3)
    for _ in range(50):  # the first joints heard
        with pull.lock:
            if pull.jq:
                break
        time.sleep(0.05)
    with pull.lock:
        q_now = np.asarray(pull.jq[-1], float)[:6]
    def go(q_from, tip_xy, h_mm):
        """Joint rows to the fingertips at tip_xy (m), h_mm up, straight down, lifted clear first."""
        Tf = tool_pose(q_from)
        goal = P.uf0(*tip_xy, h_mm)
        qg = tool_ik(tip_m(goal), DOWN, q_from)
        for lift in (max(h_mm, 150.0), 250.0):
            jm = P.joint_move(q_from, Tf[:3, 3], wpr_of(Tf[:3, :3]), qg, tip_m(goal), DOWN, height_mm=lift)
            if jm is not None:
                return [list(r) for r in jm], qg
        raise SystemExit(f"no joint move to {np.round(np.asarray(tip_xy) * 1000)} mm from where the arm is")

    def arm_words(last):
        q = last.get("queued") or 0
        return f"moving ({q / 10:.0f} s of motion left)" if q else "still"

    def hb(where, last=None):  # a heartbeat for whoever watches (lab/ladder.py reads these lines): where + the arm
        if last is None:
            with pull.lock:
                last = dict(pull.last)
        print(f"HB\t{where}\t{arm_words(last)}", flush=True)

    def play(rows_, where):
        for k in range(0, len(rows_), 100):
            c.act(rows_[k:k + 100], P.DT)
        zero = None
        while True:  # until it has played and the arm has settled (the real cell's status says "running" all trial
            time.sleep(0.5)  # long, so: nothing left in the queue for SETTLE_S)
            with pull.lock:
                last = dict(pull.last)
            hb(where, last)
            if last.get("queued") == 0:
                zero = zero or time.time()
                if time.time() - zero >= SETTLE_S:
                    return
            else:
                zero = None

    rows = []
    if first is not None:  # quick start: one table-depth picture (from the caller), straight into the sweep
        mid = P.uf0(0.36, -0.10, 250.0)
        off = scan.cam_offset(tool_ik(tip_m(mid), DOWN, q_now), T_fc)
        segs = scan.for_camera(scan.lines(), off)
        tall, _ = scan.tallest([(scan.blank_arm(first["depth"], first["joints"]), first["joints"])])
        if tall > scan.MAX_TALL_MM:
            tip_mm = scan.FALLBACK_TIP_MM
            print(f"tallest thing on the table {tall:.0f} mm: that's the arm, not a thing; fingertips fly at "
                  f"{tip_mm:.0f} mm (the fallback)")
        else:
            tip_mm = scan.safe_tip_mm(tall)
            print(f"tallest thing on the table {tall:.0f} mm (1 picture): fingertips fly at {tip_mm:.0f} mm")
        hb("safe height known: moving to the first sweep line")
    elif a.fast:  # two table-depth pictures, the arm in two places: how high it's safe to fly
        mid = P.uf0(0.36, -0.10, 250.0)
        off = scan.cam_offset(tool_ik(tip_m(mid), DOWN, q_now), T_fc)
        segs = scan.for_camera(scan.lines(), off)
        # the first picture from the corner of the table farthest from the scan's start, so the arm is somewhere
        # else in each (taken from wherever it happened to be, both could have it in the same place)
        (x0, x1), (y0, y1) = scan.REGION
        far = max(scan.for_camera([((x, y), (x, y)) for x in (x0, x1) for y in (y0, y1)], off),
                  key=lambda s: np.linalg.norm(np.asarray(s[0]) - segs[0][0]))[0]
        r1, q1 = go(q_now, far, 250.0)
        play(r1, "moving the arm aside for table-camera picture 1 of 2")
        pics = [scan.table_depth(c)]
        hb("table-camera picture 1 of 2 taken")
        r2, q2 = go(q1, segs[0][0], 250.0)
        play(r2, "moving the arm to the other side for picture 2 of 2")
        pics.append(scan.table_depth(c))
        hb("picture 2 of 2 taken: working out how tall things are")
        tall, info = scan.tallest(pics)
        if tall > scan.MAX_TALL_MM:
            raise SystemExit(f"the table camera says something is {tall:.0f} mm tall: the arm, not a thing on the "
                             f"table (its pixels weren't all blanked); not flying that high")
        tip_mm = scan.safe_tip_mm(tall)
        print(f"tallest thing on the table {tall:.0f} mm (2 pictures, arm moved between): fingertips fly at "
              f"{tip_mm:.0f} mm; the camera rides ({off[0] * 1000:.0f}, {off[1] * 1000:.0f}) mm from them")
        q_now = q2
    start = P.uf0(*segs[0][0], tip_mm)
    r0, q0 = go(q_now, segs[0][0], tip_mm)
    rows += r0
    n_move = len(rows)
    if a.delay_ms is None:
        rows += twitch(q0)
    else:
        pull.delay = a.delay_ms / 1000
    n_twitch = len(rows)
    scan_part = scan.sweep(q0, tip_mm, speed, segs, start)
    rows += scan_part
    low = scan.lowest_tip_mm(scan_part)
    print(f"the scan: {len(scan_part) * P.DT:.0f} s, fingertips never lower than {low:.0f} mm (safe height {tip_mm:.0f})")
    with pull.lock:
        t_sent = pull.last.get("t", 0.0)
    for k in range(0, len(rows), 100):
        c.act(rows[k:k + 100], P.DT)
    print(f"motion: {len(rows)} rows, {len(rows) * P.DT:.0f} s of arm time (twitch from {n_move * P.DT:.0f} s to "
          f"{n_twitch * P.DT:.0f} s, then the scan at {speed:.0f} mm/s)")

    fwd = None
    t_wall, moved, still_since, tw = time.time(), False, None, None
    while time.time() - t_wall < 900:
        time.sleep(1.0)
        with pull.lock:
            last = dict(pull.last)
            jt, jq = np.asarray(pull.jt), np.asarray(pull.jq)
        moved |= (last.get("queued") or 0) > 0
        if fwd is None and moved and a.delay_ms is not None:
            print(f"picture delay {a.delay_ms:.0f} ms (known); feeding the depth service")
            fwd = Forward(pull, link, T_fc)
        if fwd is None and moved:  # the twitch done? measure the picture delay, then start feeding the depth service
            tw_idx = np.nonzero((np.abs(jq[:, 5] - q0[5]) > 0.5) & (jt >= t_sent + n_move * P.DT - 0.5))[0]
            if len(tw_idx) and jt[-1] - jt[tw_idx[0]] > 12.0:
                in_tw = tw_idx[jt[tw_idx] < jt[tw_idx[0]] + 10.0]
                tw = (jt[in_tw[0]] - 1.0, jt[in_tw[-1]] + 1.0)
                pull.fit()
                md = pull.measure_delay(*tw, T_fc, fx=cam["fx"])
                print(f"picture delay {md['delay_s'] * 1000:.0f} ms (from the twitch); feeding the depth service")
                fwd = Forward(pull, link, T_fc)
        g = link.grid()
        hb((f"sweeping the table: {g.get('frames', 0)} wrist pictures into the 3D map, "
            f"{g.get('something', 0)} spots sure to hold something") if fwd is not None else
           "wiggling the wrist to time the camera", last)
        rr.set_time("wall_s", duration=time.time() - t_wall)
        with pull.lock:
            f_last = pull.frames[-1] if pull.frames else None
        if f_last is not None:
            rr.log("cams/wrist", rr.EncodedImage(contents=f_last.jpeg, media_type="image/jpeg"))
        if g.get("xyz"):
            rr.log("world/something", rr.Points3D(g["xyz"], colors=g["rgb"], radii=0.001))
        if "frames" in g:
          rr.log("stats", rr.TextDocument(
            f"frames {g['frames']} · reference frames {g['searched']}/{g['refs']} searched · "
            f"{g['points']} edge points · {g['something']} voxels something · {g['nothing']} nothing · "
            f"{g['ms_per_ref']} ms per reference frame"))
        with pull.lock:
            jn = pull.jq[-1][:6]
        rr.log("world/camera", rr.Transform3D(translation=(flange_pose(jn) @ T_fc)[:3, 3],
                                              mat3x3=(flange_pose(jn) @ T_fc)[:3, :3]))
        rr.log("world/camera", rr.Pinhole(focal_length=cam["fx"], width=cam["size"][0], height=cam["size"][1],
                                          image_plane_distance=0.05))
        if moved and last.get("queued") == 0:
            still_since = still_since or time.time()
            if time.time() - still_since > 2.0 and fwd is not None:
                if not pull.stop_ev.is_set():  # the arm is done: no more pictures (a still camera adds nothing, and
                    pull.stop()                # new frames would keep the depth service's queue from ever emptying)
                    fwd.flush()
                    fwd.stop()
                st = link.stats()
                if (st["inbox"] == 0 and st["waiting"] == 0 and link.q.empty()) or time.time() - still_since > 25.0:
                    break
        else:
            still_since = None
    if not pull.stop_ev.is_set():
        pull.stop()
    hb("sweep done: finding objects in the 3D map", {"queued": 0, "status": "idle"})
    if fwd and not fwd.stop_ev.is_set():
        fwd.stop()
    g = link.grid()
    xyz = np.asarray(g["xyz"]).reshape(-1, 3)
    res = bench(xyz, objects) if objects else {"seen": None, "of": 0, "median_abs_error_mm": None,
                                               "max_abs_error_mm": None, "false_matter": None, "voxels": len(xyz),
                                               "false_fraction": None, "objects": {}}
    res["stats"] = {k: g[k] for k in ("frames", "refs", "searched", "points", "something", "nothing", "ms_per_ref")}
    res["wall_s"] = round(time.time() - t_wall)
    res["scan"] = {"speed_mm_s": speed, "tip_mm": tip_mm, "lowest_tip_mm": round(low, 1), "scan_s": round(len(scan_part) * P.DT, 1)}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "grid.json").write_text(json.dumps(g))
    if objects:
        sheet(xyz, objects, str(out / "heights.jpg"))
    # step 5: rigid objects from the grid
    from lab import objects as OB
    cols = link.columns()
    found = OB.find(cols, xyz, g["rgb"])
    mt = OB.match(found, objects) if objects else {"missed": [], "merged": {}, "extra": [], "objects": {}}
    res["rigid"] = {"found": len(found), "missed": mt["missed"], "merged": mt["merged"], "extra": mt["extra"],
                    "objects": mt["objects"]}
    (out / "objects.json").write_text(json.dumps([{k: v.tolist() if isinstance(v, np.ndarray) else v
                                                   for k, v in o.__dict__.items() if k != "cells"}
                                                  for o in found], default=float, indent=1))
    for o in found:
        ring = np.vstack([o.footprint, o.footprint[:1]])
        z = SURFACE_Z + o.top_mm / 1000
        col = [[40, 200, 90] if not o.container else [230, 150, 30]]
        rr.log(f"world/objects/{o.id}", rr.LineStrips3D([np.c_[ring, np.full(len(ring), z)]], colors=col, radii=0.0008,
                                                         labels=[f"#{o.id} {'container' if o.container else 'solid'} "
                                                                 f"{o.top_mm:.0f} mm"]), static=True)
        rr.log(f"world/objects/{o.id}/base", rr.LineStrips3D([np.c_[ring, np.full(len(ring), SURFACE_Z)]], colors=col,
                                                              radii=0.0005), static=True)
    print(f"rigid objects: {len(found)} found; missed {mt['missed']}; merged {mt['merged']}; extra {mt['extra']}")
    for k, v in mt["objects"].items():
        print(f"  {k:22} -> #{v['found']}  top {v['top_mm']} (true {v['true_top_mm']})  "
              f"{'container' if v['container'] else 'solid'}  free inside {v['free_inside']}")
    for k, v in res["objects"].items():
        print(f"  {k:22} true top {v['true_top_mm']:5.1f} mm  measured {v['measured_top_mm']}  ({v['voxels']} voxels)")
    print(f"seen {res['seen']} of {res['of']}; tops off by median {res['median_abs_error_mm']} mm, worst "
          f"{res['max_abs_error_mm']} mm; false matter {res['false_matter']} of {res['voxels']} voxels "
          f"({res['false_fraction']}); {res['stats']['ms_per_ref']} ms per reference frame; {res['wall_s']} s")
    rr.log("summary", rr.TextDocument(json.dumps({k: v for k, v in res.items() if k != "objects"}, indent=1, default=float)),
           static=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
