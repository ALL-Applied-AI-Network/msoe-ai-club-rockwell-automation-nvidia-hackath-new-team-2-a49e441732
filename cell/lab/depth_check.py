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
from pathlib import Path
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


STOPS_X = (0.30, 0.38, 0.46)                  # where the wrist camera stops (base m): ~80-90 mm apart, so each spot
STOPS_Y = (-0.27, -0.18, -0.09, 0.00, 0.07)   # on the table is seen from several (the depth service pairs pictures
CLUSTER = ((0.0, 0.0), (0.025, 0.0), (0.025, 0.025), (0.0, 0.025))  # at each stop, still pictures this far apart: the
# depth service matches patches between nearby views (stops 80+ mm apart alone gave 94% false matter, Oct 30)
STOPS = [(x + dx, y + dy) for i, y in enumerate(STOPS_Y)  # 15-120 mm apart)
         for x in (STOPS_X if i % 2 == 0 else STOPS_X[::-1]) for dx, dy in CLUSTER]
STILL_HOP = 200.0  # mm/s between stops


def still_pic(c, cam, timeout: float = 8.0):
    """A wrist picture once the arm has stopped: two readings in a row with the same joints and nothing queued
    (the real arm trails its queue by ~0.8 s, so the queue alone isn't enough). Returns (jpeg bytes, joints)."""
    import cv2
    prev = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        ob = c.observe(["wrist"], tuple(cam["size"]), "jpeg", 90)
        j = np.asarray(ob["joints"], float)
        if prev is not None and ob["queued"] == 0 and np.abs(j[:6] - prev[:6]).max() < 0.02:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(ob["images"]["wrist"], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            return buf.tobytes(), j
        prev = j
        time.sleep(0.12)
    return None, None


def stills_scan(c, link, q, tip_mm, off, T_fc, cam, hb) -> int:
    """The arm to each of STOPS (the camera's spot; fingertips at tip_mm), a still picture at each, sent to the depth
    service with its exact pose. Returns how many were taken."""
    from lab import scan
    n = 0
    p = None
    for k, xy in enumerate(STOPS):
        tip_xy = np.asarray(xy) - off
        goal = P.uf0(*tip_xy, tip_mm)
        if p is None:  # the first: a joint move there (lifted clear)
            Tf = tool_pose(q)
            qg = tool_ik(tip_m(goal), DOWN, q)
            jm = P.joint_move(q, Tf[:3, 3], wpr_of(Tf[:3, :3]), qg, tip_m(goal), DOWN, height_mm=max(tip_mm, 150.0))
            rows = [list(r) for r in jm] if jm is not None else []
            q = qg
        else:
            rows = []
            try:
                for row in P.line(p, goal, float(DOWN[2]), float(DOWN[2]), 0, STILL_HOP):
                    q = tool_ik(tip_m(row[:3]), row[3:6], q, near=True)
                    rows.append([*q.tolist(), 0.0])
            except ValueError:
                hb(f"stop {k + 1} of {len(STOPS)} is out of reach: skipped", {"queued": 0, "status": "idle"})
                continue
        if rows:
            scan.feed(c, rows, lambda: c._call("GET", "observation", params={"cameras": "wrist", "width": 64,
                                                                               "height": 36, "format": "jpeg"})["queued"])
        p = goal
        hb(f"stop {k + 1} of {len(STOPS)}: waiting for the arm to be still, then a picture",
           {"queued": len(rows), "status": "moving"})
        jpeg, j = still_pic(c, cam)
        if jpeg is None:
            hb(f"stop {k + 1}: the arm never settled, no picture", {"queued": 0, "status": "idle"})
            continue
        link.send(f"still{k}", time.time(), flange_pose(j[:6]) @ T_fc, jpeg)
        n += 1
    return n


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
    ap.add_argument("--record", default=None, help="save every wrist picture and the joint record to this folder")
    ap.add_argument("--angled", action="store_true",
                    help="after the sweep, orbit each thing the table camera sees: the camera on a sphere round it")
    ap.add_argument("--stills", action="store_true",
                    help="stop and shoot: still wrist pictures at a grid of stops instead of a moving sweep")
    ap.add_argument("--short-twitch", action="store_true", help="measure the picture delay with a 4 s wrist twitch")
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
        rows_, qg = scan.move_to(q_from, P.uf0(*tip_xy, h_mm), lifts=(max(h_mm, 150.0), 250.0))
        if rows_ is None:  # (qg: why) every row checked against the padded self-collision model
            raise SystemExit(f"no joint move to {np.round(np.asarray(tip_xy) * 1000)} mm from where the arm is: {qg}")
        return rows_, qg

    def arm_words(last):
        q = last.get("queued") or 0
        return f"moving ({q / 10:.0f} s of motion left)" if q else "still"

    def hb(where, last=None):  # a heartbeat for whoever watches (lab/ladder.py reads these lines): where + the arm
        if last is None:
            with pull.lock:
                last = dict(pull.last)
        print(f"HB\t{where}\t{arm_words(last)}", flush=True)

    def queued():
        with pull.lock:
            return pull.last.get("queued") or 0

    def play(rows_, where):
        scan.feed(c, rows_, queued)
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
        region = scan.region_with(first["depth"], first["joints"])
        if region != scan.REGION:
            print(f"scan region grown to cover what the table camera sees: x {region[0][0] * 1000:.0f}-"
                  f"{region[0][1] * 1000:.0f}, y {region[1][0] * 1000:.0f}-{region[1][1] * 1000:.0f} mm")
        segs = scan.for_camera(scan.lines(region), off)
        tall, _ = scan.tallest([(scan.blank_arm(first["depth"], first["joints"]), first["joints"])], region)
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
        if tall > scan.MAX_TALL_MM:  # the arm misread as a thing: don't stop, fly at the fallback height
            tip_mm = scan.FALLBACK_TIP_MM
            print(f"tallest thing on the table {tall:.0f} mm: that's the arm, not a thing; fingertips fly at "
                  f"{tip_mm:.0f} mm (the fallback)")
        else:
            tip_mm = scan.safe_tip_mm(tall)
        print(f"tallest thing on the table {tall:.0f} mm (2 pictures, arm moved between): fingertips fly at "
              f"{tip_mm:.0f} mm; the camera rides ({off[0] * 1000:.0f}, {off[1] * 1000:.0f}) mm from them")
        q_now = q2
    segs = [c_ for c_ in (scan.clip_reach(a_, b_) for a_, b_ in segs) if c_ is not None]  # inside the safe reach
    for k_ in range(len(segs)):  # the first line whose start the arm can reach from here
        try:
            go(q_now, segs[k_][0], tip_mm)
            segs = segs[k_:]
            break
        except SystemExit:
            print(f"line {k_ + 1}'s start is out of reach: starting from the next")
    start = P.uf0(*segs[0][0], tip_mm)
    if a.stills:  # stop and shoot: the arm still at each spot, so the pose of every picture is exact (no delay)
        pull.stop()
        fwd, t_wall = None, time.time()
        low, scan_part = tip_mm, []
        n_shot = stills_scan(c, link, q_now, tip_mm, off, T_fc, cam, hb)
        print(f"stills: {n_shot} still pictures from {len(STOPS)} stops in {time.time() - t_wall:.0f} s")
        t0_ = time.time()
        while time.time() - t0_ < 30:  # the depth service works through them
            st = link.stats()
            hb(f"stills taken: the GPU box is matching them ({st.get('inbox', '?')} waiting)",
               {"queued": 0, "status": "idle"})
            if st["inbox"] == 0 and st["waiting"] == 0 and link.q.empty():
                break
            time.sleep(1.0)
    else:
        r0, q0 = go(q_now, segs[0][0], tip_mm)
        rows += r0
        n_move = len(rows)
        if a.delay_ms is None:
            rows += twitch(q0, short=a.short_twitch)
        else:
            pull.delay = a.delay_ms / 1000
        n_twitch = len(rows)
        scan_part = scan.sweep(q0, tip_mm, speed, segs, start)
        if a.angled and first is not None:  # then orbit each hypothesis: the camera on a sphere round it, looking at it
            q_end = np.asarray(scan_part[-1][:6], float)
            extra = []
            hyps = scan.hypotheses(first["depth"], first["joints"], region)
            for k, (ctr, top) in enumerate(hyps):
                orb = scan.orbit(q_end, ctr, top, T_fc, min_tip_mm=max(top + 30.0, 60.0), speed=40.0)
                if not orb:
                    continue
                q_o = np.asarray(orb[0][:6], float)
                To = tool_pose(q_o)
                jm, _ = scan.move_to(q_end, np.array([*(To[:3, 3] * 1000)[:2], To[2, 3] * 1000 + UF0_Z_MM]),
                                     wpr_of(To[:3, :3]), lifts=(tip_mm, 150.0))
                if jm is None:
                    continue
                extra += [list(r) for r in jm] + orb
                q_end = np.asarray(orb[-1][:6], float)
                print(f"orbit {k + 1}: round the thing at ({ctr[0] * 1000:.0f}, {ctr[1] * 1000:.0f}) mm, "
                      f"{top:.0f} mm tall: {len(orb) * P.DT:.0f} s")
            if extra:
                scan_part = scan_part + extra + [extra[-1]] * 10
        rows += scan_part
        low = scan.lowest_tip_mm(scan_part)
        print(f"the scan: {len(scan_part) * P.DT:.0f} s, fingertips never lower than {low:.0f} mm (safe height {tip_mm:.0f})")
        with pull.lock:
            t_sent = pull.last.get("t", 0.0)
        import threading
        feed_err = []

        def feeding():
            try:
                scan.feed(c, rows, queued)
            except Exception as e:  # the cell refused (a fault, the trial over): the scan can't go on
                feed_err.append(f"{type(e).__name__}: {e}")
        threading.Thread(target=feeding, daemon=True).start()  # topped up as it plays
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
            if feed_err:
                raise SystemExit(f"the cell refused the scan's moves: {feed_err[0]}")
            if not moved and time.time() - t_wall > 20.0:
                raise SystemExit("the arm hasn't started moving in 20 s: is a trial running on the cell?")
            if fwd is None and moved and a.delay_ms is not None:
                print(f"picture delay {a.delay_ms:.0f} ms (known); feeding the depth service")
                fwd = Forward(pull, link, T_fc)
            if fwd is None and moved:  # the twitch done? measure the picture delay, then start feeding the depth service
                tw_idx = np.nonzero((np.abs(jq[:, 5] - q0[5]) > 0.5) & (jt >= t_sent + n_move * P.DT - 0.5))[0]
                tw_s = (n_twitch - n_move) * P.DT  # how long the twitch lasts
                if len(tw_idx) and jt[-1] - jt[tw_idx[0]] > tw_s + 1.0:
                    in_tw = tw_idx[jt[tw_idx] < jt[tw_idx[0]] + tw_s - 1.0]
                    tw = (jt[in_tw[0]] - 1.0, jt[in_tw[-1]] + 1.0)
                    pull.fit()
                    md = pull.measure_delay(*tw, T_fc, fx=cam["fx"])
                    print(f"picture delay {md['delay_s'] * 1000:.0f} ms (from the twitch, match {md['score']:.2f}); "
                          f"feeding the depth service")
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
    if a.record:  # every wrist picture and the joint record, for offline work (lab/timing_real.py, depth benches)
        rec = Path(a.record)
        (rec / "frames").mkdir(parents=True, exist_ok=True)
        with pull.lock:
            fr, jt_, jq_ = list(pull.frames), list(pull.jt), [np.asarray(q_) for q_ in pull.jq]
        lines_ = []
        for i, f in enumerate(fr):
            (rec / "frames" / f"{i:05d}.jpg").write_bytes(f.jpeg)
            lines_.append(json.dumps({"n": i, "t_first": f.t_first, "tc": f.tc, "k": f.k, "sure": f.sure, "hash": f.hash}))
        (rec / "frames.jsonl").write_text("\n".join(lines_))
        np.savez(rec / "joints.npz", t=np.array(jt_), q=np.array(jq_), delay=pull.delay if pull.delay is not None else np.nan,
                 T_fc=T_fc)
        print(f"recorded {len(fr)} wrist pictures and {len(jt_)} joint readings to {rec}")
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
