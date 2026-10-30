"""Step 2: pull the wrist camera and joints from the honest Isaac cell, and work out when each picture was taken.

    uv run python -m lab.pull_check [--inflight 3] [--no-reset]

The arm goes over the table, twitches its wrist (lab.pull.twitch), then glides round the rectangle of lab.honest_check
while lab.pull.Pull keeps `inflight` wrist requests going plus a joints thread. Then Pull fits the camera's beat and the
fixed delay (from the twitch), and every picture gets a capture time. The server's record of every frame it took says
how right that is:
    timing error   estimated capture time - true capture time, against the naive one (the time of the answer the
                   picture came in, whose joints a policy would otherwise use)
    pixel error    the objects' true outlines drawn on each picture from the pose at the estimated time (green) and the
                   naive time (red), against the pose at the true time
Passes when capture times are within 10 ms and no frame is missed. Writes runs/<stamp>_pull/pull.rrd and opens it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import cv2
import numpy as np
import requests
import rerun as rr
import trimesh

from calibration.geometry import UF0_Z_MM, flange_pose
from calibration.kinematics import DOWN, tool_ik, tool_pose, wpr_of
from lab import primitives as P
from lab.edges3d import Lens
from lab.honest_check import CAM_MM, HONEST, LEGS, STOP
from lab.pull import Pull, twitch
from lab.recorder import RUNS
from lab.scene3d import cameras


def motion(q_now):
    """Joint rows: over the table, the twitch, the rectangle. And the joints where the twitch starts."""
    def tip_m(p):
        return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000
    q_now = np.asarray(q_now, float)[:6]
    start = P.uf0(*LEGS[0][0], CAM_MM)
    q0 = tool_ik(tip_m(start), DOWN, q_now)
    T = tool_pose(q_now)
    rows = [list(r) for r in P.joint_move(q_now, T[:3, 3], wpr_of(T[:3, :3]), q0, tip_m(start), DOWN, height_mm=CAM_MM)]
    rows += twitch(q0)
    p, q, r = start, q0, float(DOWN[2])
    for xy, speed in LEGS[1:]:
        goal = P.uf0(*xy, CAM_MM)
        for row in P.line(p, goal, r, r, 0, speed) + P.hold(goal, r, 0, STOP):
            q = tool_ik(tip_m(row[:3]), row[3:6], q, near=True)
            rows.append([*q.tolist(), 0.0])
        p = goal
    return rows, q0


def outlines(lens, T_cam, objects) -> list[np.ndarray]:
    """Each object's silhouette on the picture (convex hull of its projected vertices)."""
    out = []
    for V in objects:
        uv, z = lens.project(T_cam, V)
        if (z > 0.02).all():
            out.append(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inflight", type=int, default=3)
    ap.add_argument("--trial", default="sort_cubes")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()

    from lab.isaaccell import PORT, IsaacCell, tunnel
    from lab.trial import build, load_trial
    tunnel()
    url = f"http://127.0.0.1:{PORT}"
    api = url + "/api/v1/"
    sim = build(load_trial(a.trial), 0)
    if not a.no_reset:
        IsaacCell(sim)
    truth = requests.get(api + "sim/truth", timeout=30).json()["poses"]
    objects = []
    for i in sim.instances:
        v = truth.get(str(i.id))
        if v is None:
            continue
        T = trimesh.transformations.quaternion_matrix(v[3:7])
        T[:3, 3] = v[:3]
        objects.append(trimesh.transform_points(sim.local[i.id].convex_hull.vertices, T))
    requests.post(api + "sim/honest", json=HONEST, timeout=30)
    cam, T_fc, _ = cameras(None)
    lens = Lens(cam)

    pull = Pull(url, "sim", inflight=a.inflight)
    time.sleep(1.0)
    with pull.lock:
        q_now = pull.jq[-1]
    rows, q0 = motion(q_now)
    from cell_client import CellClient
    c = CellClient(url, token="sim")
    for k in range(0, len(rows), 100):
        c.act(rows[k:k + 100], P.DT)
    print(f"motion: {len(rows)} rows, {len(rows) * P.DT:.0f} s of arm time; pulling with {a.inflight} in flight")
    t_wall, moved, still_since = time.time(), False, None
    while time.time() - t_wall < 300:
        time.sleep(0.3)
        with pull.lock:
            last = dict(pull.last)
        moved |= (last.get("queued") or 0) > 0
        if moved and last.get("queued") == 0 and last.get("status") == "idle":
            still_since = still_since or last["t"]
            if last["t"] - still_since > 1.5:
                break
        else:
            still_since = None
    pull.stop()
    print(f"{pull.answers} answers ({pull.errors} failed), {len(pull.frames)} distinct pictures, "
          f"wall {time.time() - t_wall:.0f} s")

    jt, jq = np.asarray(pull.jt), np.asarray(pull.jq)
    twisting = np.nonzero(np.abs(jq[:, 5] - q0[5]) > 0.5)[0]
    in_twitch = twisting[jt[twisting] < jt[twisting[0]] + 8.0]  # the twitch, not the rectangle after it
    t0, t1 = jt[in_twitch[0]] - 1.0, jt[in_twitch[-1]] + 1.0
    pull.fit()
    ts = dict(pull.thumb_stats); dd = ts.pop("d", [])
    print("thumbnails:", ts, "unclear (best, second) e.g.", dd[:8])
    md_tw = pull.measure_delay(t0, t1, T_fc, fx=cam["fx"])
    print(f"delay from the twitch alone {md_tw['delay_s'] * 1000:.0f} ms, plateau {md_tw['plateau_ms']}")
    md = pull.measure_delay(jt[0], jt[-1], T_fc, fx=cam["fx"])  # every start and stop of the run, not just the twitch
    print(f"delay from the whole run {md['delay_s'] * 1000:.0f} ms, plateau {md['plateau_ms']}")
    cap = requests.get(api + "sim/captures", timeout=30).json()
    true_t = {x["id"]: x["t"] for x in cap["captures"]}
    fr = [f for f in sorted(pull.frames, key=lambda f: f.k) if f.truth.get("frame") in true_t]
    tt = np.array([true_t[f.truth["frame"]] for f in fr])
    est = np.array([f.tc for f in fr])
    naive = np.array([f.t_first for f in fr])
    err, nerr = (est - tt) * 1000, (naive - tt) * 1000
    ke = np.array([f.k for f in fr]) - (np.array([f.truth["frame"] for f in fr]) - fr[0].truth["frame"])
    vals, cnt = np.unique(ke - np.bincount(ke - ke.min()).argmax() - ke.min(), return_counts=True)
    print(f"frame counts: {dict(zip(vals.tolist(), cnt.tolist()))} (offset from the true count; 0 = right)")
    gk = np.array([pull.grid(f) for f in fr])
    dtrue = (gk - tt) * 1000
    print(f"true delay (grid - true capture): median {np.median(dtrue):.0f} ms, spread 10-90% "
          f"{np.percentile(dtrue, 10):.0f}..{np.percentile(dtrue, 90):.0f}")
    print("twitch plateau (ms):", md["plateau_ms"])
    for D, sc in zip(md["Ds"][::10], md["scores"][::10]):
        print(f"   D {D * 1000:5.0f} ms  fit {sc:+.2f}")
    window = [x for x in cap["captures"] if fr and tt[0] <= x["t"] <= tt[-1]]
    missed = len(window) - len({f.truth["frame"] for f in fr})
    print(f"camera beat {pull.period * 1000:.2f} ms (true 100), delay from the twitch {md['delay_s'] * 1000:.0f} ms "
          f"(fit {md['score']:.2f} over {md['frames']} frames)")
    print(f"capture time error, estimated: median {np.median(np.abs(err)):.1f} ms, 90% {np.percentile(np.abs(err), 90):.1f}, "
          f"max {np.abs(err).max():.1f};  naive: median {np.median(nerr):.0f} ms late")
    print(f"frames: {len(fr)} seen of {len(window)} taken ({missed} missed)")

    # pixel error: the objects' outlines from the pose at the estimated / naive time against the true time
    def centres(ts):
        cs = []
        for T in [flange_pose(q) @ T_fc for q in pull.joints_at(ts)]:
            cs.append(np.array([lens.project(T, V.mean(0, keepdims=True))[0][0] for V in objects]))
        return np.array(cs)
    ct, ce, cn = centres(tt), centres(est), centres(naive)
    pe, pn = np.linalg.norm(ce - ct, axis=2).mean(1), np.linalg.norm(cn - ct, axis=2).mean(1)
    print(f"object position on the picture, off by: estimated median {np.median(pe):.2f} px (max {pe.max():.1f}), "
          f"naive median {np.median(pn):.1f} px (max {pn.max():.1f})")

    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_pull"
    out.mkdir(parents=True)
    res = {"inflight": a.inflight, "answers": pull.answers, "pictures": len(pull.frames), "frames_taken": len(window),
           "missed": missed, "period_ms": pull.period * 1000, "delay_ms": md["delay_s"] * 1000,
           "err_ms": {"median": float(np.median(np.abs(err))), "p90": float(np.percentile(np.abs(err), 90)),
                      "max": float(np.abs(err).max())}, "naive_ms": float(np.median(nerr)),
           "px": {"est_median": float(np.median(pe)), "est_max": float(pe.max()), "naive_median": float(np.median(pn)),
                  "naive_max": float(pn.max())}}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    (out / "frames.json").write_text(json.dumps({
        "frames": [{"t_first": f.t_first, "k": f.k, "truth": f.truth.get("frame"), "true_t": true_t.get(f.truth.get("frame")),
                    "seen": f.seen, "sure": f.sure} for f in sorted(pull.frames, key=lambda f: f.t_first)],
        "captures": cap["captures"], "jt": pull.jt, "jq": [q.tolist() for q in pull.jq], "twitch": [t0, t1]}))
    import pickle
    (out / "jpegs.pkl").write_bytes(pickle.dumps({f.hash: f.jpeg for f in pull.frames}))
    (out / "hashes.json").write_text(json.dumps([f.hash for f in sorted(pull.frames, key=lambda f: f.t_first)]))

    rr.init("pull")
    rr.log("timing/error_ms/naive", rr.SeriesPoints(names="naive: time of the answer", colors=[[220, 60, 50]], marker_sizes=3), static=True)
    rr.log("timing/error_ms/estimated", rr.SeriesPoints(names="estimated capture time", colors=[[40, 170, 90]], marker_sizes=3), static=True)
    rr.log("timing/px/naive", rr.SeriesLines(names="naive", colors=[[220, 60, 50]]), static=True)
    rr.log("timing/px/estimated", rr.SeriesLines(names="estimated", colors=[[40, 170, 90]]), static=True)
    for f, t, e, n, a_px, n_px in zip(fr, tt, err, nerr, pe, pn):
        rr.set_time("cell_s", duration=t)
        rr.log("timing/error_ms/naive", rr.Scalars(n))
        rr.log("timing/error_ms/estimated", rr.Scalars(e))
        rr.log("timing/px/naive", rr.Scalars(n_px))
        rr.log("timing/px/estimated", rr.Scalars(a_px))
        rr.log("wrist/image", rr.EncodedImage(contents=f.jpeg, media_type="image/jpeg"))
        Tn = flange_pose(pull.joints_at(f.t_first)[0]) @ T_fc
        Te = flange_pose(pull.joints_at(f.tc)[0]) @ T_fc
        rr.log("wrist/naive", rr.LineStrips2D([np.vstack([o, o[:1]]) for o in outlines(lens, Tn, objects)],
                                              colors=[[230, 60, 50]], radii=1.2))
        rr.log("wrist/estimated", rr.LineStrips2D([np.vstack([o, o[:1]]) for o in outlines(lens, Te, objects)],
                                                  colors=[[40, 200, 90]], radii=1.2))
    for D, s in zip(md["Ds"], md["scores"]):
        rr.set_time("candidate_delay_ms", sequence=int(round(D * 1000)))
        rr.log("delay_fit", rr.Scalars(s))
    txt = (f"### Step 2: when was each picture taken?\n\n"
           f"| | naive (answer's time) | estimated |\n|---|---|---|\n"
           f"| capture time off by (median) | {np.median(nerr):.0f} ms late | {np.median(np.abs(err)):.1f} ms |\n"
           f"| 90% within | {np.percentile(np.abs(nerr), 90):.0f} ms | {np.percentile(np.abs(err), 90):.1f} ms |\n"
           f"| objects on the picture off by (median) | {np.median(pn):.1f} px | {np.median(pe):.2f} px |\n"
           f"| worst frame | {pn.max():.1f} px | {pe.max():.1f} px |\n\n"
           f"Camera beat {pull.period * 1000:.2f} ms (true 100). Fixed delay from the twitch {md['delay_s'] * 1000:.0f} ms. "
           f"{a.inflight} wrist requests in flight + a joints thread: {len(fr)} of {len(window)} frames seen.\n\n"
           f"Green outlines use the estimated capture time, red the naive one. Scrub the timeline during the fast legs.")
    rr.log("summary", rr.TextDocument(txt, media_type=rr.MediaType.MARKDOWN), static=True)
    import rerun.blueprint as rrb
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial2DView(origin="wrist", name="wrist: true outlines, green = estimated time, red = naive"),
        rrb.Vertical(rrb.TimeSeriesView(origin="timing/error_ms", name="capture time error (ms)"),
                     rrb.TimeSeriesView(origin="timing/px", name="objects off on the picture (px)"),
                     rrb.TextDocumentView(origin="summary", name="summary")), column_shares=[3, 2])))
    rr.save(str(out / "pull.rrd"))

    # a contact sheet for a quick look: the six frames where the naive pose is worst
    sheet = []
    for i in np.argsort(-pn)[:6]:
        f = fr[i]
        img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
        for ts, col in ((f.t_first, (50, 60, 230)), (f.tc, (90, 200, 40))):
            T = flange_pose(pull.joints_at(ts)[0]) @ T_fc
            cv2.polylines(img, [o.astype(np.int32) for o in outlines(lens, T, objects)], True, col, 2)
        cv2.putText(img, f"naive {pn[i]:.0f} px, estimated {pe[i]:.1f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        sheet.append(cv2.resize(img, (424, 240)))
    if sheet:
        while len(sheet) % 3:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(str(out / "sheet.jpg"), np.vstack([np.hstack(sheet[i:i + 3]) for i in range(0, len(sheet), 3)]))
    print(f"saved {out}")
    if not a.no_view:
        subprocess.Popen(["rerun", str(out / "pull.rrd")])


if __name__ == "__main__":
    main()
