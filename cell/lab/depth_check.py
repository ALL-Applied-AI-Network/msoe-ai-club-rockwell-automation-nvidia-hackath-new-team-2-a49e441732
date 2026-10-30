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

CAM_MM = 150  # fingertips above the table (camera ~280 mm up)
LINES = [((0.26, -0.30), (0.50, -0.30)), ((0.50, -0.20), (0.26, -0.20)), ((0.26, -0.10), (0.50, -0.10)),
         ((0.50, 0.00), (0.26, 0.00)), ((0.26, 0.08), (0.50, 0.08)), ((0.30, 0.08), (0.30, -0.30)),
         ((0.40, -0.30), (0.40, 0.08)), ((0.48, 0.08), (0.48, -0.30))]


def tip_m(p):
    return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000


def scan_rows(q, start, speed: float, hop: float = 120.0) -> list:
    """Joint rows along LINES from `start` (UF0 mm) at joints q: hop to each line's start, glide along it."""
    rows, p, r = [], start, float(DOWN[2])
    for a, b in LINES:
        for goal, v in ((P.uf0(*a, CAM_MM), hop), (P.uf0(*b, CAM_MM), speed)):
            seg = []
            try:
                for row in P.line(p, goal, r, r, 0, v):
                    q = tool_ik(tip_m(row[:3]), row[3:6], q, near=True)
                    seg.append([*q.tolist(), 0.0])
            except ValueError:
                print(f"skipped a leg to {np.round(goal[:2])} mm: out of reach on this wrist branch")
                break
            rows += seg
            p = goal
    return rows + [rows[-1]] * 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="sort_cubes")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--speed", type=float, default=60.0, help="mm/s along the scan lines")
    a = ap.parse_args()

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
    link = DepthLink.start({"fx": cam["fx"], "k1": cam["k1"], "size": list(cam["size"])})

    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_depth"
    out.mkdir(parents=True)
    rr.init("depth_live")
    try:
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(out / "depth.rrd")))
        rr.spawn(connect=False)
    except Exception:
        rr.spawn()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for name, m in objects.items():
        e = m.edges_unique
        rr.log(f"world/truth/{name.replace(' ', '_').replace('#', '')}",
               rr.LineStrips3D(m.vertices[e], colors=[[160, 160, 160]], radii=0.0003), static=True)
    rr.log("world/table", rr.Boxes3D(centers=[[0.42, -0.1, SURFACE_Z - 0.001]], half_sizes=[[0.2, 0.3, 0.001]],
                                     colors=[[235, 235, 235]]), static=True)

    pull = Pull(url, "sim", inflight=3)
    time.sleep(1.0)
    with pull.lock:
        q_now = np.asarray(pull.jq[-1], float)[:6]
    start = P.uf0(0.26, -0.30, CAM_MM)
    q0 = tool_ik(tip_m(start), DOWN, q_now)
    T = tool_pose(q_now)
    rows = [list(r) for r in P.joint_move(q_now, T[:3, 3], wpr_of(T[:3, :3]), q0, tip_m(start), DOWN, height_mm=CAM_MM)]
    n_move = len(rows)
    rows += twitch(q0)
    n_twitch = len(rows)
    rows += scan_rows(q0, start, a.speed)
    from cell_client import CellClient
    c = CellClient(url, token="sim")
    for k in range(0, len(rows), 100):
        c.act(rows[k:k + 100], P.DT)
    print(f"motion: {len(rows)} rows, {len(rows) * P.DT:.0f} s of arm time (twitch from {n_move * P.DT:.0f} s to "
          f"{n_twitch * P.DT:.0f} s, then the scan at {a.speed:.0f} mm/s)")

    fwd = None
    t_wall, moved, still_since, tw = time.time(), False, None, None
    while time.time() - t_wall < 900:
        time.sleep(1.0)
        with pull.lock:
            last = dict(pull.last)
            jt, jq = np.asarray(pull.jt), np.asarray(pull.jq)
        moved |= (last.get("queued") or 0) > 0
        if fwd is None and moved:  # the twitch done? measure the picture delay, then start feeding the depth service
            tw_idx = np.nonzero(np.abs(jq[:, 5] - q0[5]) > 0.5)[0]
            if len(tw_idx) and jt[-1] - jt[tw_idx[0]] > 12.0:
                in_tw = tw_idx[jt[tw_idx] < jt[tw_idx[0]] + 10.0]
                tw = (jt[in_tw[0]] - 1.0, jt[in_tw[-1]] + 1.0)
                pull.fit()
                md = pull.measure_delay(*tw, T_fc, fx=cam["fx"])
                print(f"picture delay {md['delay_s'] * 1000:.0f} ms (from the twitch); feeding the depth service")
                fwd = Forward(pull, link, T_fc)
        g = link.grid()
        rr.set_time("wall_s", duration=time.time() - t_wall)
        if g["xyz"]:
            rr.log("world/something", rr.Points3D(g["xyz"], colors=g["rgb"], radii=0.001))
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
        if moved and last.get("queued") == 0 and last.get("status") == "idle":
            still_since = still_since or time.time()
            if time.time() - still_since > 2.0 and fwd is not None:
                fwd.flush()
                st = link.stats()
                if st["inbox"] == 0 and st["waiting"] == 0 and link.q.empty():
                    break
        else:
            still_since = None
    pull.stop()
    if fwd:
        fwd.stop()
    g = link.grid()
    xyz = np.asarray(g["xyz"]).reshape(-1, 3)
    res = bench(xyz, objects)
    res["stats"] = {k: g[k] for k in ("frames", "refs", "searched", "points", "something", "nothing", "ms_per_ref")}
    res["wall_s"] = round(time.time() - t_wall)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "grid.json").write_text(json.dumps(g))
    sheet(xyz, objects, str(out / "heights.jpg"))
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
