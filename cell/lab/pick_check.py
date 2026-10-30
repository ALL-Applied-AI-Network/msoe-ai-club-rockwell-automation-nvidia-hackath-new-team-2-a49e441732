"""Step 6: one pick and place (T1), every height solved from the measured shapes, no LLM.

    uv run python -m lab.depth_check --fast        (first: the scan and the map, left in the depth service)
    uv run python -m lab.pick_check [--object 5] [--into 4]

From the map in the depth service: the objects (lab/objects.py), then a solid one (default: the one nearest a
container) put into a container (default: the nearest) by lab/handling.put. The plan is shown in Rerun (the path, the
grip, the heights) and played on the honest Isaac cell, joints solved on one wrist branch. Then the sim's truth says
where the object ended up: inside the container's walls, below its rim = done. Writes runs/<stamp>_pick/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import requests
import rerun as rr

from calibration.geometry import SURFACE_Z, UF0_Z_MM
from calibration.kinematics import tool_ik
from lab import handling as HD
from lab import objects as OB
from lab import primitives as P
from lab.depth_bench import Poly, sim_objects
from lab.depth_live import DepthLink, tunnel
from lab.recorder import RUNS


def joints_for(tcp_rows, q):
    """Fingertip rows (UF0 mm, deg, grip) -> joint rows, each solved on the branch of the one before."""
    out = []
    for r in tcp_rows:
        tip = np.array([r[0], r[1], r[2] - UF0_Z_MM]) / 1000
        q = tool_ik(tip, r[3:6], q, near=True)
        out.append([*q.tolist(), float(r[6])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="sort_cubes")
    ap.add_argument("--object", type=int, default=None, help="found object id to move (default: nearest a container)")
    ap.add_argument("--into", type=int, default=None, help="found container id (default: nearest)")
    a = ap.parse_args()

    from lab.isaaccell import PORT
    from lab.isaaccell import tunnel as isaac_tunnel
    isaac_tunnel()
    tunnel()
    api = f"http://127.0.0.1:{PORT}/api/v1/"
    link = DepthLink()
    g, cols = link.grid(), link.columns()
    objs = OB.find(cols, g["xyz"], g["rgb"])
    solids = [o for o in objs if not o.container]
    boxes = [o for o in objs if o.container]
    if not solids or not boxes:
        raise SystemExit(f"need a solid object and a container in the map: {len(solids)} solid, {len(boxes)} containers")
    o = next(x for x in objs if x.id == a.object) if a.object else \
        min(solids, key=lambda s: min(np.linalg.norm(s.centre - b.centre) for b in boxes))
    dest = next(x for x in objs if x.id == a.into) if a.into else min(boxes, key=lambda b: np.linalg.norm(b.centre - o.centre))

    from cell_client import CellClient
    c = CellClient(f"http://127.0.0.1:{PORT}", token="sim")
    obs = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
    cur, q = obs["tcp_pose"], np.asarray(obs["joints"], float)[:6]
    plan = HD.put(o, dest, objs, cols, cur)
    print(f"move #{o.id} ({'solid'}, top {o.top_mm:.0f} mm, colour {o.rgb}) into #{dest.id} (container, rim "
          f"{dest.rim_mm:.0f} mm)")
    if not plan.ok:
        raise SystemExit(f"no plan: {plan.why}")
    for n in plan.notes:
        print("  " + n)
    tcp = HD.rows(plan, cur)
    jrows = joints_for(tcp, q)
    low = np.array(P.tcp_rows(np.asarray(jrows)))[:, 2] - P.TABLE_UF0
    print(f"  {len(jrows)} rows, {len(jrows) * P.DT:.0f} s; fingertips as low as {low.min():.1f} mm (at the grip)")

    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_pick"
    out.mkdir(parents=True)
    rr.init("pick")
    try:
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(out / "pick.rrd")))
        rr.spawn(connect=False)
    except Exception:
        rr.spawn()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/map", rr.Points3D(g["xyz"], colors=g["rgb"], radii=0.001), static=True)
    for x in objs:
        ring = np.vstack([x.footprint, x.footprint[:1]])
        col = [230, 150, 30] if x.container else ([230, 60, 50] if x.id == o.id else [40, 200, 90])
        rr.log(f"world/objects/{x.id}", rr.LineStrips3D([np.c_[ring, np.full(len(ring), SURFACE_Z + x.top_mm / 1000)]],
                                                       colors=[col], radii=0.0008, labels=[f"#{x.id} {x.top_mm:.0f} mm"]),
               static=True)
    path = np.array([[r[0] / 1000, r[1] / 1000, (r[2] - UF0_Z_MM) / 1000] for r in tcp])
    rr.log("world/plan/fingertips", rr.LineStrips3D([path], colors=[[60, 120, 230]], radii=0.0012), static=True)
    rr.log("plan", rr.TextDocument("\n".join(plan.notes)), static=True)

    truth = sim_objects(a.trial)  # before the move: which true object is ours, and which container
    tn = min((n for n in truth if not ("box" in n)), key=lambda n: np.linalg.norm(truth[n].vertices[:, :2].mean(0) - o.centre))
    bn = min((n for n in truth if "box" in n), key=lambda n: np.linalg.norm(truth[n].vertices[:, :2].mean(0) - dest.centre))
    print(f"  (in the sim that's {tn} into {bn})")
    for k in range(0, len(jrows), 100):
        c.act(jrows[k:k + 100], P.DT)
    t0 = time.time()
    while time.time() - t0 < 300:
        time.sleep(1.0)
        o2 = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
        if o2["queued"] == 0 and o2["status"] == "idle":
            break
    time.sleep(2.0)
    after = requests.get(api + "sim/truth", timeout=30).json()["poses"]
    oid = tn.split("#")[-1]
    p = np.asarray(after[oid][:3])
    box = truth[bn]
    inside = Poly(box.vertices[:, :2][np.argsort(np.arctan2(*(box.vertices[:, :2] - box.vertices[:, :2].mean(0)).T[::-1]))]
                  ).contains_points(p[None, :2])[0]
    below_rim = (p[2] - SURFACE_Z) * 1000 < (box.vertices[:, 2].max() - SURFACE_Z) * 1000
    done = bool(inside and below_rim)
    res = {"object": o.id, "true_object": tn, "into": dest.id, "true_container": bn, "done": done,
           "final_xyz_mm": np.round(p * 1000, 1).tolist(), "plan": plan.notes,
           "heights": {"grip": plan.grip_mm, "hang": plan.hang_mm, "carry": plan.carry_mm, "release": plan.release_mm,
                       "highest_on_path": plan.path_max_mm}}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{tn} ended at {np.round(p * 1000)} mm: {'IN ' + bn if done else 'not in ' + bn}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
