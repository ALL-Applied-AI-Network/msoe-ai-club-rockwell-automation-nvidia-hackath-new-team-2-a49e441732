"""Step 1: does the Isaac cell, in honest mode, hand out wrist pictures the way the real cell does?

    uv run python -m lab.honest_check [--inflight 1] [--trial sort_cubes]

The arm glides over the table at 30-150 mm/s with short stops while this asks for wrist pictures, `inflight` requests
at a time, as fast as answers come back. The server records every frame the camera took (its time and the joints then);
each answer carries, for this check only, which frame it was. From that:
    repeats   an answer with the same picture as the answer before it (asked before the camera had a new frame)
    missed    frames the camera took that no answer ever carried
    age       how old the picture was when the joints in the same answer were read
Oct 28 on the real cell, one request at a time: 755 answers in 84 s, 99 ms apart (1 in 10 over 167 ms), 10% repeats,
19% of the camera's frames missed. Writes runs/<stamp>_honest_n<k>/honest.rrd and opens it in Rerun.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import requests
import rerun as rr

from cell_client import CellClient
from lab import primitives as P
from lab.recorder import RUNS

REAL = {"spacing_ms": 99, "spacing_p90_ms": 167, "repeats": 0.10, "missed": 0.19}  # Oct 28, one request at a time
HONEST = {"rtt_ms": 48, "rtt_sigma": 0.45, "pipe_ms": 40, "pipe_jitter_ms": 30, "fps": 10}
CAM_MM = 150  # fingertips this high above the table: the camera ~280 mm up
LEGS = [((0.33, -0.10), 150), ((0.48, -0.10), 50), ((0.48, 0.10), 100), ((0.33, 0.10), 30), ((0.33, -0.10), 150)]
STOP = 10  # rows (1 s) still after each leg


def plan(q_now) -> list:
    """Joint rows: a joint move from where the arm is to over the table (a straight fingertip line from home needs the
    wrist to flip, which the cell refuses), then around a rectangle, a different speed each side, each fingertip row
    solved on the same branch as the one before."""
    from calibration.geometry import UF0_Z_MM
    from calibration.kinematics import DOWN, tool_ik, tool_pose, wpr_of
    q_now = np.asarray(q_now, float)[:6]
    def tip_m(p):  # UF0 mm -> base frame m
        return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000
    start = P.uf0(*LEGS[0][0], CAM_MM)
    q0 = tool_ik(tip_m(start), DOWN, q_now)
    T = tool_pose(q_now)
    rows = [list(r) for r in P.joint_move(q_now, T[:3, 3], wpr_of(T[:3, :3]), q0, tip_m(start), DOWN, height_mm=CAM_MM)]
    p, q = start, q0
    r = float(DOWN[2])
    for xy, speed in LEGS[1:]:
        goal = P.uf0(*xy, CAM_MM)
        for row in P.line(p, goal, r, r, 0, speed) + P.hold(goal, r, 0, STOP):
            q = tool_ik(tip_m(row[:3]), row[3:6], q, near=True)
            rows.append([*q.tolist(), 0.0])
        p = goal
    return rows


def poll(url: str, out: list, stop: threading.Event, lock: threading.Lock) -> None:
    c = CellClient(url, token="sim")
    params = {"cameras": "wrist", "width": 848, "height": 480, "format": "jpeg", "quality": 90}
    while not stop.is_set():
        asked = time.time()
        try:
            o = c._call("GET", "observation", params=params)
        except Exception as e:  # keep asking: one failed request must not end the stream
            print(f"request failed: {e}", flush=True)
            time.sleep(0.1)
            continue
        b64 = o["images"].get("wrist")
        rec = {"asked": asked, "got": time.time(), "t": o["t"], "queued": o["queued"], "status": o["status"],
               "joints": o["joints"], "hash": hashlib.md5(b64.encode()).hexdigest() if b64 else None,
               "jpeg": base64.b64decode(b64) if b64 else None, "truth": o.get("_truth", {})}
        with lock:
            out.append(rec)


def analyse(answers: list, caps: list, pipe_ms: float) -> dict:
    a = sorted((x for x in answers if x["hash"]), key=lambda x: x["t"])
    t = np.array([x["t"] for x in a])
    rep = sum(1 for i in range(1, len(a)) if a[i]["hash"] == a[i - 1]["hash"])
    seen = {x["truth"]["frame"] for x in a}
    window = [c for c in caps if c["t"] >= t[0] - 0.25 and c["t"] + (pipe_ms + 100) / 1000 <= t[-1]]
    missed = [c for c in window if c["id"] not in seen]
    age = np.array([(x["t"] - x["truth"]["t_capture"]) * 1000 for x in a])
    gaps = np.diff(t) * 1000
    span = t[-1] - t[0]
    return {"answers": len(a), "span_s": round(span, 1), "spacing_ms": round(float(np.median(gaps))),
            "spacing_p90_ms": round(float(np.percentile(gaps, 90))), "repeats": rep / max(1, len(a) - 1),
            "missed": len(missed) / max(1, len(window)), "frames_taken": len(window), "frames_seen": len(window) - len(missed),
            "age_ms": [round(float(np.percentile(age, q))) for q in (10, 50, 90)], "missed_ids": [c["id"] for c in missed]}


def log_rerun(path: Path, answers: list, caps: list, res: dict, inflight: int) -> None:
    rr.init("honest_sim")
    rr.log("timing/frame/camera", rr.SeriesLines(names="newest frame the camera took", widths=1.5), static=True)
    rr.log("timing/frame/answer", rr.SeriesPoints(names="frame an answer carried", marker_sizes=3), static=True)
    rr.log("timing/frame/repeat", rr.SeriesPoints(names="repeat (same picture again)", colors=[[230, 120, 40]],
                                                  marker_sizes=5), static=True)
    rr.log("timing/frame/missed", rr.SeriesPoints(names="taken, never sent", colors=[[200, 40, 40]], markers="cross",
                                                  marker_sizes=5), static=True)
    rr.log("timing/age_ms", rr.SeriesPoints(names="picture age when the joints were read (ms)", marker_sizes=3), static=True)
    missed = set(res["missed_ids"])
    for c in caps:
        rr.set_time("sim_s", duration=c["t"])
        rr.log("timing/frame/camera", rr.Scalars(c["id"]))
        if c["id"] in missed:
            rr.log("timing/frame/missed", rr.Scalars(c["id"]))
    prev = None
    for x in sorted((x for x in answers if x["hash"]), key=lambda x: x["t"]):
        rr.set_time("sim_s", duration=x["t"])
        f = x["truth"]["frame"]
        rr.log("timing/frame/answer", rr.Scalars(f))
        if prev == x["hash"]:
            rr.log("timing/frame/repeat", rr.Scalars(f))
        rr.log("timing/age_ms", rr.Scalars((x["t"] - x["truth"]["t_capture"]) * 1000))
        rr.log("wrist", rr.EncodedImage(contents=x["jpeg"], media_type="image/jpeg"))
        prev = x["hash"]
    sim = (f"| | real cell, Oct 28 | honest sim, {inflight} in flight |\n|---|---|---|\n"
           f"| answers apart (median) | {REAL['spacing_ms']} ms | {res['spacing_ms']} ms |\n"
           f"| 1 in 10 over | {REAL['spacing_p90_ms']} ms | {res['spacing_p90_ms']} ms |\n"
           f"| repeated pictures | {REAL['repeats']:.0%} | {res['repeats']:.0%} |\n"
           f"| camera frames missed | {REAL['missed']:.0%} | {res['missed']:.0%} |\n"
           f"| picture age, 10/50/90% | ~100-140 ms (estimated) | {' / '.join(map(str, res['age_ms']))} ms |\n")
    rr.log("summary", rr.TextDocument(sim, media_type=rr.MediaType.MARKDOWN), static=True)
    try:
        import rerun.blueprint as rrb
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(rrb.TimeSeriesView(origin="timing/frame", name="frame numbers: taken vs carried"),
                         rrb.TimeSeriesView(origin="timing/age_ms", name="picture age (ms)")),
            rrb.Vertical(rrb.Spatial2DView(origin="wrist", name="wrist pictures as they arrived"),
                         rrb.TextDocumentView(origin="summary", name="real vs sim")))))
    except Exception as e:  # the layout is a nicety
        print(f"(no blueprint: {e})")
    rr.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--trial", default="sort_cubes")
    ap.add_argument("--no-reset", action="store_true", help="keep the objects already in Isaac")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--rtt-ms", type=float, default=HONEST["rtt_ms"],
                    help="the simulated round trip; the link to the GPU box adds its own on top")
    a = ap.parse_args()

    from lab.isaaccell import PORT, tunnel
    tunnel()
    url = f"http://127.0.0.1:{PORT}"
    api = url + "/api/v1/"
    if not a.no_reset:
        from lab.isaaccell import IsaacCell
        from lab.trial import build, load_trial
        IsaacCell(build(load_trial(a.trial), 0))
    print("honest mode:", requests.post(api + "sim/honest", json={**HONEST, "rtt_ms": a.rtt_ms}, timeout=30).json())
    c = CellClient(url, token="sim")
    o = c._call("GET", "observation", params={"cameras": "wrist", "width": 64, "height": 36, "format": "jpeg"})
    rows = plan(o["joints"])
    print(f"motion: {len(rows)} rows, {len(rows) * P.DT:.0f} s of arm time")

    answers, lock, stop = [], threading.Lock(), threading.Event()
    threads = [threading.Thread(target=poll, args=(url, answers, stop, lock), daemon=True) for _ in range(a.inflight)]
    for th in threads:
        th.start()
    time.sleep(1.0)
    for k in range(0, len(rows), 100):
        c.act(rows[k:k + 100], P.DT)
    t_wall = time.time()
    while True:  # until the queue is empty and the arm has been still for a second of sim time
        time.sleep(0.5)
        with lock:
            last = answers[-1] if answers else None
            moved = next((i for i, x in enumerate(answers) if x["queued"] > 0), None)
            idle = [] if moved is None else [x for x in answers[moved:] if x["queued"] == 0 and x["status"] == "idle"]
        if last and idle and last["t"] - idle[0]["t"] > 1.0 and time.time() - t_wall > 5:
            break
        if time.time() - t_wall > 300:
            print("gave up waiting for the arm")
            break
    stop.set()
    for th in threads:
        th.join(timeout=5)
    cap = requests.get(api + "sim/captures", timeout=30).json()
    print(f"wall {time.time() - t_wall:.0f} s for {cap['clock']:.0f} s on the sim clock")
    res = analyse(answers, cap["captures"], HONEST["pipe_ms"])

    out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_honest_n{a.inflight}"
    out.mkdir(parents=True)
    (out / "answers.json").write_text(json.dumps([{k: v for k, v in x.items() if k != "jpeg"} for x in answers]))
    (out / "captures.json").write_text(json.dumps(cap))
    (out / "result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "missed_ids"}, indent=1))
    print(f"{'':28}{'real Oct 28':>14}{'honest sim':>14}")
    print(f"{'answers apart (median)':28}{REAL['spacing_ms']:>11} ms{res['spacing_ms']:>11} ms")
    print(f"{'1 in 10 over':28}{REAL['spacing_p90_ms']:>11} ms{res['spacing_p90_ms']:>11} ms")
    print(f"{'repeated pictures':28}{REAL['repeats']:>14.0%}{res['repeats']:>14.0%}")
    print(f"{'camera frames missed':28}{REAL['missed']:>14.0%}{res['missed']:>14.0%}")
    print(f"picture age 10/50/90%: {res['age_ms']} ms;  {res['answers']} answers, {res['frames_seen']} of "
          f"{res['frames_taken']} frames seen")
    log_rerun(out / "honest.rrd", answers, cap["captures"], res, a.inflight)
    print(f"saved {out / 'honest.rrd'}")
    if not a.no_view:
        subprocess.Popen(["rerun", str(out / "honest.rrd")])


if __name__ == "__main__":
    main()
