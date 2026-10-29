"""Data runs: a trial spent on camera sweeps instead of a sort, recording everything; and a speed test of the arm.

From the app (the Data run and Speed test buttons, or the chat: "do a data run", "how fast can you go?"), on whichever
cell it's connected to; or alone:
    uv run python -m lab.datarun --sim [--speed]        the Isaac cell (no booking)
    uv run python -m lab.datarun --real [--speed]       the real cell: our approved request, its next ready trial

Data run. The camera stays CAM_MM above the table (fingertips ~CAM_MM - 118 mm, over any cup) and nothing is picked.
While it moves, a second thread asks for wrist pictures as fast as the cell gives them (wrist only, native size, no
depth: ~13 a second on the real cell); at every stop it also takes one full set (three cameras, both depths). Each
picture is saved with the joints, the fingertip pose and the cell's clock from the same answer, and when we asked and
got it back. The plan, around the centre (where the parts are):
    1. home: one full set
    2. cross: stop-and-go, 11 spots 10 mm apart along x, then 10 along y (the edge sweep's input)
    3. glides: the same lines without stopping, at 50 and then 100 mm/s (do the pictures and the joints agree while it
       moves? the timing test)
    4. orbit: 8 views tilted 20 deg from 300 mm, all aimed at the centre (side walls; wide angles for structure from
       motion; the ones out of reach are skipped)
    5. home: one full set (did anything move?)

Speed test. Out and back along a 240 mm line across the table, fingertips SPEED_TIP_MM up, commanded at each of
SPEEDS_MM_S; a poller records the fingertip as fast as the cell answers. For each speed: the peak speed it really
reached, how long the move took against the command, how late it started, and how long it took to settle.

runs/<time>_datarun_<real|sim>/  or  runs/<time>_speedtest_<real|sim>/
    frames/NNNNN.jpg + frames.jsonl   every wrist picture: phase, cell time t, asked/got (wall clock), joints, tcp
    stops/NNN_*.jpg, NNN.npz           the full sets: table/side/wrist pictures, table/wrist depth, joints
    track.jsonl, speed.json            the speed test: the fingertip over time, and the results per speed
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from calibration.geometry import SURFACE_Z, load
from calibration.kinematics import tool_ik, tool_pose, wpr_of
from cell_client import CellClient
from lab import armcheck, primitives as P
from lab.views import _cam_in_tool, _look_at

RUNS = Path(__file__).resolve().parents[1] / "runs"
FRAME_SIZE = (848, 480)   # the wrist camera's own size: anything bigger is upscaled by the cell
MIN_TIP_MM = 95.0         # the fingertips never come lower than this above the table
ORBIT_MM = 300.0          # the tilted views stand further off: tilting brings the fingertips down (250 mm: 79 mm up)
SPEED_TIP_MM = 150.0      # the speed test's height: clear of everything
SPEED_LINE = ((0.40, -0.12), (0.40, 0.12))  # base frame m: across the middle of the table, well inside reach
SPEEDS_MM_S = (100.0, 200.0, 300.0, 400.0, 500.0)


def fork(client: CellClient) -> CellClient:
    """A second client on the same cell (its own connection: one per thread)."""
    c = CellClient.__new__(CellClient)
    c.api, c.s = client.api, requests.Session()
    c.s.headers.update(client.s.headers)
    c.s.verify = client.s.verify
    return c


class Recorder(threading.Thread):
    """Pictures (or, small=True, just the joints) as fast as the cell answers, until stopped."""

    def __init__(self, client: CellClient, out: Path, small: bool = False):
        super().__init__(daemon=True)
        self.client, self.out, self.small = client, out, small
        if not small:
            (out / "frames").mkdir(parents=True, exist_ok=True)
        self.log = open(out / ("track.jsonl" if small else "frames.jsonl"), "a", buffering=1)
        self.phase, self.n, self.on, self.errors = "start", 0, True, 0

    def run(self) -> None:
        while self.on:
            t0 = time.time()
            try:
                o = self.client.observe(["wrist"], (64, 36) if self.small else FRAME_SIZE, "jpeg", quality=95)
            except Exception:
                self.errors += 1
                time.sleep(0.2)
                continue
            t1 = time.time()
            if not self.small:
                cv2.imwrite(str(self.out / "frames" / f"{self.n:05d}.jpg"),
                            cv2.cvtColor(o["images"]["wrist"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.log.write(json.dumps({"n": self.n, "phase": self.phase, "t": float(o["t"]), "asked": t0, "got": t1,
                                       "joints": np.asarray(o["joints"], float).tolist(),
                                       "tcp": None if o.get("tcp_pose") is None else list(map(float, o["tcp_pose"])),
                                       "queued": o.get("queued")}) + "\n")
            self.n += 1

    def stop(self) -> None:
        self.on = False
        self.join(timeout=5)
        self.log.close()


def full_set(client: CellClient, out: Path, k: int, label: str) -> None:
    o = client.observe(["table", "side", "wrist"], (1280, 720), "jpeg", quality=95, depth=["table", "wrist"])
    d = out / "stops"
    d.mkdir(exist_ok=True)
    for c, img in o["images"].items():
        cv2.imwrite(str(d / f"{k:03d}_{c}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(d / f"{k:03d}.npz", joints=np.asarray(o["joints"], float), t=float(o["t"]), label=label,
                        **{f"depth_{c}": v for c, v in (o.get("depth") or {}).items()})


def plan(centre, cam_mm: float, step_mm: float = 10.0):
    """(phase, camera pose, speed mm/s or None for stop-and-go) for each target."""
    c = np.array([centre[0], centre[1], SURFACE_Z])
    down = np.array([0, 0, -1.0])
    out = []
    for k in range(11):
        out.append(("cross_x", _look_at(c + [(k - 5) * step_mm / 1000, 0, 0], down, cam_mm / 1000, 0.0), None))
    for k in range(10):
        out.append(("cross_y", _look_at(c + [0, (k - 4.5) * step_mm / 1000, 0], down, cam_mm / 1000, 0.0), None))
    for speed in (50.0, 100.0):
        for dx, dy in [(-0.06, 0), (0.06, 0), (0, -0.06), (0, 0.06)]:
            out.append((f"glide_{speed:.0f}", _look_at(c + [dx, dy, 0], down, cam_mm / 1000, 0.0), speed))
    tilt = np.radians(20)
    for k in range(8):
        a = 2 * np.pi * k / 8
        d = np.array([-np.sin(tilt) * np.cos(a), -np.sin(tilt) * np.sin(a), -np.cos(tilt)])
        out.append(("orbit", _look_at(c, d, ORBIT_MM / 1000, 0.0), None))
    return out


def _now(client: CellClient) -> np.ndarray:
    return np.asarray(client.observe(["wrist"], (64, 36), "jpeg")["joints"][:6], float)


def _move(cell, client, tip, wpr, speed=None, settle: int = 5, joint_deg_s: float | None = None) -> str | None:
    """A joint move to the fingertip pose (tip m, wpr deg), at speed mm/s (None: the usual); None if done, else why not."""
    if (tip[2] - SURFACE_Z) * 1000 < MIN_TIP_MM - 1e-6 or not P.reachable(*tip[:2], margin_mm=15):
        return f"fingertips would be {(tip[2] - SURFACE_Z) * 1000:.0f} mm up, or out of reach"
    q_now = _now(client)
    try:
        q = tool_ik(tip, wpr, q_now)
    except ValueError:
        return "no joint solution"
    T_now = tool_pose(q_now)
    travel, step = P.TRAVEL, P.JOINT_STEP_DEG
    try:
        P.TRAVEL = speed or travel
        if joint_deg_s:
            P.JOINT_STEP_DEG = joint_deg_s * P.DT
        rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q, tip, wpr, height_mm=MIN_TIP_MM, settle=settle)
    finally:
        P.TRAVEL, P.JOINT_STEP_DEG = travel, step
    if rows is None or not armcheck.check_joints(np.asarray(rows)[:, :6], 0.0).ok:
        return "the move fails the arm check"
    P.send(cell, rows)
    return None


def run(cell, client: CellClient, centre, cam_mm: float = 250.0, say=print, out: Path | None = None,
        stopped=lambda: False) -> Path:
    """The data run on this cell (the trial is already running). Returns the folder it recorded into."""
    out = out or RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_datarun_{'real' if hasattr(cell, 'check') else 'sim'}"
    out.mkdir(parents=True, exist_ok=True)
    T_ct = _cam_in_tool(load()["wrist"])
    targets = plan(centre, cam_mm)
    (out / "plan.json").write_text(json.dumps({"centre_m": list(centre), "cam_mm": cam_mm, "frame_size": FRAME_SIZE,
                                               "targets": [(p, T.tolist(), s) for p, T, s in targets]}))
    rec = Recorder(fork(client), out)
    full_set(client, out, 0, "home")
    q_home = _now(client)
    rec.start()
    t0, k, phase_said = time.time(), 1, None
    try:
        for phase, T_cam, speed in targets:
            if stopped():
                say("Data run stopped.")
                break
            Tt = T_cam @ np.linalg.inv(T_ct)
            rec.phase = phase
            why = _move(cell, client, Tt[:3, 3], wpr_of(Tt[:3, :3]), speed, settle=0 if speed else 5)
            if why:
                continue
            if speed is None:
                full_set(client, out, k, phase)
                k += 1
            if phase != phase_said:
                say(f"Data run: {phase.replace('_', ' ')} ({rec.n} pictures so far).")
                phase_said = phase
        rec.phase = "home"
        T_home = tool_pose(q_home)
        _move(cell, client, T_home[:3, 3], wpr_of(T_home[:3, :3]), settle=5)
        full_set(client, out, k, "home again")
    finally:
        rec.stop()
    dt = time.time() - t0
    say(f"Data run done: {rec.n} wrist pictures in {dt:.0f} s ({rec.n / dt:.1f} a second), {k + 1} full sets, "
        f"{rec.errors} failed requests. Saved in {out.name}.")
    return out


def speed_test(cell, client: CellClient, say=print, out: Path | None = None, stopped=lambda: False) -> dict:
    """Out and back along SPEED_LINE at each speed; the fingertip tracked as fast as the cell answers."""
    out = out or RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_speedtest_{'real' if hasattr(cell, 'check') else 'sim'}"
    out.mkdir(parents=True, exist_ok=True)
    z = SURFACE_Z + SPEED_TIP_MM / 1000
    a, b = (np.array([*p, z]) for p in SPEED_LINE)
    wpr = np.array([-180.0, 0.0, 0.0])
    q_home = _now(client)
    if _move(cell, client, a, wpr) is not None:
        say("Speed test: can't reach the start of the line; not run.")
        return {}
    rec = Recorder(fork(client), out, small=True)
    rec.start()
    marks = []
    try:
        for v in SPEEDS_MM_S:
            if stopped():
                break
            for name, goal in ((f"{v:.0f} out", b), (f"{v:.0f} back", a)):
                rec.phase = name
                t_send = time.time()
                why = _move(cell, client, goal, wpr, speed=v, settle=8, joint_deg_s=230.0)  # J1's own limit
                marks.append({"phase": name, "speed": v, "sent": t_send, "back": time.time(), "skipped": why})
            say(f"Speed test: {v:.0f} mm/s done.")
        rec.phase = "home"
        T_home = tool_pose(q_home)
        _move(cell, client, T_home[:3, 3], wpr_of(T_home[:3, :3]))
    finally:
        rec.stop()
    res = analyse(out, marks, np.linalg.norm(b - a) * 1000)
    (out / "speed.json").write_text(json.dumps({"marks": marks, "results": res, "line_mm": np.linalg.norm(b - a) * 1000}))
    lines = [f"{r['speed']:.0f} mm/s asked: reached {r['peak']:.0f} mm/s, the 240 mm took {r['took']:.2f} s "
             f"(asked {r['asked']:.2f} s), started {r['late']:.2f} s late, settled {r['settle']:.2f} s after"
             for r in res if r.get("peak") is not None]
    say("Speed test done:\n" + "\n".join(lines) if lines else "Speed test: no moves were recorded.")
    return {"results": res, "folder": out.name}


def analyse(out: Path, marks: list[dict], line_mm: float) -> list[dict]:
    """Per speed, from the 'out' leg: peak speed (over 0.2 s), duration from first motion to arrival, start lag, settle."""
    rows = [json.loads(l) for l in open(out / "track.jsonl")]
    res = []
    for m in marks:
        if not m["phase"].endswith("out") or m["skipped"]:
            continue
        seg = [r for r in rows if r["phase"] == m["phase"] and r.get("tcp")]
        if len(seg) < 4:
            res.append({"speed": m["speed"], "peak": None})
            continue
        t = np.array([r["got"] for r in seg])
        p = np.array([r["tcp"][:3] for r in seg])
        d = np.linalg.norm(p - p[0], axis=1)
        moving = np.nonzero(d > 1.0)[0]
        arrived = np.nonzero(np.abs(d - line_mm) < 1.0)[0]
        if not len(moving) or not len(arrived):
            res.append({"speed": m["speed"], "peak": None})
            continue
        i0, i1 = moving[0], arrived[0]
        v = []
        for i in range(len(t)):
            j = np.searchsorted(t, t[i] + 0.2)
            if j < len(t):
                v.append(np.linalg.norm(p[j] - p[i]) / (t[j] - t[i]))
        still = np.nonzero(np.r_[np.linalg.norm(np.diff(p, axis=0), axis=1), 0] < 0.2)[0]
        settled = next((t[i] for i in still if i > i1), t[i1])
        res.append({"speed": m["speed"], "peak": float(max(v)) if v else None, "took": float(t[i1] - t[i0]),
                    "asked": line_mm / m["speed"], "late": float(t[i0] - m["sent"]), "settle": float(settled - t[i1]),
                    "samples": len(seg), "sample_hz": len(seg) / max(t[-1] - t[0], 1e-6)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true", help="the Isaac cell (the 4090 box)")
    g.add_argument("--real", action="store_true", help="the real cell, with our approved request")
    ap.add_argument("--speed", action="store_true", help="the speed test instead of the data run")
    ap.add_argument("--centre", default="430,40", help="where the parts are, x,y mm in the base frame")
    ap.add_argument("--cam-mm", type=float, default=250.0, help="camera above the table (mm)")
    ap.add_argument("--trial", default="sort_cups15", help="--sim: the layout")
    a = ap.parse_args()
    if a.real:
        from lab.realcell import RealCell
        from lab.robot import _approved_token, SPEEDS
        url, token = _approved_token("", None)
        cell = RealCell(url, token)
        print("starting trial:", cell.start_trial()["trial_id"], flush=True)
        P.TRAVEL, P.APPROACH = SPEEDS["real"]
    else:
        from lab.isaaccell import IsaacCell
        from lab.trial import build, load_trial
        cell = IsaacCell(build(load_trial(a.trial), 0))
    if a.speed:
        speed_test(cell, cell.client)
    else:
        run(cell, cell.client, [float(v) / 1000 for v in a.centre.split(",")], a.cam_mm)


if __name__ == "__main__":
    main()
