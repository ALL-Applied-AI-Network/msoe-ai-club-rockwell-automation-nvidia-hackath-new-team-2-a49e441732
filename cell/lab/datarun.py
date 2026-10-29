"""A data run: a trial spent on camera sweeps instead of a sort, recording everything, for multi-view depth offline.

    uv run python -m lab.datarun --sim                  rehearse it on the Isaac cell (no booking)
    uv run python -m lab.datarun --real                 the real cell: our approved request, its next ready trial
    uv run python -m lab.datarun --real --centre 430,40 --cam-mm 250

The arm never goes near the table: the camera stays CAM_MM above it (fingertips ~CAM_MM - 118 mm, over any cup), and
nothing is picked. While it moves, a second thread asks for wrist pictures as fast as the cell gives them (wrist only,
native size, no depth: ~13 a second on the real cell); at every stop it also takes one full set (three cameras, both
depths). Each picture is saved with the joints, the fingertip pose and the cell's clock from the same answer, and when
we asked and got it back.

The plan, around --centre (where the parts are):
    1. home: one full set
    2. cross: stop-and-go, 11 spots 10 mm apart along x, then 10 along y (the edge sweep's input)
    3. glides: the same lines without stopping, at 50 and then 100 mm/s (do the pictures and the joints agree while it
       moves? the timing test)
    4. orbit: 8 views tilted 20 deg from 300 mm, all aimed at the centre (side walls; wide angles for structure
       from motion)
    5. home: one full set (did anything move?)

runs/<time>_datarun_<real|sim>/
    frames/NNNNN.jpg + frames.jsonl   every wrist picture: phase, cell time t, asked/got (wall clock), joints, tcp
    stops/NNN_*.jpg, NNN.npz           the full sets: table/side/wrist pictures, table/wrist depth, joints
    plan.json                          the targets, the speeds, the camera height
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

from calibration.geometry import SURFACE_Z, load
from calibration.kinematics import tool_ik, tool_pose, wpr_of
from cell_client import CellClient
from lab import armcheck, primitives as P
from lab.views import _cam_in_tool, _look_at

RUNS = Path(__file__).resolve().parents[1] / "runs"
FRAME_SIZE = (848, 480)   # the wrist camera's own size: anything bigger is upscaled by the cell
MIN_TIP_MM = 95.0         # the fingertips never come lower than this above the table
ORBIT_MM = 300.0          # the tilted views stand further off: tilting brings the fingertips down (250 mm: 79 mm up)


class Recorder(threading.Thread):
    """Wrist pictures as fast as the cell answers, until stopped; the phase label is whatever the plan says now."""

    def __init__(self, client: CellClient, out: Path):
        super().__init__(daemon=True)
        self.client, self.out = client, out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self.log = open(out / "frames.jsonl", "a", buffering=1)
        self.phase, self.n, self.on, self.errors = "start", 0, True, 0

    def run(self) -> None:
        while self.on:
            t0 = time.time()
            try:
                o = self.client.observe(["wrist"], FRAME_SIZE, "jpeg", quality=95)
            except Exception:
                self.errors += 1
                time.sleep(0.2)
                continue
            t1 = time.time()
            cv2.imwrite(str(self.out / "frames" / f"{self.n:05d}.jpg"), cv2.cvtColor(o["images"]["wrist"], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true", help="the Isaac cell (the 4090 box)")
    g.add_argument("--real", action="store_true", help="the real cell, with our approved request")
    ap.add_argument("--centre", default="430,40", help="where the parts are, x,y mm in the base frame")
    ap.add_argument("--cam-mm", type=float, default=250.0, help="camera above the table (mm)")
    ap.add_argument("--trial", default="sort_cups15", help="--sim: the layout")
    a = ap.parse_args()
    centre = [float(v) / 1000 for v in a.centre.split(",")]
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_datarun_{'real' if a.real else 'sim'}"
    out.mkdir(parents=True)

    if a.real:
        from lab.realcell import RealCell
        from lab.robot import _approved_token, SPEEDS
        url, token = _approved_token("", None)
        cell = RealCell(url, token)
        print("starting trial:", cell.start_trial()["trial_id"], flush=True)
        client = CellClient(url, token)
        P.TRAVEL, P.APPROACH = SPEEDS["real"]
    else:
        from lab.isaaccell import PORT, IsaacCell
        from lab.trial import build, load_trial
        cell = IsaacCell(build(load_trial(a.trial), 0))
        client = CellClient(f"http://127.0.0.1:{PORT}", token="sim")
    cams = load()
    T_ct = _cam_in_tool(cams["wrist"])
    targets = plan(centre, a.cam_mm)
    (out / "plan.json").write_text(json.dumps({"centre_m": centre, "cam_mm": a.cam_mm, "frame_size": FRAME_SIZE,
                                               "targets": [(p, T.tolist(), s) for p, T, s in targets]}))
    rec = Recorder(client, out)
    full_set(client, out, 0, "home")
    obs = client.observe(["wrist"], (64, 36), "jpeg")
    q_home = np.asarray(obs["joints"][:6], float)
    rec.start()
    t0, k, travel = time.time(), 1, P.TRAVEL
    try:
        for phase, T_cam, speed in targets:
            Tt = T_cam @ np.linalg.inv(T_ct)
            tip, wpr = Tt[:3, 3], wpr_of(Tt[:3, :3])
            if (tip[2] - SURFACE_Z) * 1000 < MIN_TIP_MM or not P.reachable(*tip[:2], margin_mm=15):
                print(f"  skip {phase}: fingertips would be {(tip[2] - SURFACE_Z) * 1000:.0f} mm up, or out of reach", flush=True)
                continue
            q_now = np.asarray(client.observe(["wrist"], (64, 36), "jpeg")["joints"][:6], float)
            try:
                q = tool_ik(tip, wpr, q_now)
            except ValueError:
                print(f"  skip {phase}: no joint solution", flush=True)
                continue
            T_now = tool_pose(q_now)
            P.TRAVEL = speed or travel
            rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q, tip, wpr, height_mm=MIN_TIP_MM,
                                settle=0 if speed else 5)
            P.TRAVEL = travel
            if rows is None or not armcheck.check_joints(np.asarray(rows)[:, :6], 0.0).ok:
                print(f"  skip {phase}: the move fails the arm check", flush=True)
                continue
            rec.phase = phase
            P.send(cell, rows)
            if speed is None:
                full_set(client, out, k, phase)
                k += 1
            print(f"  {phase}: camera at ({T_cam[0, 3] * 1000:.0f}, {T_cam[1, 3] * 1000:.0f}, "
                  f"{(T_cam[2, 3] - SURFACE_Z) * 1000:.0f}) mm; {rec.n} pictures so far", flush=True)
        rec.phase = "home"
        q_now = np.asarray(client.observe(["wrist"], (64, 36), "jpeg")["joints"][:6], float)
        T_now, T_home = tool_pose(q_now), tool_pose(q_home)
        rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q_home, T_home[:3, 3], wpr_of(T_home[:3, :3]),
                            height_mm=MIN_TIP_MM)
        if rows is not None:
            P.send(cell, rows)
        full_set(client, out, k, "home again")
    finally:
        rec.stop()
    dt = time.time() - t0
    print(f"done: {rec.n} wrist pictures in {dt:.0f} s ({rec.n / dt:.1f} a second), {k + 1} full sets, "
          f"{rec.errors} failed requests -> {out}", flush=True)


if __name__ == "__main__":
    main()
