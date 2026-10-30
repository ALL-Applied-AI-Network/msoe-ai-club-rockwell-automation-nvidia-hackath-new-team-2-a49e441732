"""The wrist camera's calibration, refined from a real video sweep: no board, no stills.

    uv run python -m lab.calib_video runs/<time>_datarun_real [--times runs/.../frame_times_s0.json]

The poses come from the joints (metric, exact up to the camera's mounting) at each picture's moment (solved per
picture by lab/frame_times.py, or first appearance - delay). Features are tracked through runs of WINDOW pictures;
each track is triangulated from all the pictures that saw it and reprojected into them. The leftover reprojection
error, in pixels, is how far the calibration is from the truth. Seven numbers are fitted to minimise it: the focal
length (a scale on fx), and small corrections to the camera's mounting on the flange (3 rotations, deg; 3
translations, mm). Writes <run>/calib_video.json: the corrections, the error before and after, and the corrected
T_flange_cam and fx (not applied anywhere: calibration/cameras.json is left as it is).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration.geometry import flange_pose
from lab.scene3d import cameras
from lab.timing_real import joints_at, load

WINDOW = 8       # pictures per tracking window
MIN_SEEN = 4     # a track must be seen in at least this many
MAX_TRACKS = 2500


def track_windows(frames):
    """Multi-picture tracks: [(picture indices, (n, 2) pixels)]."""
    grey = {}
    def G(i):
        if i not in grey:
            grey[i] = cv2.cvtColor(cv2.imread(str(frames[i]["path"])), cv2.COLOR_BGR2GRAY)
        return grey[i]
    out = []
    for s in range(0, len(frames) - WINDOW, WINDOW // 2):
        idx = list(range(s, s + WINDOW))
        if len({frames[i]["phase"] for i in idx}) > 1 or frames[idx[-1]]["t"] - frames[idx[0]]["t"] > WINDOW * 0.2:
            continue
        p0 = cv2.goodFeaturesToTrack(G(idx[0]), 150, 0.01, 10)
        if p0 is None:
            continue
        pts = [p0[:, 0]]
        alive = np.ones(len(p0), bool)
        for a_, b_ in zip(idx[:-1], idx[1:]):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(G(a_), G(b_), pts[-1].reshape(-1, 1, 2).astype(np.float32), None,
                                                 winSize=(21, 21), maxLevel=3)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(G(b_), G(a_), p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb[:, 0] - pts[-1], axis=1) < 0.5)
            alive &= ok
            pts.append(p1[:, 0])
        P = np.stack(pts, 1)[alive]  # (n, WINDOW, 2)
        for row in P:
            out.append((idx, row))
    return out


def undistort(cam, uv):
    w, h = cam["size"]
    xd, yd = (uv[:, 0] - w / 2) / cam["fx"], (uv[:, 1] - h / 2) / cam["fx"]
    x, y = xd, yd
    for _ in range(8):
        d = 1 + cam["k1"] * (x * x + y * y)
        x, y = xd / d, yd / d
    return np.stack([x, y], 1)


def project(cam, T, X):
    Ti = np.linalg.inv(T)
    pc = X @ Ti[:3, :3].T + Ti[:3, 3]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    d = 1 + cam["k1"] * (x * x + y * y)
    w, h = cam["size"]
    return np.stack([cam["fx"] * x * d + w / 2, cam["fx"] * y * d + h / 2], 1)


def correction(p):
    """p = [fx scale - 1, rx, ry, rz (deg), tx, ty, tz (mm)] -> (fx factor, 4x4 change to T_flange_cam)."""
    D = np.eye(4)
    D[:3, :3] = Rotation.from_euler("xyz", p[1:4], degrees=True).as_matrix()
    D[:3, 3] = np.asarray(p[4:7]) / 1000
    return 1 + p[0], D


def residuals(p, trk, Tf, cam0, T_fc):
    s, D = correction(p)
    cam = dict(cam0, fx=cam0["fx"] * s)
    Tfc = T_fc @ D
    res = []
    for idx, uv in trk:
        Ts = [Tf[i] @ Tfc for i in idx]
        rays = undistort(cam, uv)
        # linear triangulation (midpoint-free DLT on normalised rays)
        A = []
        for T, (x, y) in zip(Ts, rays):
            Ti = np.linalg.inv(T)[:3]
            A.append(x * Ti[2] - Ti[0])
            A.append(y * Ti[2] - Ti[1])
        _, _, vt = np.linalg.svd(np.array(A))
        X = vt[-1]
        if abs(X[3]) < 1e-9:
            res.append(np.full(2 * len(idx), 5.0))
            continue
        X = X[:3] / X[3]
        r = np.concatenate([project(cam, T, X[None])[0] - u for T, u in zip(Ts, uv)])
        res.append(np.clip(r, -5, 5))
    return np.concatenate(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--times", type=Path, default=None)
    ap.add_argument("--delay-ms", type=float, default=140.0)
    a = ap.parse_args()
    cam, T_fc, _ = cameras(None)
    jt, jq, frames = load(a.run)
    if a.times:
        tm = json.loads(a.times.read_text())["times"]
        for f in frames:
            f["t"] = tm.get(str(f["path"]), f["t"] - a.delay_ms / 1000)
    else:
        for f in frames:
            f["t"] -= a.delay_ms / 1000
    Tf = [flange_pose(joints_at(jt, jq, f["t"])) for f in frames]
    trk = track_windows(frames)
    # keep tracks the camera moved enough along for depth (a still window says nothing)
    trk = [(i, uv) for i, uv in trk if np.linalg.norm(Tf[i[0]][:3, 3] - Tf[i[-1]][:3, 3]) > 0.01]
    rng = np.random.default_rng(0)
    if len(trk) > MAX_TRACKS:
        trk = [trk[k] for k in rng.choice(len(trk), MAX_TRACKS, replace=False)]
    print(f"{len(frames)} pictures, {len(trk)} tracks of {WINDOW}")
    p0 = np.zeros(7)
    r0 = residuals(p0, trk, Tf, cam, T_fc)
    fit = least_squares(residuals, p0, args=(trk, Tf, cam, T_fc), loss="soft_l1", f_scale=1.0,
                        x_scale=[0.01, 0.5, 0.5, 0.5, 2, 2, 2], max_nfev=60)
    r1 = fit.fun
    s, D = correction(fit.x)
    out = {"tracks": len(trk), "reproj_px_before": {"median": round(float(np.median(np.abs(r0))), 3),
                                                     "p90": round(float(np.percentile(np.abs(r0), 90)), 3)},
           "reproj_px_after": {"median": round(float(np.median(np.abs(r1))), 3), "p90": round(float(np.percentile(np.abs(r1), 90)), 3)},
           "fx_scale": round(float(s), 4), "rot_deg": np.round(fit.x[1:4], 3).tolist(), "trans_mm": np.round(fit.x[4:7], 2).tolist(),
           "T_flange_cam": (T_fc @ D).tolist(), "fx_848": float(cam["fx"] * s)}
    (a.run / "calib_video.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "T_flange_cam"}, indent=1))


if __name__ == "__main__":
    main()
