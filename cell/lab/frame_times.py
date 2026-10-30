"""Each wrist picture's own moment on the arm's path, solved from the pictures (no fixed delay).

    uv run python -m lab.frame_times runs/<time>_datarun_real [--smooth 0|2] [--delay-ms 140]

The joints give the camera's path (lab/timing_real.py: every answer's joints at the cell's clock; --smooth k averages
each joint reading with k on either side, against readings whose own age wanders). Each distinct picture starts at its
first appearance minus the fixed delay; then, a few rounds over all of them, each picture's time is moved (+-60 ms) to
where points on the table, cast onto the table plane from it and from its neighbours, land closest together (the
table is flat and its height known). Neighbours fix each other's relative times; the shared offset left over (a common
shift is invisible picture to picture at a steady speed) is set by passes that went opposite ways over the same table.
Writes <run>/frame_times.json: {picture path: solved time} and the before/after numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose
from lab.scene3d import cameras, rays_to_plane
from lab.timing_real import cross, joints_at, load, tracks


def smooth_joints(jt, jq, k: int):
    if k <= 0:
        return jt, jq
    out = jq.copy()
    for i in range(len(jq)):
        lo, hi = max(0, i - k), min(len(jq), i + k + 1)
        out[i] = jq[lo:hi].mean(0)
    return jt, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--delay-ms", type=float, default=140.0)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    cam, T_fc, _ = cameras(None)
    jt, jq, frames = load(a.run)
    jt, jq = smooth_joints(jt, jq, a.smooth)
    trk = tracks(frames)
    by = {}
    for i, j, ua, ub in trk:
        by.setdefault(i, []).append((j, ua, ub, True))
        by.setdefault(j, []).append((i, ub, ua, False))
    s = np.array([f["t"] for f in frames]) - a.delay_ms / 1000
    pose_cache = {}

    def pose(t):
        key = round(t, 4)
        if key not in pose_cache:
            pose_cache[key] = flange_pose(joints_at(jt, jq, t)) @ T_fc
        return pose_cache[key]

    def slide_at(i, ti):
        tot = []
        Ti = pose(ti)
        for j, ui, uj, _ in by.get(i, []):
            Tj = pose(s[j])
            if np.linalg.norm(Ti[:3, 3] - Tj[:3, 3]) < 0.002:
                continue
            pa = rays_to_plane(dict(cam, T=Ti), ui, SURFACE_Z)
            pb = rays_to_plane(dict(cam, T=Tj), uj, SURFACE_Z)
            tot.append(np.median(np.linalg.norm(pa - pb, axis=1)) * 1000)
        return float(np.mean(tot)) if tot else None

    def overall():
        v = [slide_at(i, s[i]) for i in by]
        v = [x for x in v if x is not None]
        return float(np.median(v)), len(v)

    before, n = overall()
    print(f"{len(frames)} pictures, {n} with neighbours; median table slide picture to picture: {before:.2f} mm "
          f"(fixed delay {a.delay_ms:.0f} ms, joints smoothed over {a.smooth})")
    offsets = np.arange(-0.06, 0.0601, 0.005)
    for r in range(a.rounds):
        moved = []
        for i in by:
            base = s[i]
            scores = [(slide_at(i, base + d), d) for d in offsets]
            scores = [x for x in scores if x[0] is not None]
            if not scores:
                continue
            best, d = min(scores)
            s[i] = base + d
            moved.append(d)
        after, _ = overall()
        print(f"round {r + 1}: median table slide {after:.2f} mm; pictures moved by median "
              f"{np.median(np.abs(moved)) * 1000:.0f} ms (90th percentile {np.percentile(np.abs(moved), 90) * 1000:.0f})")
    # the shared offset: passes going opposite ways over the same table
    for f, t in zip(frames, s):
        f["t_solved"] = float(t)
    fr = [dict(f, t=f["t_solved"]) for f in frames]
    Ds = np.arange(-0.10, 0.101, 0.01)
    D, curve, npairs = cross(fr, jt, jq, cam, T_fc, Ds)
    print(f"shared offset from opposite passes: {D * 1000:+.0f} ms ({npairs} picture pairs; gap at best "
          f"{np.nanmin(curve):.1f} mm, at 0: {curve[len(Ds) // 2]:.1f} mm)")
    s_final = s - D
    resid = s_final - (np.array([f["t"] for f in frames]) - a.delay_ms / 1000)
    out = {"delay_ms_start": a.delay_ms, "smooth": a.smooth, "slide_before_mm": round(before, 2),
           "slide_after_mm": round(after, 2), "shared_offset_ms": round(D * 1000), "cross_gap_mm": round(float(np.nanmin(curve)), 2),
           "moved_ms": {"median_abs": round(float(np.median(np.abs(resid))) * 1000), "p90_abs": round(float(np.percentile(np.abs(resid), 90)) * 1000)},
           "times": {str(f["path"]): float(t) for f, t in zip(frames, s_final)}}
    (a.run / f"frame_times_s{a.smooth}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "times"}, indent=1))


if __name__ == "__main__":
    main()
