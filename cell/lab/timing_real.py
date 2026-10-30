"""The wrist pictures' timing, measured from a real recording with the table as the ruler.

    uv run python -m lab.timing_real runs/<time>_datarun_real [--max-ms 400]

A data run (lab/datarun.py) saved every wrist picture with the cell's clock and the joints from the same answer. The
picture is older than the joints by an unknown delay. Points on the table are tracked from picture to picture (corner
features, optical flow, checked forwards and backwards); each one, cast from its camera onto the table plane (whose
height we know), must land on the same spot in both pictures. With the right delay the cameras are where they really
were and table points stay put; with a wrong one every camera is off by speed x error and they slide. So: for each
candidate delay, the median slide over all tracked points; the delay with the least is the measured one. Points on
things above the table slide a little even then (they're not on the plane): the median ignores them.

Also: each distinct picture's first appearance (repeats are dropped by their bytes), and the same measurement split by
the arm's direction of travel (both should agree: a wrong delay slides opposite ways going opposite ways).
Writes <run>/timing.json and <run>/timing.jpg (the slide against the delay).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose
from lab.scene3d import cameras, rays_to_plane


def load(run: Path):
    rows = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["t"])
    jt = np.array([r["t"] for r in rows])
    jq = np.array([r["joints"][:6] for r in rows], float)
    seen, frames = set(), []
    for r in rows:  # each distinct picture, at its first appearance
        f = run / "frames" / f"{r['n']:05d}.jpg"
        if not f.exists():
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        frames.append({"t": r["t"], "path": f, "phase": r["phase"]})
    return jt, jq, frames


def on_beat(frames, period: float = 0.1) -> None:
    """Each picture's time snapped back to the camera's 10 Hz beat (the phase where first appearances bunch up just
    after: lab/pull.py's convention), stored as f["t"]; the first appearance kept as f["t_seen"]."""
    t = np.array([f["t"] for f in frames])
    best = max(np.arange(0, period, 0.001), key=lambda ph: np.sum(((t - ph) % period) < 0.03))
    for f, ti in zip(frames, t):
        f["t_seen"] = ti
        f["t"] = ti - ((ti - best) % period)


def joints_at(jt, jq, t):
    return np.array([np.interp(t, jt, jq[:, k]) for k in range(6)])


def tracks(frames, step: int = 1, most: int = 300):
    """Pixel pairs between pictures i and i+step: [(i, j, uv_i, uv_j)]."""
    out = []
    grey = [cv2.cvtColor(cv2.imread(str(f["path"])), cv2.COLOR_BGR2GRAY) for f in frames]
    for i in range(len(frames) - step):
        j = i + step
        if frames[i]["phase"] != frames[j]["phase"] or frames[j]["t"] - frames[i]["t"] > 0.5:
            continue
        p0 = cv2.goodFeaturesToTrack(grey[i], most, 0.01, 8)
        if p0 is None or len(p0) < 10:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(grey[i], grey[j], p0, None, winSize=(21, 21), maxLevel=3)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(grey[j], grey[i], p1, None, winSize=(21, 21), maxLevel=3)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 0.5)
        if good.sum() < 10:
            continue
        out.append((i, j, p0[good, 0].astype(float), p1[good, 0].astype(float)))
    return out


def slide(frames, trk, jt, jq, cam, T_fc, delay: float, which=None):
    """Median distance (mm) between where each tracked point lands on the table from its two pictures, the pictures
    taken `delay` s before the cell's clock said; and the per-pair medians with each pair's direction of travel."""
    pose = {}
    def T(i):
        if i not in pose:
            pose[i] = flange_pose(joints_at(jt, jq, frames[i]["t"] - delay)) @ T_fc
        return pose[i]
    per = []
    for i, j, a, b in trk:
        Ti, Tj = T(i), T(j)
        move = (Tj[:3, 3] - Ti[:3, 3]) * 1000
        if np.linalg.norm(move) < 2.0:  # not moving: no information about the delay
            continue
        if which is not None and not which(move):
            continue
        pa = rays_to_plane(dict(cam, T=Ti), a, SURFACE_Z)
        pb = rays_to_plane(dict(cam, T=Tj), b, SURFACE_Z)
        per.append(float(np.median(np.linalg.norm(pa - pb, axis=1)) * 1000))
    return (float(np.median(per)) if per else np.nan), len(per)


def cross(frames, jt, jq, cam, T_fc, Ds, every: int = 3, near_mm: float = 40.0):
    """The delay from passes going opposite ways over the same table: pictures from two passes whose cameras were
    within near_mm (and travelling opposite ways) are matched (SIFT); each match cast onto the table plane from both;
    a wrong delay puts the two 2 x speed x error apart. Returns (delay s, curve, pairs used)."""
    sift = cv2.SIFT_create(800)
    idx = list(range(0, len(frames), every))
    def vel(i):
        t = frames[i]["t"]
        a_ = flange_pose(joints_at(jt, jq, t - 0.05))[:3, 3]
        b_ = flange_pose(joints_at(jt, jq, t + 0.05))[:3, 3]
        return (b_ - a_) / 0.1
    V = {i: vel(i) for i in idx}
    C = {i: flange_pose(joints_at(jt, jq, frames[i]["t"]))[:3, 3] for i in idx}
    feats = {}
    def F(i):
        if i not in feats:
            g = cv2.cvtColor(cv2.imread(str(frames[i]["path"])), cv2.COLOR_BGR2GRAY)
            feats[i] = sift.detectAndCompute(g, None)
        return feats[i]
    pairs = []
    for ii, i in enumerate(idx):
        if np.linalg.norm(V[i]) < 0.02:  # moving at least 20 mm/s
            continue
        for j in idx[ii + 1:]:
            if np.linalg.norm(V[j]) < 0.02 or V[i] @ V[j] > -0.5 * np.linalg.norm(V[i]) * np.linalg.norm(V[j]):
                continue  # not going the opposite way
            if np.linalg.norm(C[i] - C[j]) * 1000 > near_mm:
                continue
            ki, di = F(i)
            kj, dj = F(j)
            if di is None or dj is None:
                continue
            m = cv2.BFMatcher().knnMatch(di, dj, k=2)
            good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.7 * x[1].distance]
            if len(good) < 12:
                continue
            a_ = np.array([ki[g.queryIdx].pt for g in good])
            b_ = np.array([kj[g.trainIdx].pt for g in good])
            pairs.append((i, j, a_, b_))
            if len(pairs) >= 200:
                break
        if len(pairs) >= 200:
            break
    curve = []
    for D in Ds:
        per = []
        for i, j, a_, b_ in pairs:
            Ti = flange_pose(joints_at(jt, jq, frames[i]["t"] - D)) @ T_fc
            Tj = flange_pose(joints_at(jt, jq, frames[j]["t"] - D)) @ T_fc
            pa = rays_to_plane(dict(cam, T=Ti), a_, SURFACE_Z)
            pb = rays_to_plane(dict(cam, T=Tj), b_, SURFACE_Z)
            per.append(float(np.median(np.linalg.norm(pa - pb, axis=1)) * 1000))
        curve.append(float(np.median(per)) if per else np.nan)
    k = int(np.nanargmin(curve)) if np.isfinite(curve).any() else 0
    return float(Ds[k]), curve, len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--max-ms", type=float, default=400.0)
    ap.add_argument("--beat", action="store_true", help="measure from the camera's beat, not first appearance")
    a = ap.parse_args()
    cam, T_fc, _ = cameras(None)
    jt, jq, frames = load(a.run)
    if a.beat:
        on_beat(frames)
        print("times snapped to the camera's 10 Hz beat (lab/pull.py's convention)")
    trk = tracks(frames)
    print(f"{len(frames)} distinct pictures, {len(trk)} tracked pairs")
    Ds = np.arange(-100.0, a.max_ms + 1, 10.0) / 1000
    curve = [slide(frames, trk, jt, jq, cam, T_fc, D)[0] for D in Ds]
    k = int(np.nanargmin(curve))
    fine = np.arange(Ds[k] - 0.012, Ds[k] + 0.0121, 0.002)
    fc = [slide(frames, trk, jt, jq, cam, T_fc, D)[0] for D in fine]
    best = float(fine[int(np.nanargmin(fc))])
    # the same, split by which way the arm travels (x or y, either sign)
    splits = {}
    for name, w in {"+x": lambda m: m[0] > abs(m[1]), "-x": lambda m: -m[0] > abs(m[1]),
                    "+y": lambda m: m[1] > abs(m[0]), "-y": lambda m: -m[1] > abs(m[0])}.items():
        c = [slide(frames, trk, jt, jq, cam, T_fc, D, w)[0] for D in Ds]
        if np.isfinite(c).any():
            splits[name] = {"delay_ms": round(float(Ds[int(np.nanargmin(c))]) * 1000), "slide_mm": round(float(np.nanmin(c)), 2),
                            "pairs": slide(frames, trk, jt, jq, cam, T_fc, Ds[int(np.nanargmin(c))], w)[1]}
    Dx, cx, npx = cross(frames, jt, jq, cam, T_fc, Ds)
    print(f"opposite passes: {npx} picture pairs; best delay {Dx * 1000:.0f} ms; the gap between where the two passes "
          f"put table points: " + ", ".join(f"{D * 1000:.0f} ms {c:.1f} mm" for D, c in zip(Ds[::3], cx[::3])))
    s0 = slide(frames, trk, jt, jq, cam, T_fc, 0.0)[0]
    s_best = slide(frames, trk, jt, jq, cam, T_fc, best)[0]
    res = {"cross_delay_ms": round(Dx * 1000), "cross_pairs": npx,
           "cross_curve": {f"{D * 1000:.0f}": round(float(c), 2) for D, c in zip(Ds, cx)},
           "delay_ms": round(best * 1000, 1), "slide_mm_at_delay": round(s_best, 2), "slide_mm_at_0": round(s0, 2),
           "pictures": len(frames), "pairs": len(trk), "by_direction": splits,
           "curve": {f"{D * 1000:.0f}": round(float(c), 2) for D, c in zip(Ds, curve)}}
    (a.run / ("timing_beat.json" if a.beat else "timing.json")).write_text(json.dumps(res, indent=1))
    # the curve as a picture
    W, H = 640, 300
    img = np.full((H, W, 3), 255, np.uint8)
    cv = np.array(curve)
    top = np.nanmax(cv)
    pts = [(int(40 + (D - Ds[0]) / (Ds[-1] - Ds[0]) * (W - 60)), int(H - 30 - c / top * (H - 60))) for D, c in zip(Ds, cv)
           if np.isfinite(c)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (180, 90, 30), 2)
    xb = int(40 + (best - Ds[0]) / (Ds[-1] - Ds[0]) * (W - 60))
    cv2.line(img, (xb, 20), (xb, H - 30), (40, 40, 200), 1)
    cv2.putText(img, f"delay {best * 1000:.0f} ms: table points slide {s_best:.2f} mm (at 0 ms: {s0:.2f})", (40, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"{Ds[0] * 1000:.0f} ms", (30, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, f"{Ds[-1] * 1000:.0f} ms", (W - 70, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(a.run / ("timing_beat.jpg" if a.beat else "timing.jpg")), img)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
