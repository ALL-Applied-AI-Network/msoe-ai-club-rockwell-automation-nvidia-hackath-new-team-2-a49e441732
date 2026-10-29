"""3D from the wrist video itself, every frame: follow points through the video, place each one in 3D from its rays.

    uv run python -m lab.tracks3d runs/<time>_datarun_real      -> runs/<...>/tracks3d.rrd, then
    uv run rerun runs/<time>_datarun_real/tracks3d.rrd

1. Follow points: every frame, pyramidal Lucas-Kanade from the last frame, kept only if tracking back lands within
   BACK_PX of where it started; new corners wherever nothing is tracked.
2. Re-time the frames. A picture and the joints that come with it are not taken at the same instant: on the
   Oct 28 run the picture was ~1.4 frames (~140 ms) older while the arm moved, and the gap changes frame to frame,
   so a moving frame's camera was 2-9 px off. Each frame gets its own offset along the joints' path, the one
   that makes the tracks' rays meet best (alternating with placing the points). Saved as frame_offsets.npy.
3. Watch it build: at each frame every track it sees adds that ray; a track's 3D point is the point nearest all its
   rays so far (least squares, kept as running sums). It counts once the rays span MIN_ANGLE_DEG and pass within
   GOOD_PX of it.

Timeline "frame", in Rerun:
    world/wrist/cam   the picture and its tracks: grey still gathering rays, coloured by height once placed
    world/tracks/now  the 3D points of the tracks this frame sees, coloured by height (blue table, red HEIGHT_TOP up)
    world/tracks/all  every placed point so far, in its colour from the picture
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z, flange_pose
from lab.scene3d import cameras

LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
BACK_PX = 0.5          # forward-backward tracking check
MAX_CORNERS = 3000
MIN_GAP_PX = 6         # new corners at least this far from any track
MIN_LEN = 6            # frames a track must last to be used
MIN_ANGLE_DEG = 3.0    # rays must span this much before a point counts
GOOD_PX = 1.0          # rms distance of its rays from the point, in the image
LAG0 = -1.0            # starting guess: the picture is a frame older than its joints
OFFSETS = np.arange(-3.0, 1.01, 0.1)  # first search around the current offset, in frames (narrows each round)
ROUNDS = 4
REGION = ((0.15, 0.70), (-0.40, 0.40), (-0.005, 0.15))  # x, y, height above the table (m)
HEIGHT_TOP = 0.06      # colour scale top (m): a standing battery is ~50 mm


def follow(run: Path, rows) -> list:
    """Tracks through the whole video: lists of (frame, (u, v)), MIN_LEN frames or longer."""
    live, done, nid, prev = {}, [], 0, None
    for k, r in enumerate(rows):
        g = cv2.cvtColor(cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg")), cv2.COLOR_BGR2GRAY)
        if prev is not None and live:
            ids = list(live)
            uv = np.array([live[i][-1][1] for i in ids], np.float32)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, uv[:, None], None, **LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None, **LK)
            nxt = nxt[:, 0]
            ok = ((st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back[:, 0] - uv, axis=1) < BACK_PX)
                  & (nxt[:, 0] > 4) & (nxt[:, 0] < 844) & (nxt[:, 1] > 4) & (nxt[:, 1] < 476))
            for i, keep, q in zip(ids, ok, nxt):
                if keep:
                    live[i].append((k, q.copy()))
                else:
                    done.append(live.pop(i))
        mask = np.full(g.shape, 255, np.uint8)
        for t in live.values():
            cv2.circle(mask, (int(t[-1][1][0]), int(t[-1][1][1])), MIN_GAP_PX, 0, -1)
        c = cv2.goodFeaturesToTrack(g, MAX_CORNERS - len(live), 0.005, MIN_GAP_PX, mask=mask, blockSize=5) \
            if len(live) < MAX_CORNERS - 50 else None
        if c is not None:
            for q in c[:, 0]:
                live[nid] = [(k, q.copy())]
                nid += 1
        prev = g
    return [t for t in done + list(live.values()) if len(t) >= MIN_LEN]


class Obs:
    """All the tracks' observations, flat: track id, frame, pixel, and the camera-frame ray (undistorted)."""

    def __init__(self, tracks, cam):
        self.tid = np.concatenate([[i] * len(t) for i, t in enumerate(tracks)])
        self.fr = np.concatenate([[k for k, _ in t] for t in tracks])
        self.uv = np.concatenate([[q for _, q in t] for t in tracks]).astype(float)
        self.m = len(tracks)
        self.fx, self.k1 = cam["fx"], cam["k1"]
        self.cx, self.cy = cam["size"][0] / 2, cam["size"][1] / 2
        xd, yd = (self.uv[:, 0] - self.cx) / self.fx, (self.uv[:, 1] - self.cy) / self.fx
        x, y = xd, yd
        for _ in range(8):
            d = 1 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        r = np.stack([x, y, np.ones_like(x)], 1)
        self.ray = r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, T, P):
        pc = np.einsum("...ji,...j->...i", T[..., :3, :3], P - T[..., :3, 3])
        x, y = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.stack([self.fx * x * d + self.cx, self.fx * y * d + self.cy], -1)

    def place(self, T, sel=slice(None)):
        """Robust least-squares points for every track from observations sel under camera poses T (frames, 4, 4)."""
        tid, fr, ray, uv = self.tid[sel], self.fr[sel], self.ray[sel], self.uv[sel]
        r = np.einsum("nij,nj->ni", T[fr, :3, :3], ray)
        M0 = np.eye(3)[None] - r[:, :, None] * r[:, None, :]
        Mo0 = np.einsum("nij,nj->ni", M0, T[fr, :3, 3])
        wt = np.ones(len(tid))
        for _ in range(2):
            A, b = np.zeros((self.m, 3, 3)), np.zeros((self.m, 3))
            np.add.at(A, tid, M0 * wt[:, None, None])
            np.add.at(b, tid, Mo0 * wt[:, None])
            P = np.linalg.solve(A + 1e-9 * np.eye(3), b[:, :, None])[:, :, 0]
            e = np.linalg.norm(self.project(T[fr], P[tid]) - uv, axis=1)
            wt = 1 / np.maximum(1, e / 1.5)
        rms = np.sqrt(np.bincount(tid, e ** 2, self.m) / np.maximum(np.bincount(tid, minlength=self.m), 1))
        return P, e, rms


def poses(J, T_fc, f):
    """Camera poses at fractional frame indices f, the joints interpolated between neighbouring frames."""
    f = np.clip(f, 0, len(J) - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, len(J) - 1)
    a = (f - i0)[..., None]
    q = ((1 - a) * J[i0] + a * J[i1]).reshape(-1, J.shape[1])
    return np.array([flange_pose(j) @ T_fc for j in q]).reshape(*np.shape(f), 4, 4)


def retime(obs: Obs, J, T_fc, say=print):
    """Per-frame offsets (frames) along the joint path that make the tracks' rays meet; alternates with placing."""
    n = len(J)
    off = np.full(n, LAG0)
    D = OFFSETS.copy()
    by_frame = [np.nonzero(obs.fr == k)[0] for k in range(n)]
    for rnd in range(ROUNDS + 1):
        T = poses(J, T_fc, np.arange(n) + off)
        P, e, rms = obs.place(T)
        say(f"round {rnd}: observations median {np.median(e):.2f} px off; tracks within 1 px {np.mean(rms < 1):.0%}, "
            f"within 2 px {np.mean(rms < 2):.0%}")
        if rnd == ROUNDS:
            return off, T
        good = rms < 3
        for k in range(n):
            o = by_frame[k]
            o = o[good[obs.tid[o]]]
            if len(o) < 15:
                continue
            Tk = poses(J, T_fc, k + off[k] + D)
            err = np.linalg.norm(obs.project(Tk[:, None], P[obs.tid[o]][None]) - obs.uv[o][None], axis=2)
            # a small pull toward LAG0: where the arm stood still every offset fits, keep the plain one
            cost = np.median(np.minimum(err, 5), 1) + 0.02 * np.abs(off[k] + D - LAG0)
            off[k] += D[np.argmin(cost)]
        D = D * 0.6


def in_region(p):
    (x0, x1), (y0, y1), (h0, h1) = REGION
    with np.errstate(invalid="ignore"):
        h = p[:, 2] - SURFACE_Z
        return (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (h > h0) & (h < h1)


def height_colours(p):
    if len(p) == 0:
        return np.zeros((0, 3), np.uint8)
    h = np.clip((p[:, 2] - SURFACE_Z) / HEIGHT_TOP, 0, 1)
    return cv2.applyColorMap((h * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]


def main(run: Path) -> None:
    w, T_fc, _ = cameras(run)
    rows = [json.loads(l) for l in open(run / "frames.jsonl")]
    J = np.array([r["joints"] for r in rows])
    tic = time.time()
    tracks = follow(run, rows)
    print(f"{len(tracks)} tracks of {MIN_LEN}+ frames ({time.time() - tic:.0f} s)", flush=True)
    obs = Obs(tracks, w)
    off, T = retime(obs, J, T_fc)
    np.save(run / "frame_offsets.npy", off)
    print(f"frame offsets saved ({time.time() - tic:.0f} s)", flush=True)
    first_rgb = {}

    rr.init("tracks3d", spawn=False)
    rr.save(str(run / "tracks3d.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    (x0, x1), (y0, y1), _ = REGION
    rr.log("world/table", rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, SURFACE_Z - 0.001]],
                                    half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, 0.0005]], colors=[[200, 200, 200]]), static=True)
    K = np.array([[w["fx"], 0, 424], [0, w["fx"], 240], [0, 0, 1]])
    # running sums per track, in frame order, with the re-timed poses
    r_all = np.einsum("nij,nj->ni", T[obs.fr, :3, :3], obs.ray)
    o_all = T[obs.fr, :3, 3]
    A, b, c = np.zeros((obs.m, 3, 3)), np.zeros((obs.m, 3)), np.zeros(obs.m)
    cnt = np.zeros(obs.m, int)
    d0 = np.full((obs.m, 3), np.nan)
    ang = np.zeros(obs.m)
    rgb = np.zeros((obs.m, 3), np.uint8)
    P = np.full((obs.m, 3), np.nan)
    ok_all = np.zeros(obs.m, bool)
    order = np.argsort(obs.fr, kind="stable")
    starts = np.searchsorted(obs.fr[order], np.arange(len(rows) + 1))
    t0 = rows[0]["t"]
    for k, r in enumerate(rows):
        img = cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg"))
        o_ = order[starts[k]:starts[k + 1]]
        ids = obs.tid[o_]
        rk, ok_ = r_all[o_], o_all[o_]
        M = np.eye(3)[None] - rk[:, :, None] * rk[:, None, :]
        Mo = np.einsum("nij,nj->ni", M, ok_)
        A[ids] += M
        b[ids] += Mo
        c[ids] += (Mo * ok_).sum(1)
        cnt[ids] += 1
        new = np.isnan(d0[ids, 0])
        d0[ids[new]] = rk[new]
        uvk = obs.uv[o_]
        rgb[ids[new]] = img[uvk[new, 1].astype(int), uvk[new, 0].astype(int), ::-1]
        ang[ids] = np.maximum(ang[ids], np.degrees(np.arccos(np.clip((d0[ids] * rk).sum(1), -1, 1))))
        # re-place the tracks this frame sees
        p = np.linalg.solve(A[ids] + 1e-9 * np.eye(3), b[ids][:, :, None])[:, :, 0]
        ss = np.einsum("ni,nij,nj->n", p, A[ids], p) - 2 * (b[ids] * p).sum(1) + c[ids]
        err = np.sqrt(np.maximum(ss, 0) / cnt[ids]) / np.maximum(np.linalg.norm(p - ok_, axis=1), 1e-3) * w["fx"]
        P[ids] = p
        good = (ang[ids] >= MIN_ANGLE_DEG) & (err <= GOOD_PX) & (cnt[ids] >= MIN_LEN) & in_region(p)
        ok_all[ids] = good

        rr.set_time("frame", sequence=k)
        rr.set_time("seconds", duration=r["t"] - t0)
        Tk = T[k]
        rr.log("world/wrist", rr.Transform3D(translation=Tk[:3, 3], mat3x3=Tk[:3, :3]))
        rr.log("world/wrist/cam", rr.Pinhole(image_from_camera=K, resolution=[848, 480], image_plane_distance=0.05))
        rr.log("world/wrist/cam", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        col2 = np.full((len(ids), 3), 150, np.uint8)
        col2[good] = height_colours(p[good])
        rr.log("world/wrist/cam/tracks", rr.Points2D(uvk, colors=col2, radii=1.5))
        rr.log("world/tracks/now", rr.Points3D(p[good], colors=height_colours(p[good]), radii=0.0012))
        rr.log("world/tracks/all", rr.Points3D(P[ok_all], colors=rgb[ok_all], radii=0.0008))
        rr.log("info", rr.TextLog(f"frame {k}: {r['phase']}, {len(ids)} tracks, {good.sum()} placed now, "
                                  f"{ok_all.sum()} placed so far; picture re-timed {off[k]:+.2f} frames"))
        if k % 100 == 0:
            print(f"frame {k:4d} {r['phase']:9s}: {len(ids)} tracks, {good.sum():4d} placed now, "
                  f"{ok_all.sum():6d} so far", flush=True)
    np.savez(run / "tracks3d.npz", p=P[ok_all], rgb=rgb[ok_all])
    print(f"{ok_all.sum()} points placed ({time.time() - tic:.0f} s); saved {run / 'tracks3d.rrd'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
