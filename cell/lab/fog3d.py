"""Evidence for and against something being in each voxel, from every frame of the wrist video, watched as it builds.

    uv run python -m lab.fog3d runs/<time>_datarun_real      -> runs/<...>/fog3d.rrd, then
    uv run rerun runs/<time>_datarun_real/fog3d.rrd

Each 2 mm voxel keeps a weight of evidence L (log odds): 0 is fog (no idea), +inf "something is here", -inf "nothing
is here". It only ever adds evidence, never guesses:
    1. Follow points through the video and re-time each frame along the joint path (lab/tracks3d.py).
    2. Match against every earlier view, not just the last frame: once a point has a 3D estimate it is predicted in
       every frame that sees it, and a small search there (Lucas-Kanade from the predicted spot, against the frame
       where it was seen from the nearest viewpoint) either finds it within REMATCH_PX of the prediction and with a
       patch that agrees (NCC >= REMATCH_NCC) or adds nothing. Far-apart frames are far-apart eyes.
    3. A point counts once its rays meet within GOOD_PX and its depth is known to SIGMA_MM (the 1-sigma long axis
       of its uncertainty from all its rays: wider spacing and nearer cameras shrink it). Blank table has no points,
       so it stays fog: no match means no evidence.
    4. Each observation of a counted point adds HIT * quality at the point (quality = VOX / sigma, at most 1), and
       FREE (negative) to every voxel its ray passed through on the way there, stopping 3 sigma short.

Timeline "frame", in Rerun:
    world/wrist/cam        the picture and its points: grey still unsure, coloured by height once counted
    world/fog/something    voxels with L >= SURE (in their colour from the picture)
    world/fog/nothing      voxels with L <= -SURE, every FOG_EVERY frames, as 6 mm blocks (toggle on to see)
    fog_map                top-down: red something, blue nothing seen through down to the table, white fog
    plots/                 voxel counts and the median depth uncertainty over time
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z
from lab.scene3d import cameras
from lab.tracks3d import Obs, follow, height_colours, in_region, retime

VOX = 0.002
REGION = ((0.15, 0.70), (-0.40, 0.40), (-0.006, 0.15))  # x, y, height above the table (m)
GOOD_PX = 1.0           # rms of a point's rays, in the image
SIGMA_MM = 2.0          # depth known this well (1 sigma, long axis) before a point counts
PX_SIGMA = 0.7          # one observation's pixel noise, for the uncertainty
REMATCH_PX = 2.0
REMATCH_NCC = 0.85
HIT, FREE = 0.85, -0.4  # log odds per observation (p = 0.70 something, p = 0.40 nothing)
SURE = 2.2              # |L| for "sure" (90 %)
FOG_EVERY = 10


def sigma_long(obs: Obs, T, P, sel=slice(None)):
    """1-sigma long axis (m) of each track's 3D uncertainty from its rays: H = sum (I - d d') / (range * angle sigma)^2."""
    tid, fr, ray = obs.tid[sel], obs.fr[sel], obs.ray[sel]
    d = np.einsum("nij,nj->ni", T[fr, :3, :3], ray)
    rng = np.linalg.norm(P[tid] - T[fr, :3, 3], axis=1)
    s = rng * PX_SIGMA / obs.fx
    M = (np.eye(3)[None] - d[:, :, None] * d[:, None, :]) / (s * s)[:, None, None]
    H = np.zeros((obs.m, 3, 3))
    np.add.at(H, tid, M)
    lam = np.linalg.eigvalsh(H + 1e-6 * np.eye(3))[:, 0]
    return 1 / np.sqrt(np.maximum(lam, 1e-12))


def ncc(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-6)


def patches(img, uv, r=5):
    """(n, (2r+1)^2) patches around integer-rounded uv (inside the image)."""
    o = np.arange(-r, r + 1)
    u = np.clip(np.rint(uv[:, 0]).astype(int)[:, None, None] + o[None, None, :], 0, img.shape[1] - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(int)[:, None, None] + o[None, :, None], 0, img.shape[0] - 1)
    return img[v, u].reshape(len(uv), -1).astype(np.float32)


def rematch(obs: Obs, T, P, ok, grays, say=print):
    """New observations of the points ok in every frame that should see them; returns (tid, frame, uv) arrays."""
    n = len(grays)
    seen = set(zip(obs.tid.tolist(), obs.fr.tolist()))
    centres = T[:, :3, 3]
    ids = np.nonzero(ok)[0]
    # each point's own views: (frame, uv), to pick the template from the nearest viewpoint
    own = {}
    for i, (t, f) in enumerate(zip(obs.tid, obs.fr)):
        if ok[t]:
            own.setdefault(t, []).append(i)
    out_t, out_f, out_uv = [], [], []
    for k in range(n):
        q = obs.project(T[k][None], P[ids])
        front = np.einsum("j,nj->n", T[k][:3, 2], P[ids] - T[k][:3, 3]) > 0.05
        inside = front & (q[:, 0] > 12) & (q[:, 0] < 836) & (q[:, 1] > 12) & (q[:, 1] < 468)
        cand = [(t, uv) for t, uv, m in zip(ids, q, inside) if m and (t, k) not in seen]
        if not cand:
            continue
        groups = {}
        for t, uv in cand:
            views = own[t]
            j = views[int(np.argmin(np.linalg.norm(centres[obs.fr[views]] - centres[k], axis=1)))]
            groups.setdefault(obs.fr[j], []).append((t, obs.uv[j], uv))
        for j, g in groups.items():
            src = np.array([a for _, a, _ in g], np.float32)
            pred = np.array([b for _, _, b in g], np.float32)
            got, st, _ = cv2.calcOpticalFlowPyrLK(grays[j], grays[k], src[:, None], pred[:, None].copy(),
                                                  winSize=(15, 15), maxLevel=1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[k], grays[j], got, src[:, None].copy(),
                                                    winSize=(15, 15), maxLevel=1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            got = got[:, 0]
            good = ((st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back[:, 0] - src, axis=1) < 0.5)
                    & (np.linalg.norm(got - pred, axis=1) < REMATCH_PX))
            if good.any():
                good[good] = ncc(patches(grays[j], src[good]), patches(grays[k], got[good])) >= REMATCH_NCC
            for (t, _, _), m, uv in zip(g, good, got):
                if m:
                    out_t.append(t)
                    out_f.append(k)
                    out_uv.append(uv)
        if k % 150 == 0:
            say(f"  re-matching: frame {k}, {len(out_t)} new observations so far", flush=True)
    return np.array(out_t, int), np.array(out_f, int), np.array(out_uv, float).reshape(-1, 2)


def grid_index(p):
    (x0, _), (y0, _), (h0, _) = REGION
    return np.floor(np.stack([(p[:, 0] - x0) / VOX, (p[:, 1] - y0) / VOX, (p[:, 2] - SURFACE_Z - h0) / VOX], 1)).astype(int)


def main(run: Path) -> None:
    w, T_fc, _ = cameras(run)
    rows = [json.loads(l) for l in open(run / "frames.jsonl")]
    J = np.array([r["joints"] for r in rows])
    tic = time.time()
    tracks = follow(run, rows)
    obs = Obs(tracks, w)
    off, T = retime(obs, J, T_fc)
    P, e, rms = obs.place(T)
    sig = sigma_long(obs, T, P)
    ok = (rms < GOOD_PX) & (sig < SIGMA_MM / 1000) & in_region(P)
    n_obs0 = np.bincount(obs.tid, minlength=obs.m)
    before = dict(points=int(ok.sum()), obs=float(np.median(n_obs0[ok])), sigma=float(np.median(sig[ok]) * 1000))
    print(f"frame-to-frame: {ok.sum()} points count, median {before['obs']:.0f} views each, "
          f"depth sigma {before['sigma']:.2f} mm ({time.time() - tic:.0f} s)", flush=True)

    grays = [cv2.cvtColor(cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg")), cv2.COLOR_BGR2GRAY) for r in rows]
    # any point with a fair estimate is worth predicting; the re-match itself is the test
    cand = (rms < 2 * GOOD_PX) & (sig < 3 * SIGMA_MM / 1000) & in_region(P)
    nt, nf, nuv = rematch(obs, T, P, cand, grays)
    print(f"re-matched against every view: {len(nt)} new observations ({time.time() - tic:.0f} s)", flush=True)
    # fold them in (same rays as Obs would build)
    xd, yd = (nuv[:, 0] - obs.cx) / obs.fx, (nuv[:, 1] - obs.cy) / obs.fx
    x, y = xd, yd
    for _ in range(8):
        dd = 1 + obs.k1 * (x * x + y * y)
        x, y = xd / dd, yd / dd
    r = np.stack([x, y, np.ones_like(x)], 1)
    obs.tid = np.concatenate([obs.tid, nt])
    obs.fr = np.concatenate([obs.fr, nf])
    obs.uv = np.concatenate([obs.uv, nuv])
    obs.ray = np.concatenate([obs.ray, r / np.linalg.norm(r, axis=1, keepdims=True)])
    P, e, rms = obs.place(T)
    sig = sigma_long(obs, T, P)
    # an observation more than 3 px from its point is dropped from the evidence
    keep_obs = e < 3.0
    ok = (rms < GOOD_PX) & (sig < SIGMA_MM / 1000) & in_region(P)
    n_obs1 = np.bincount(obs.tid[keep_obs], minlength=obs.m)
    after = dict(points=int(ok.sum()), obs=float(np.median(n_obs1[ok])), sigma=float(np.median(sig[ok]) * 1000))
    print(f"all views:      {ok.sum()} points count, median {after['obs']:.0f} views each, "
          f"depth sigma {after['sigma']:.2f} mm", flush=True)

    # quality and colour per point; the order the evidence arrives in: frame by frame
    q = np.clip(VOX / np.maximum(sig, 1e-6), 0, 1)
    rgb = np.zeros((obs.m, 3), np.uint8)
    shape = tuple(int(np.ceil((b - a) / VOX)) for a, b in REGION)
    L = np.zeros(shape, np.float32)
    RGB = np.zeros(shape + (3,), np.uint8)
    use = keep_obs & ok[obs.tid]
    order = np.argsort(obs.fr, kind="stable")
    starts = np.searchsorted(obs.fr[order], np.arange(len(rows) + 1))
    gi_point = grid_index(P[ok])
    point_vox = np.full((obs.m, 3), -1)
    point_vox[ok] = gi_point

    rr.init("fog3d", spawn=False)
    rr.save(str(run / "fog3d.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    (x0, x1), (y0, y1), _ = REGION
    rr.log("world/table", rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, SURFACE_Z - 0.001]],
                                    half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, 0.0005]], colors=[[200, 200, 200]]), static=True)
    rr.log("plots/something", rr.SeriesLines(colors=[[200, 60, 40]], names="voxels: something"), static=True)
    rr.log("plots/nothing", rr.SeriesLines(colors=[[60, 110, 200]], names="voxels: nothing (/100)"), static=True)
    K = np.array([[w["fx"], 0, 424], [0, w["fx"], 240], [0, 0, 1]])
    t0 = rows[0]["t"]
    steps = np.arange(0, 0.40, VOX / 2)  # along each ray, from the camera
    counted = np.zeros(obs.m, bool)
    for k, row in enumerate(rows):
        img = cv2.imread(str(run / "frames" / f"{row['n']:05d}.jpg"))
        o_ = order[starts[k]:starts[k + 1]]
        o_ = o_[use[o_]]
        tids = obs.tid[o_]
        newc = ~counted[tids]
        if newc.any():
            uvn = obs.uv[o_][newc]
            c = img[np.clip(uvn[:, 1].astype(int), 0, 479), np.clip(uvn[:, 0].astype(int), 0, 847), ::-1]
            rgb[tids[newc]] = c
            counted[tids] = True
        if len(o_):
            cam_o = T[k][:3, 3]
            Pt = P[tids]
            dist = np.linalg.norm(Pt - cam_o, axis=1)
            dirs = (Pt - cam_o) / dist[:, None]
            stop = dist - 3 * np.maximum(sig[tids], VOX)
            pts = cam_o[None, None] + dirs[:, None] * steps[None, :, None]
            valid = steps[None, :] < stop[:, None]
            gi = grid_index(pts[valid])
            rid = np.nonzero(valid)[0]
            inb = np.all((gi >= 0) & (gi < np.array(shape)), 1)
            gi, rid = gi[inb], rid[inb]
            # once per ray per voxel
            flat = np.ravel_multi_index(gi.T, shape)
            u = np.unique(np.stack([rid, flat], 1), axis=0)[:, 1]
            np.add.at(L.reshape(-1), u, FREE)
            pv = point_vox[tids]
            inb = np.all((pv >= 0) & (pv < np.array(shape)), 1)
            fl = np.ravel_multi_index(pv[inb].T, shape)
            np.add.at(L.reshape(-1), fl, HIT * q[tids[inb]])
            RGB.reshape(-1, 3)[fl] = rgb[tids[inb]]

        rr.set_time("frame", sequence=k)
        rr.set_time("seconds", duration=row["t"] - t0)
        Tk = T[k]
        rr.log("world/wrist", rr.Transform3D(translation=Tk[:3, 3], mat3x3=Tk[:3, :3]))
        rr.log("world/wrist/cam", rr.Pinhole(image_from_camera=K, resolution=[848, 480], image_plane_distance=0.05))
        rr.log("world/wrist/cam", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        all_k = order[starts[k]:starts[k + 1]]
        tk = obs.tid[all_k]
        good = ok[tk] & keep_obs[all_k]
        c2 = np.full((len(tk), 3), 150, np.uint8)
        c2[good] = height_colours(P[tk[good]])
        rr.log("world/wrist/cam/points", rr.Points2D(obs.uv[all_k], colors=c2, radii=1.5))
        sv = np.argwhere(L >= SURE)
        cen = np.array([x0, y0, SURFACE_Z + REGION[2][0]]) + (sv + 0.5) * VOX
        rr.log("world/fog/something", rr.Points3D(cen, colors=RGB[tuple(sv.T)], radii=VOX * 0.6))
        n_nothing = int((L <= -SURE).sum())
        rr.log("plots/something", rr.Scalars(len(sv)))
        rr.log("plots/nothing", rr.Scalars(n_nothing / 100))
        if k % FOG_EVERY == 0 or k == len(rows) - 1:
            # 6 mm blocks with any sure-nothing voxel, and the top-down map
            sh = tuple(s // 3 for s in shape)
            Lc = L[:sh[0] * 3, :sh[1] * 3, :sh[2] * 3].reshape(sh[0], 3, sh[1], 3, sh[2], 3)
            nb = np.argwhere((Lc <= -SURE).any((1, 3, 5)))
            rr.log("world/fog/nothing", rr.Points3D(np.array([x0, y0, SURFACE_Z + REGION[2][0]]) + (nb + 0.5) * 3 * VOX,
                                                    colors=[[90, 140, 230, 60]], radii=VOX * 0.8))
            some = (L >= SURE).any(2)
            # nothing seen through down to near the table: the lowest sure-nothing voxel within 10 mm of it
            low = (L[:, :, :8] <= -SURE).any(2)
            m = np.full(some.shape + (3,), 255, np.uint8)
            m[low] = (150, 185, 240)
            m[some] = (210, 60, 40)
            # rows = x (far at the top, as the wrist sees it), columns = y
            rr.log("fog_map", rr.Image(np.ascontiguousarray(m[:, ::-1])))
        if k % 100 == 0:
            print(f"frame {k:4d}: {len(sv)} voxels something, {n_nothing} nothing", flush=True)
    np.save(run / "fog_L.npy", L)
    print(f"before re-matching: {before}\nafter:              {after}\n"
          f"something {int((L >= SURE).sum())}, nothing {int((L <= -SURE).sum())}, fog {int((np.abs(L) < SURE).sum())} "
          f"voxels ({time.time() - tic:.0f} s); saved {run / 'fog3d.rrd'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
