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
    4. Edges too (lab/edges3d.py): every frame the camera has moved REF_MOVE_MM or turned REF_TURN_DEG since the
       last one used, each colour-edge pixel is tried along its ray against far-apart frames; kept only with a
       clear, narrow match peak.
    5. The depth camera, at the stops, once calibrated against the video: its mount (and depth scale) is fitted so
       its depth agrees with the video's points, and what is left over is its noise, which sets its weight.
    Every measurement adds HIT * quality where it says something is (quality = VOX / its sigma, at most 1; the
    depth camera's is scaled again by DEPTH_TRUST: its errors repeat from stop to stop) and FREE * quality to every
    voxel its ray passed through on the way there, stopping 3 sigma short.

Timeline "frame", in Rerun:
    world/wrist/cam        the picture and its points: grey still unsure, coloured by height once counted
    world/edges/now        this frame's edge points (by height); world/depthcam/now, the depth camera at a stop
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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration.geometry import SURFACE_Z, flange_pose, load
from lab.edges3d import Lens, edge_points
from lab.scene3d import cameras
from lab.tracks3d import Obs, follow, height_colours, in_region, poses, retime

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
REF_MOVE_MM, REF_TURN_DEG = 4.0, 2.0
DEPTH_STRIDE = 4        # every 4th depth pixel, each way
DEPTH_TRUST = 0.5


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


def video_points(run: Path, rows, w, J, T_fc) -> dict:
    """Tracks, re-timing, re-matching against every view: the video's points, cached in fog_points.npz."""
    cache = run / "fog_points.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"video points from {cache.name} (delete it to redo)")
        return {k: z[k] for k in z.files}
    tic = time.time()
    tracks = follow(run, rows)
    obs = Obs(tracks, w)
    off, T = retime(obs, J, T_fc)
    P, e, rms = obs.place(T)
    sig = sigma_long(obs, T, P)
    ok = (rms < GOOD_PX) & (sig < SIGMA_MM / 1000) & in_region(P)
    n_obs0 = np.bincount(obs.tid, minlength=obs.m)
    before = dict(points=int(ok.sum()), views=float(np.median(n_obs0[ok])), sigma_mm=float(np.median(sig[ok]) * 1000))
    print(f"frame-to-frame: {ok.sum()} points count, median {before['views']:.0f} views each, "
          f"depth sigma {before['sigma_mm']:.2f} mm ({time.time() - tic:.0f} s)", flush=True)
    grays = [cv2.cvtColor(cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg")), cv2.COLOR_BGR2GRAY) for r in rows]
    # any point with a fair estimate is worth predicting; the re-match itself is the test
    cand = (rms < 2 * GOOD_PX) & (sig < 3 * SIGMA_MM / 1000) & in_region(P)
    nt, nf, nuv = rematch(obs, T, P, cand, grays)
    print(f"re-matched against every view: {len(nt)} new observations ({time.time() - tic:.0f} s)", flush=True)
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
    keep_obs = e < 3.0  # an observation more than 3 px from its point is dropped from the evidence
    ok = (rms < GOOD_PX) & (sig < SIGMA_MM / 1000) & in_region(P)
    n_obs1 = np.bincount(obs.tid[keep_obs], minlength=obs.m)
    after = dict(points=int(ok.sum()), views=float(np.median(n_obs1[ok])), sigma_mm=float(np.median(sig[ok]) * 1000))
    print(f"all views:      {ok.sum()} points count, median {after['views']:.0f} views each, "
          f"depth sigma {after['sigma_mm']:.2f} mm", flush=True)
    out = dict(off=off, T=T, P=P, sig=sig, ok=ok, tid=obs.tid, fr=obs.fr, uv=obs.uv, keep=keep_obs,
               before=json.dumps(before), after=json.dumps(after))
    np.savez(cache, **out)
    return out


def edge_stage(run: Path, rows, w, T) -> dict:
    """Edge points from every frame that moved or turned enough since the last one used; cached in fog_edges.npz."""
    cache = run / "fog_edges.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"edge points from {cache.name} (delete it to redo)")
        return {k: z[k] for k in z.files}
    tic = time.time()
    refs, last = [], None
    for k in range(len(rows)):
        if last is None or np.linalg.norm(T[k][:3, 3] - T[last][:3, 3]) * 1000 >= REF_MOVE_MM or \
                np.degrees(Rotation.from_matrix(T[k][:3, :3] @ T[last][:3, :3].T).magnitude()) >= REF_TURN_DEG:
            refs.append(k)
            last = k
    cols = [cv2.imread(str(run / "frames" / f"{r['n']:05d}.jpg")) for r in rows]
    grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32) for c in cols]
    lens = Lens(w)
    out = {"k": [], "P": [], "sig": [], "rgb": []}
    tot = kept = 0
    for i, k in enumerate(refs):
        P, s, c, st = edge_points(lens, grays, cols, T, k)
        out["k"].append(np.full(len(P), k))
        out["P"].append(P)
        out["sig"].append(s)
        out["rgb"].append(c)
        tot += st["edges"]
        kept += st["kept"]
        if i % 25 == 0:
            print(f"  edges: frame {k} ({i + 1} of {len(refs)}), {kept} of {tot} edge pixels kept so far "
                  f"({time.time() - tic:.0f} s)", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["refs"] = np.array(refs)
    np.savez(cache, **out)
    print(f"edges: {len(refs)} frames used, {kept} of {tot} edge pixels kept ({time.time() - tic:.0f} s)")
    return out


def depth_stage(run: Path, rows, P_video) -> dict:
    """The depth camera at the stops, its mount and depth scale fitted to the video's points."""
    cam = load()["wrist_depth"]
    t_rows = np.array([r["t"] for r in rows])
    stops = []
    for f in sorted(run.glob("stops/*.npz")):
        z = np.load(f)
        k = int(np.argmin(np.abs(t_rows - float(z["t"]))))
        stops.append((k, np.asarray(z["joints"], float), z["depth_wrist"].astype(np.float64)))
    F = [flange_pose(j) for _, j, _ in stops]

    def unpack(p):
        D = np.eye(4)
        D[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
        D[:3, 3] = p[3:6] / 1000
        return D, 1 + p[6]

    def project(p):
        """Per stop: the video points' predicted depth (m) and the depth camera's reading (m, 0 = none) there."""
        D, scale = unpack(p)
        out = []
        for (k, _, dep), Fs in zip(stops, F):
            Tc = Fs @ cam["T"] @ D
            pc = (P_video - Tc[:3, 3]) @ Tc[:3, :3]
            z = np.maximum(pc[:, 2], 1e-3)
            u = np.rint(cam["fx"] * pc[:, 0] / z + cam["cx"]).astype(int)
            v = np.rint(cam["fy"] * pc[:, 1] / z + cam["cy"]).astype(int)
            inside = (pc[:, 2] > 0.05) & (u >= 0) & (u < dep.shape[1]) & (v >= 0) & (v < dep.shape[0])
            d = np.zeros(len(pc))
            d[inside] = dep[v[inside], u[inside]] / 1000 * scale
            out.append((pc[:, 2], d))
        return out

    # the pairs (stop, point) compared: those the depth camera read at the start (a fixed set, so the fit is smooth)
    pairs = [d > 0 for _, d in project(np.zeros(7))]

    def resid(p, ret=False):
        r = np.concatenate([np.where(d[m] > 0, d[m] - z[m], 0.0) for (z, d), m in zip(project(p), pairs)]) * 1000
        return r if ret else np.clip(r, -25, 25)

    r0 = resid(np.zeros(7), True)
    s = least_squares(resid, np.zeros(7), loss="soft_l1", f_scale=2.0, x_scale=[0.01] * 3 + [1] * 3 + [0.01])
    r1 = resid(s.x, True)
    D, scale = unpack(s.x)
    mad = lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))
    stats = dict(before_median_mm=float(np.median(r0)), before_spread_mm=float(mad(r0)),
                 after_median_mm=float(np.median(r1)), after_spread_mm=float(mad(r1)), points=len(r1),
                 turned_deg=float(np.degrees(np.linalg.norm(s.x[:3]))), moved_mm=s.x[3:6].round(1).tolist(),
                 scale=float(scale))
    print(f"depth camera against the video's points: off by {stats['before_median_mm']:+.1f} mm "
          f"(spread {stats['before_spread_mm']:.1f}) -> {stats['after_median_mm']:+.1f} mm (spread "
          f"{stats['after_spread_mm']:.1f}); mount turned {stats['turned_deg']:.2f} deg, moved {stats['moved_mm']} mm, "
          f"depth x {scale:.4f}", flush=True)
    views = []
    for (k, _, dep), Fs in zip(stops, F):
        Tc = Fs @ cam["T"] @ D
        d = dep[::DEPTH_STRIDE, ::DEPTH_STRIDE]
        v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * DEPTH_STRIDE
        m = d > 0
        z = d[m] / 1000 * scale
        pc = np.stack([(u[m] - cam["cx"]) / cam["fx"] * z, (v[m] - cam["cy"]) / cam["fy"] * z, z], -1)
        views.append((k, Tc[:3, 3], pc @ Tc[:3, :3].T + Tc[:3, 3], z))
    return {"views": views, "sigma": stats["after_spread_mm"] / 1000, "stats": stats}


def vote(L, RGB, o, P, sig, w, rgb=None):
    """Rays from camera centre(s) o to points P (sigma along the ray): +HIT*w at P, FREE*w on the way, 3 sigma short."""
    shape = np.array(L.shape)
    Lf, Cf = L.reshape(-1), RGB.reshape(-1, 3)
    o = np.broadcast_to(np.asarray(o, float), P.shape)
    w = np.broadcast_to(np.asarray(w, float), (len(P),))
    sig = np.broadcast_to(np.asarray(sig, float), (len(P),))
    for a in range(0, len(P), 4000):
        sl = slice(a, a + 4000)
        d = P[sl] - o[sl]
        dist = np.linalg.norm(d, axis=1)
        dirs = d / np.maximum(dist, 1e-6)[:, None]
        stop = dist - 3 * np.maximum(sig[sl], VOX)
        steps = np.arange(0, max(stop.max(), 0) + VOX, VOX / 2)
        gi = grid_index((o[sl][:, None] + dirs[:, None] * steps[None, :, None]).reshape(-1, 3)).reshape(len(d), -1, 3)
        valid = (steps[None] < stop[:, None]) & np.all((gi >= 0) & (gi < shape), 2)
        flat = np.where(valid, np.ravel_multi_index(np.moveaxis(np.clip(gi, 0, shape - 1), 2, 0), tuple(shape)), -1)
        first = valid & np.concatenate([np.ones((len(d), 1), bool), flat[:, 1:] != flat[:, :-1]], 1)
        np.add.at(Lf, flat[first], FREE * np.broadcast_to(w[sl][:, None], first.shape)[first])
    gi = grid_index(P)
    inb = np.all((gi >= 0) & (gi < shape), 1)
    fl = np.ravel_multi_index(gi[inb].T, tuple(shape))
    np.add.at(Lf, fl, HIT * w[inb])
    if rgb is not None:
        Cf[fl] = rgb[inb]


def main(run: Path) -> None:
    w, T_fc, _ = cameras(run)
    rows = [json.loads(l) for l in open(run / "frames.jsonl")]
    J = np.array([r["joints"] for r in rows])
    tic = time.time()
    V = video_points(run, rows, w, J, T_fc)
    T, P, sig, ok, keep = V["T"], V["P"], V["sig"], V["ok"], V["keep"]
    tid, fr, uv = V["tid"], V["fr"], V["uv"]
    E = edge_stage(run, rows, w, T)
    Dc = depth_stage(run, rows, P[ok])
    depth_at = {}
    for k, o, pts, z in Dc["views"]:
        depth_at.setdefault(k, []).append((o, pts, z))
    zmed = np.median(np.concatenate([z for _, _, _, z in Dc["views"]]))

    q = np.clip(VOX / np.maximum(sig, 1e-6), 0, 1)
    m = len(P)
    rgb = np.zeros((m, 3), np.uint8)
    counted = np.zeros(m, bool)
    shape = tuple(int(np.ceil((b - a) / VOX)) for a, b in REGION)
    L = np.zeros(shape, np.float32)
    RGB = np.zeros(shape + (3,), np.uint8)
    use = keep & ok[tid]
    order = np.argsort(fr, kind="stable")
    starts = np.searchsorted(fr[order], np.arange(len(rows) + 1))
    e_order = np.argsort(E["k"], kind="stable")
    e_starts = np.searchsorted(E["k"][e_order], np.arange(len(rows) + 1))

    rr.init("fog3d", spawn=False)
    rr.save(str(run / "fog3d.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    (x0, x1), (y0, y1), _ = REGION
    origin = np.array([x0, y0, SURFACE_Z + REGION[2][0]])
    rr.log("world/table", rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, SURFACE_Z - 0.001]],
                                    half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, 0.0005]], colors=[[200, 200, 200]]), static=True)
    rr.log("plots/something", rr.SeriesLines(colors=[[200, 60, 40]], names="voxels: something"), static=True)
    rr.log("plots/nothing", rr.SeriesLines(colors=[[60, 110, 200]], names="voxels: nothing (/100)"), static=True)
    K = np.array([[w["fx"], 0, 424], [0, w["fx"], 240], [0, 0, 1]])
    t0 = rows[0]["t"]
    for k, row in enumerate(rows):
        img = cv2.imread(str(run / "frames" / f"{row['n']:05d}.jpg"))
        o_ = order[starts[k]:starts[k + 1]]
        o_ = o_[use[o_]]
        tids = tid[o_]
        newc = ~counted[tids]
        if newc.any():
            uvn = uv[o_][newc]
            rgb[tids[newc]] = img[np.clip(uvn[:, 1].astype(int), 0, 479), np.clip(uvn[:, 0].astype(int), 0, 847), ::-1]
            counted[tids] = True
        if len(o_):
            vote(L, RGB, T[k][:3, 3], P[tids], sig[tids], q[tids], rgb[tids])
        ei = e_order[e_starts[k]:e_starts[k + 1]]
        if len(ei):
            vote(L, RGB, T[k][:3, 3], E["P"][ei], E["sig"][ei], np.clip(VOX / E["sig"][ei], 0, 1), E["rgb"][ei])
        for o, pts, z in depth_at.get(k, []):
            s_o = Dc["sigma"] * (z / zmed) ** 2
            vote(L, RGB, o, pts, s_o, DEPTH_TRUST * np.clip(VOX / s_o, 0, 1))

        rr.set_time("frame", sequence=k)
        rr.set_time("seconds", duration=row["t"] - t0)
        Tk = T[k]
        rr.log("world/wrist", rr.Transform3D(translation=Tk[:3, 3], mat3x3=Tk[:3, :3]))
        rr.log("world/wrist/cam", rr.Pinhole(image_from_camera=K, resolution=[848, 480], image_plane_distance=0.05))
        rr.log("world/wrist/cam", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        all_k = order[starts[k]:starts[k + 1]]
        tk = tid[all_k]
        good = ok[tk] & keep[all_k]
        c2 = np.full((len(tk), 3), 150, np.uint8)
        c2[good] = height_colours(P[tk[good]])
        rr.log("world/wrist/cam/points", rr.Points2D(uv[all_k], colors=c2, radii=1.5))
        if len(ei):
            rr.log("world/edges/now", rr.Points3D(E["P"][ei], colors=height_colours(E["P"][ei]), radii=0.0008))
        for o, pts, z in depth_at.get(k, []):
            rr.log("world/depthcam/now", rr.Points3D(pts[::3], colors=height_colours(pts[::3]), radii=0.0008))
        sv = np.argwhere(L >= SURE)
        rr.log("world/fog/something", rr.Points3D(origin + (sv + 0.5) * VOX, colors=RGB[tuple(sv.T)], radii=VOX * 0.6))
        n_nothing = int((L <= -SURE).sum())
        rr.log("plots/something", rr.Scalars(len(sv)))
        rr.log("plots/nothing", rr.Scalars(n_nothing / 100))
        if k % FOG_EVERY == 0 or k == len(rows) - 1:
            # 6 mm blocks with any sure-nothing voxel, and the top-down map
            sh = tuple(s // 3 for s in shape)
            Lc = L[:sh[0] * 3, :sh[1] * 3, :sh[2] * 3].reshape(sh[0], 3, sh[1], 3, sh[2], 3)
            nb = np.argwhere((Lc <= -SURE).any((1, 3, 5)))
            rr.log("world/fog/nothing", rr.Points3D(origin + (nb + 0.5) * 3 * VOX, colors=[[90, 140, 230, 60]],
                                                    radii=VOX * 0.8))
            some = (L >= SURE).any(2)
            low = (L[:, :, :8] <= -SURE).any(2)  # seen through down to within 10 mm of the table
            mp = np.full(some.shape + (3,), 255, np.uint8)
            mp[low] = (150, 185, 240)
            mp[some] = (210, 60, 40)
            rr.log("fog_map", rr.Image(np.ascontiguousarray(mp[:, ::-1])))  # rows x (far at the top), columns y
        if k % 100 == 0:
            print(f"frame {k:4d}: {len(sv)} voxels something, {n_nothing} nothing ({time.time() - tic:.0f} s)", flush=True)
    np.save(run / "fog_L.npy", L)
    np.save(run / "fog_rgb.npy", RGB)
    print(f"video points before re-matching {V['before']}, after {V['after']}\ndepth camera {Dc['stats']}\n"
          f"something {int((L >= SURE).sum())}, nothing {int((L <= -SURE).sum())}, fog {int((np.abs(L) < SURE).sum())} "
          f"voxels ({time.time() - tic:.0f} s); saved {run / 'fog3d.rrd'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
