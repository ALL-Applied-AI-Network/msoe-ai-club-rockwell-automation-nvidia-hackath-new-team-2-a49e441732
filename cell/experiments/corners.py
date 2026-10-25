"""Corner tracking + triangulation with the arm's known camera poses, vs raw depth, at the corners (where the depth
camera is worst: edges smear and drop out).

    uv run python -m experiments.corners            # sim (scored against the true surfaces) and real (wrist video)

Sim: the wrist slides 10 cm past the objects as a 40-frame video. Corners (Shi-Tomasi) are tracked frame to frame
(pyramidal Lucas-Kanade, kept only if tracking back lands within 0.5 px of where it started), then each track is
triangulated from all its frames with the known camera poses. Each corner's 3D point is scored against the truth, next
to what the raw depth camera says at the same pixel of the first frame.
Real: the same on the lab's wrist video; no truth meshes, but nothing can be below the table.
Writes experiments/corners.png and prints the numbers.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
from calibration.kinematics import tool_ik, wpr_of
from lab import views
from lab.sensor import realistic_depth
from lab.synth import Renderer
from lab.trial import build, load_trial

HERE = Path(__file__).parent
FB_PX = 0.5          # forward-backward tracking error allowed
MIN_FRAMES = 8       # a track must last this many frames to be triangulated
LAG_FRAMES = 1       # real recordings: a wrist image matches the joints logged one frame (~100 ms) earlier; the best
                     # fit fell between 60 and 140 ms on different stretches (jittery timing)


def render_rgb(r: Renderer, K: np.ndarray, size, T: np.ndarray, ss: int = 2, rng=None) -> np.ndarray:
    """Colour from camera pose T with pinhole K, rendered at ss x the resolution and averaged down (a real lens's
    soft edges, not jagged ones), plus a little sensor noise."""
    w, h = size
    v, u = np.mgrid[0:h * ss, 0:w * ss] + 0.5
    d = np.stack([(u / ss - K[0, 2]) / K[0, 0], (v / ss - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1) @ T[:3, :3].T
    hit = r._cast(T[:3, 3], d)
    gid = hit["geometry_ids"].numpy().astype(np.int64)
    n = hit["primitive_normals"].numpy()
    light = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])
    shade = 0.35 + 0.65 * np.abs(n @ light)
    img = np.full((*gid.shape, 3), (60, 60, 65), float)
    ok = gid < len(r.colors)
    img[ok] = np.array(r.colors)[gid[ok]] * shade[ok, None]
    img = img.reshape(h, ss, w, ss, 3).mean((1, 3))
    if rng is not None:
        img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def track(imgs: list[np.ndarray], max_corners: int = 600) -> list[dict]:
    """Corner tracks over the frames: {frame: (u, v)} each. New corners are found every 5 frames where none are."""
    grays = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in imgs]
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    tracks, live = [], []
    for k, g in enumerate(grays):
        if k:
            if live:
                p0 = np.float32([t[k - 1] for t in live]).reshape(-1, 1, 2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[k - 1], g, p0, None, **lk)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, grays[k - 1], p1, None, **lk)
                good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < FB_PX)
                keep = []
                for t, p, ok in zip(live, p1[:, 0], good):
                    if ok and 0 <= p[0] < g.shape[1] and 0 <= p[1] < g.shape[0]:
                        t[k] = p.astype(float)
                        keep.append(t)
                live = keep
        if k % 5 == 0:
            mask = np.full(g.shape, 255, np.uint8)
            for t in live:
                cv2.circle(mask, tuple(int(c) for c in t[k]), 8, 0, -1)
            pts = cv2.goodFeaturesToTrack(g, max_corners, 0.02, 6, mask=mask, blockSize=5)
            if pts is not None:
                pts = cv2.cornerSubPix(g, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                for p in pts[:, 0]:
                    t = {k: p.astype(float)}
                    tracks.append(t)
                    live.append(t)
    return [t for t in tracks if len(t) >= MIN_FRAMES]


def triangulate(t: dict, K: np.ndarray, poses) -> tuple[np.ndarray, float]:
    """The 3D point (base frame) seen at t's pixels from the known camera poses (linear least squares), and the mean
    reprojection error in px."""
    A = []
    for k, (u, v) in t.items():
        P = K @ np.linalg.inv(poses[k])[:3]
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    X = X[:3] / X[3]
    err = []
    for k, (u, v) in t.items():
        pc = np.linalg.inv(poses[k]) @ np.append(X, 1)
        err.append(np.hypot(K[0, 0] * pc[0] / pc[2] + K[0, 2] - u, K[1, 1] * pc[1] / pc[2] + K[1, 2] - v))
    return X, float(np.mean(err))


def sim(n: int = 40, travel: float = 0.10, seed: int = 0):
    cell = build(load_trial("sort_mixed"), seed)
    objs = cell.object_meshes()
    cams = load()
    cam = cams["wrist_depth"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    r = Renderer(objs, cams)
    target = np.mean([o["mesh"].centroid for o in objs], 0)
    target[2] = SURFACE_Z + 0.01
    T_ct = views._cam_in_tool(cam)
    rng = np.random.default_rng(seed)
    q = np.array(cell.q, float)
    imgs, poses, depth0, truth0 = [], [], None, None
    for k, s in enumerate(np.linspace(-travel / 2, travel / 2, n)):
        e = np.radians(20)
        Tc = views._look_at(target + [0, s, 0], np.array([np.sin(e), 0, -np.cos(e)]), 0.23, 0.0)
        Tt = Tc @ np.linalg.inv(T_ct)
        q = tool_ik(Tt[:3, 3], wpr_of(Tt[:3, :3]), q)
        T = flange_pose(q) @ cam["T"]
        imgs.append(render_rgb(r, K, cam["size"], T, rng=rng))
        poses.append(T)
        if k == 0:
            clean = r.depth("wrist", joints=q)
            truth0, depth0 = clean / 1000.0, realistic_depth(clean, rng) / 1000.0
    tracks = track(imgs)
    import open3d as o3d
    from lab.synth import table_mesh
    scene = o3d.t.geometry.RaycastingScene()
    for m in [table_mesh()] + [o["mesh"] for o in objs]:
        scene.add_triangles(o3d.core.Tensor(m.vertices, o3d.core.float32), o3d.core.Tensor(m.faces, o3d.core.uint32))
    dist = lambda X: np.linalg.norm(scene.compute_closest_points(o3d.core.Tensor(np.float32(X)))["points"].numpy() - X, axis=1)
    tri, raw, reproj, depth_edge = [], [], [], []
    for t in tracks:
        if 0 not in t:
            continue
        X, e = triangulate(t, K, poses)
        u, v = t[0]
        z = depth0[int(round(v)), int(round(u))]
        tri.append(X)
        reproj.append(e)
        raw.append(np.full(3, np.nan) if z == 0 else
                   (poses[0] @ np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, 1]))[:3])
        win = truth0[max(0, int(v) - 2):int(v) + 3, max(0, int(u) - 2):int(u) + 3]
        depth_edge.append(win.max() - win.min() > 0.006)  # a depth edge at this corner (an object's outline)
    tri, raw, reproj, edge = np.array(tri), np.array(raw), np.array(reproj), np.array(depth_edge)
    d_tri = dist(tri)
    has = ~np.isnan(raw[:, 0])
    d_raw = np.full(len(raw), np.nan)
    d_raw[has] = dist(raw[has])
    return {"tracks": len(tracks), "n": len(tri), "d_tri": d_tri, "d_raw": d_raw, "edge": edge, "reproj": reproj,
            "img": imgs[0], "tracks_list": [t for t in tracks if 0 in t], "travel": travel, "frames": n}


def real(episode: int = 2, frames: int = 60):
    """Lab wrist video + joints: corners triangulated from the same stretch capture.py picked for MapAnything."""
    import av
    import pandas as pd
    root = Path.home() / ".cache/huggingface/lerobot/manipbench/r20261024-1f2d05"
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    ep = df[df.episode_index == episode].sort_values("frame_index")
    cams = load()
    cam = cams["wrist"]
    poses = [flange_pose(np.asarray(s[:6], float)) @ cam["T"] for s in ep["observation.state"]]
    xyz = np.array([P[:3, 3] for P in poses])
    best, i0 = 0, 0
    for i in range(LAG_FRAMES, len(xyz) - frames, 5):
        d = np.linalg.norm(xyz[i + frames] - xyz[i])
        if d > best and xyz[i:i + frames, 2].max() < SURFACE_Z + 0.45:
            best, i0 = d, i
    want = range(i0, i0 + frames)
    vid = sorted((root / "videos/observation.images.wrist").rglob("*.mp4"))[episode]
    imgs = []
    with av.open(str(vid)) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k in want:
                imgs.append(fr.to_ndarray(format="rgb24"))
            if k >= want[-1]:
                break
    H, W = imgs[0].shape[:2]
    s = W / cam["size"][0]
    K = np.array([[cam["fx"] * s, 0, W / 2], [0, cam["fx"] * s, H / 2], [0, 0, 1]])
    P = [poses[i - LAG_FRAMES] for i in want]  # each wrist image is ~100 ms older than the joints logged with it
    tracks = track(imgs)
    pts, rep = [], []
    for t in tracks:
        tt = {k: v for k, v in t.items()}
        X, e = triangulate(tt, K, P)
        pts.append(X)
        rep.append(e)
    return {"pts": np.array(pts), "reproj": np.array(rep), "travel": best, "n": len(tracks), "img": imgs[0],
            "tracks_list": [t for t in tracks if 0 in t]}


def plot(S, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5), facecolor="#111317")
    for a in ax.flat:
        a.set_facecolor("#171a20"); a.tick_params(colors="w")
        for sp in a.spines.values():
            sp.set_color("#2a2f38")
    a = ax[0, 0]
    a.imshow(S["img"])
    good = S["reproj"] < 1.0
    for t, d, g in zip(S["tracks_list"], S["d_tri"], good):
        if not g:
            continue
        uv = np.array([t[k] for k in sorted(t)])
        c = plt.cm.RdYlGn_r(min(d / 0.004, 1.0))
        a.plot(uv[:, 0], uv[:, 1], color=c, lw=0.8)
        a.plot(*uv[0], "o", ms=2.5, color=c)
    a.set_title(f"Sim: corner tracks over a {S['travel'] * 100:.0f} cm sweep ({S['frames']} frames), coloured by 3D error (green < 1 mm, red ≥ 4 mm)",
                color="w", fontsize=10); a.axis("off")
    a = ax[0, 1]
    bins = np.linspace(0, 12, 61)
    for sel, name, ls in ((S["edge"], "at depth edges (object outlines)", "-"), (~S["edge"], "elsewhere (on a face)", ":")):
        tri = S["d_tri"][sel & good] * 1000
        raw = S["d_raw"][sel & good] * 1000
        raw_ok = raw[~np.isnan(raw)]
        a.hist(tri, bins, histtype="step", lw=2, ls=ls, color="#7ee2a0", density=True,
               label=f"tracked corners, {name}: median {np.median(tri):.1f} mm, >4 mm {np.mean(tri > 4) * 100:.0f}%")
        a.hist(raw_ok, bins, histtype="step", lw=2, ls=ls, color="#f2cc60", density=True,
               label=f"raw depth at the same pixel, {name}: median {np.median(raw_ok):.1f} mm, >4 mm {np.mean(raw_ok > 4) * 100:.0f}%,"
                     f" no reading {np.mean(np.isnan(raw)) * 100:.0f}%")
    a.set_xlabel("distance to the true surface (mm)", color="w"); a.set_ylabel("share", color="w")
    a.legend(facecolor="#171a20", edgecolor="#2a2f38", labelcolor="w", fontsize=8)
    a.set_title("Sim: tracked + triangulated corners vs the depth camera, at the same corners", color="w", fontsize=10)
    a = ax[1, 0]
    a.imshow(R["img"])
    for t in R["tracks_list"]:
        uv = np.array([t[k] for k in sorted(t)])
        a.plot(uv[:, 0], uv[:, 1], color="#9cc4ff", lw=0.6)
    a.set_title(f"Real wrist video: {R['n']} corner tracks over {R['travel'] * 100:.0f} cm of camera travel (first frame shown)", color="w", fontsize=10)
    a.axis("off")
    a = ax[1, 1]
    z = (R["pts"][:, 2] - SURFACE_Z) * 1000
    ok = R["reproj"] < 2.0
    a.hist(z[ok], bins=np.linspace(-60, 80, 71), color="#9cc4ff", alpha=0.85)
    a.axvline(0, color="w", lw=1, ls="--"); a.axvspan(0, 35, color="#7ee2a0", alpha=0.08)
    a.set_xlabel("height above the table (mm): 0 = table top; parts are at most ~35 mm tall (green band)", color="w")
    a.set_ylabel("corners", color="w")
    a.set_title(f"Real: where the triangulated corners land (reprojection < 2 px: {ok.sum()} of {len(z)})", color="w", fontsize=10)
    plt.tight_layout()
    plt.savefig(HERE / "corners.png", dpi=105, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    S = sim()
    good = S["reproj"] < 1.0
    print(f"SIM: {S['tracks']} tracks of {MIN_FRAMES}+ frames; {S['n']} start in frame 0; {good.sum()} reproject within 1 px")
    for sel, name in ((S["edge"], "at depth edges"), (~S["edge"], "on faces")):
        tri, raw = S["d_tri"][sel & good] * 1000, S["d_raw"][sel & good] * 1000
        ro = raw[~np.isnan(raw)]
        print(f"  {name:15s} ({(sel & good).sum():3d}): tracked corners median {np.median(tri):4.1f} mm, 90% {np.percentile(tri, 90):5.1f}, >4 mm {np.mean(tri > 4) * 100:3.0f}%"
              f"   | raw depth median {np.median(ro):4.1f} mm, 90% {np.percentile(ro, 90):5.1f}, >4 mm {np.mean(ro > 4) * 100:3.0f}%, no reading {np.mean(np.isnan(raw)) * 100:3.0f}%")
    R = real()
    z = (R["pts"][:, 2] - SURFACE_Z) * 1000
    ok = R["reproj"] < 2.0
    print(f"REAL: {R['n']} tracks over {R['travel'] * 100:.0f} cm; {ok.sum()} reproject within 2 px; of those, "
          f"{np.mean((z[ok] > -5) & (z[ok] < 40)) * 100:.0f}% land between the table and part height, "
          f"{np.mean(z[ok] < -5) * 100:.0f}% more than 5 mm below the table (median of those {np.median(z[ok][z[ok] < -5]) if (z[ok] < -5).any() else 0:.0f} mm)")
    plot(S, R)
    print("wrote", HERE / "corners.png")
