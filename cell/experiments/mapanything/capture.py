"""Frames for the MapAnything test (does many frames from a moving wrist give better depth than one raw depth image?).

    uv run python -m experiments.mapanything.capture sim     # sim sweep: colour + noisy depth + clean depth + poses
    uv run python -m experiments.mapanything.capture real    # real wrist video + poses from the joints (no depth)

Writes experiments/mapanything/<name>.npz with, per frame: img (H,W,3 uint8), K (3x3), pose (4x4 cam->base, OpenCV axes),
and for sim also depth (noisy, m) and truth (clean, m).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
from calibration.kinematics import tool_ik, wpr_of
from lab.sensor import realistic_depth
from lab.synth import Renderer
from lab.trial import build, load_trial
from lab import views

HERE = Path(__file__).parent


def _rgb_at(r: Renderer, cam: dict, T: np.ndarray) -> np.ndarray:
    """A colour image from the depth camera's own viewpoint and lens (so colour and depth line up pixel for pixel)."""
    w, h = cam["size"]
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones((h, w))], -1) @ T[:3, :3].T
    hit = r._cast(T[:3, 3], d)
    gid = hit["geometry_ids"].numpy().astype(np.int64)
    n = hit["primitive_normals"].numpy()
    light = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])
    shade = 0.35 + 0.65 * np.abs(n @ light)
    img = np.full((h, w, 3), (60, 60, 65), float)
    ok = gid < len(r.colors)
    img[ok] = np.array(r.colors)[gid[ok]] * shade[ok, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def sim(n: int = 10, travel: float = 0.10, seed: int = 0) -> None:
    """The wrist slides `travel` m sideways past the objects, looking at them from 0.23 m, tilted 20 deg."""
    cell = build(load_trial("sort_mixed"), seed)
    objs = cell.object_meshes()
    cams = load()
    cam = cams["wrist_depth"]
    r = Renderer(objs, cams)
    target = np.mean([o["mesh"].centroid for o in objs], 0)
    target[2] = SURFACE_Z + 0.01
    T_ct = views._cam_in_tool(cam)
    rng = np.random.default_rng(seed)
    q = np.array(cell.q, float)
    out = {"img": [], "depth": [], "truth": [], "pose": [], "K": []}
    for s in np.linspace(-travel / 2, travel / 2, n):
        aim = target + [0, s, 0]
        e = np.radians(20)
        Tc = views._look_at(aim, np.array([np.sin(e), 0, -np.cos(e)]), 0.23, 0.0)
        Tt = Tc @ np.linalg.inv(T_ct)
        q = tool_ik(Tt[:3, 3], wpr_of(Tt[:3, :3]), q)
        T = flange_pose(q) @ cam["T"]  # the camera where the arm actually puts it
        clean = r.depth("wrist", joints=q)
        out["truth"].append(clean / 1000.0)
        out["depth"].append(realistic_depth(clean, rng) / 1000.0)
        out["img"].append(_rgb_at(r, cam, T))
        out["pose"].append(T)
        out["K"].append(np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]))
    np.savez_compressed(HERE / "sim.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"sim: {n} frames, {travel * 100:.0f} cm sweep, target {np.round(target, 3)}")


def real(episode: int = 0, n: int = 10, min_move_m: float = 0.08) -> None:
    """A stretch of real wrist video where the camera moved at least min_move_m, n frames spread along it."""
    import av
    import pandas as pd
    root = Path.home() / ".cache/huggingface/lerobot/manipbench/r20261024-1f2d05"
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    ep = df[df.episode_index == episode].sort_values("frame_index")
    cams = load()
    cam = cams["wrist"]
    poses = [flange_pose(np.asarray(s[:6], float)) @ cam["T"] for s in ep["observation.state"]]
    xyz = np.array([P[:3, 3] for P in poses])
    # the window with the most camera travel in ~6 s, the arm low enough to see the table well
    best, best_i = 0, 0
    for i in range(0, len(xyz) - 60, 5):
        d = np.linalg.norm(xyz[i + 60] - xyz[i])
        if d > best and xyz[i:i + 60, 2].max() < SURFACE_Z + 0.45:
            best, best_i = d, i
    if best < min_move_m:
        raise SystemExit(f"episode {episode}: no stretch with {min_move_m} m of camera travel (best {best:.3f})")
    pick = np.linspace(best_i, best_i + 60, n).astype(int)
    frames = ep.frame_index.to_numpy()[pick]
    import json
    info = json.loads((root / "meta/info.json").read_text())
    vids = sorted((root / "videos/observation.images.wrist").rglob("*.mp4"))
    vid = vids[episode] if len(vids) > episode else vids[0]
    want, imgs = set(int(f) for f in frames), {}
    with av.open(str(vid)) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k in want:
                imgs[k] = fr.to_ndarray(format="rgb24")
            if len(imgs) == len(want):
                break
    W, H = info["features"]["observation.images.wrist"]["shape"][1], info["features"]["observation.images.wrist"]["shape"][0]
    s = W / cam["size"][0]
    K = np.array([[cam["fx"] * s, 0, W / 2], [0, cam["fx"] * s, H / 2], [0, 0, 1]])
    np.savez_compressed(HERE / f"real{episode}.npz", img=np.array([imgs[int(f)] for f in frames]),
                        pose=np.array([poses[i] for i in pick]), K=np.array([K] * n), frames=frames)
    print(f"real ep {episode}: frames {frames[0]}-{frames[-1]}, camera moved {best * 100:.0f} cm")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "sim"
    sim() if what == "sim" else real(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
