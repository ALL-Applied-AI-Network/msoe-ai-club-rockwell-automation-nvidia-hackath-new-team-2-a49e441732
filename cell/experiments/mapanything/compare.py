"""Score MapAnything against raw depth, and draw both in Rerun.

    uv run python -m experiments.mapanything.compare        # needs sim*.npz / real*.npz from capture.py + infer.py

Sim: every 3D point's distance to the true surfaces (table and objects separately).
Real (no depth, no truth meshes): where the table lands vs its known height, and how flat it is.
Writes compare.rrd (open with `rerun experiments/mapanything/compare.rrd`) and prints a table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr

from calibration.geometry import SURFACE_Z
from lab.synth import table_mesh
from lab.trial import build, load_trial

HERE = Path(__file__).parent
MODES = ["images", "images_poses", "images_poses_depth"]


def _scene(objs):
    s = o3d.t.geometry.RaycastingScene()
    for m in [table_mesh()] + [o["mesh"] for o in objs]:
        s.add_triangles(o3d.core.Tensor(m.vertices, o3d.core.float32), o3d.core.Tensor(m.faces, o3d.core.uint32))
    return s


def _dist(scene, pts):
    """Distance to the nearest true surface, and whether that surface is the table (geometry 0) or an object."""
    q = o3d.core.Tensor(pts.astype(np.float32))
    c = scene.compute_closest_points(q)
    d = np.linalg.norm(c["points"].numpy() - pts, axis=1)
    return d, c["geometry_ids"].numpy() == 0


def _placed(r, in_poses):
    """Each frame's points as seen from its own camera (the model's depth), placed with OUR camera pose for that frame
    (the arm knows where the camera is better than the model does). How it would feed the evidence map."""
    out = []
    for i in range(len(in_poses)):
        P, Q = r["poses"][i], in_poses[i]
        pc = (r["pts"][i][r["mask"][i]] - P[:3, 3]) @ P[:3, :3]
        out.append(pc @ Q[:3, :3].T + Q[:3, 3])
    return np.concatenate(out)


def _travel(r, in_poses):
    """How far the model thinks the camera moved, over how far it really moved."""
    return np.linalg.norm(r["poses"][-1, :3, 3] - r["poses"][0, :3, 3]) / np.linalg.norm(in_poses[-1, :3, 3] - in_poses[0, :3, 3])


def _align(out_poses, in_poses):
    """Map the model's world onto the base frame: rotation + translation from the first camera, scale from how far
    the cameras travelled (for the images-only mode, whose frame and scale are its own guess)."""
    s = np.linalg.norm(in_poses[-1, :3, 3] - in_poses[0, :3, 3]) / max(np.linalg.norm(out_poses[-1, :3, 3] - out_poses[0, :3, 3]), 1e-9)
    A = in_poses[0] @ np.diag([s, s, s, 1.0]) @ np.linalg.inv(out_poses[0])
    S = np.eye(4); S[:3, :3] = in_poses[0][:3, :3] @ np.linalg.inv(out_poses[0][:3, :3]) * s
    S[:3, 3] = in_poses[0][:3, 3] - S[:3, :3] @ out_poses[0][:3, 3]
    return S, s


def _row(name, d, table):
    o = d[~table]
    t = d[table]
    f = lambda x, p: np.percentile(x, p) * 1000 if len(x) else np.nan
    print(f"  {name:22s} objects: median {f(o, 50):5.2f} mm, 90% {f(o, 90):6.2f} mm, >5 mm {np.mean(o > 0.005) * 100:5.1f}%"
          f"   table: median {f(t, 50):5.2f} mm, >5 mm {np.mean(t > 0.005) * 100:5.1f}%   ({len(d)} pts)")


def sim(rec_root="sim"):
    cap = np.load(HERE / "sim.npz")
    cell = build(load_trial("sort_mixed"), 0)
    objs = cell.object_meshes()
    scene = _scene(objs)
    print("SIM: distance from each 3D point to the true surface")
    for o in objs:
        rr.log(f"{rec_root}/truth/{o['id']}", rr.Mesh3D(vertex_positions=o["mesh"].vertices, triangle_indices=o["mesh"].faces,
                                                       vertex_colors=np.tile(o["rgb"], (len(o["mesh"].vertices), 1))), static=True)
    # raw depth: each frame on its own, unprojected with the arm's camera pose
    K = cap["K"][0]
    raw = []
    for i in range(len(cap["img"])):
        z = cap["depth"][i]
        v, u = np.nonzero(z > 0)
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * z[v, u], (v - K[1, 2]) / K[1, 1] * z[v, u], z[v, u]], -1)
        P = cap["pose"][i]
        raw.append(pc @ P[:3, :3].T + P[:3, 3])
    raw = np.concatenate(raw)
    raw = raw[(raw[:, 2] > SURFACE_Z - 0.05)]
    d, tab = _dist(scene, raw)
    _row("raw depth (1 frame each)", d, tab)
    _log(f"{rec_root}/raw_depth", raw, d)
    for m in MODES:
        f = HERE / f"sim_{m}.npz"
        if not f.exists():
            continue
        r = np.load(f)
        pts = _placed(r, cap["pose"])
        pts = pts[pts[:, 2] > SURFACE_Z - 0.05]
        d, tab = _dist(scene, pts)
        _row(f"MapAnything {m.replace('_', '+')}", d, tab)
        _log(f"{rec_root}/mapanything_{m}", pts, d)
        print(f"  {'':22s} ({float(r['secs']):.1f} s for {len(r['pts'])} frames; it thought the camera moved"
              f" {_travel(r, cap['pose']) * 100:.0f}% of the real distance)")


def _log(path, pts, d):
    """Points coloured by error: green under 1 mm, yellow ~3 mm, red 5 mm and worse."""
    e = np.clip(d / 0.005, 0, 1)
    col = np.stack([255 * np.clip(2 * e, 0, 1), 255 * np.clip(2 - 2 * e, 0, 1), np.full_like(e, 40)], -1).astype(np.uint8)
    keep = np.random.default_rng(0).choice(len(pts), min(len(pts), 300_000), replace=False)
    rr.log(path, rr.Points3D(pts[keep], colors=col[keep], radii=0.0006), static=True)


def real():
    print("\nREAL (wrist video + arm poses, no depth): the table should come out flat at its known height")
    for f in sorted(HERE.glob("real[0-9].npz")):
        cap = np.load(f)
        for m in ("images", "images_poses"):
            g = HERE / f"{f.stem}_{m}.npz"
            if not g.exists():
                continue
            r = np.load(g)
            pts = _placed(r, cap["pose"])
            col = (r["img"][r["mask"]] * 255).clip(0, 255).astype(np.uint8)
            z = pts[:, 2] - SURFACE_Z
            hist, edges = np.histogram(z, bins=np.arange(-0.1, 0.1, 0.002))
            mode = edges[np.argmax(hist)] + 0.001
            near = np.abs(z - mode) < 0.015
            spread = np.median(np.abs(z[near] - np.median(z[near]))) * 1.4826 * 1000
            cam_err = _travel(r, cap["pose"]) * 100
            print(f"  {f.stem} {m.replace('_', '+'):14s} table at {mode * 1000:+6.1f} mm from where it is,"
                  f" spread {spread:4.1f} mm, {near.mean() * 100:4.0f}% of points on it"
                  + f", camera travel {cam_err:.0f}% of real")
            keep = np.random.default_rng(0).choice(len(pts), min(len(pts), 200_000), replace=False)
            rr.log(f"real/{f.stem}/{m}", rr.Points3D(pts[keep], colors=col[keep], radii=0.0008), static=True)
    t = table_mesh()
    rr.log("real/table_truth", rr.Mesh3D(vertex_positions=t.vertices, triangle_indices=t.faces,
                                        vertex_colors=np.tile([90, 110, 130], (len(t.vertices), 1))), static=True)


if __name__ == "__main__":
    rr.init("mapanything_test")
    rr.save(str(HERE / "compare.rrd"))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    sim()
    real()
