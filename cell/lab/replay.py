"""Replay a recorded run (real cell, mock, or synthetic) into the viewer.

    uv run python -m lab.replay                 # the newest run in runs/
    uv run python -m lab.replay runs/<dir>      # a specific one
    uv run python -m lab.replay --no-spawn ...  # only write <dir>/replay.rrd

Per step: the camera images, the depth images and the 3D points they give, the arm, the actions it was sent, and a
narration line. If the run has truth/ (synthetic), the true objects are drawn too, with their ids on every image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh

from calibration.geometry import depth_to_base, flange_pose, load
from lab import viz
from lab.fusion import height_map
from lab.recorder import RUNS, load_step


def _truth(run: Path) -> list[dict]:
    f = run / "truth" / "objects.json"
    if not f.exists():
        return []
    objs = json.loads(f.read_text())
    for o in objs:
        o["mesh"] = trimesh.load(run / "truth" / o["mesh"])
        o["label"] = o["name"]
        o["alpha"] = 0.35
    return objs


def replay(run: Path, spawn: bool = True) -> Path:
    run = Path(run)
    out = run / "replay.rrd"
    viz.start(f"replay {run.name}", save_to=out, spawn=spawn)
    cams = load()
    truth = _truth(run)
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    step = 0
    for rdir in sorted(run.glob("rollout_*")):
        fused: list[np.ndarray] = []
        steps = sorted(rdir.glob("step_*.json"))
        for k, f in enumerate(steps):
            obs = load_step(f)
            viz.set_step(step)
            if k == 0:
                viz.narrate(f"── {rdir.name} ({meta.get('source', meta.get('tag', 'run'))}, commit {meta.get('commit', '?')}) ──")
            viz.images(obs)
            if obs["joints"] is not None:
                viz.arm(obs["joints"])
            for cam, d in obs["depth"].items():
                c = cams[f"{cam}_depth"]
                if c["frame"] == "flange" and obs["joints"] is None:
                    continue
                T = flange_pose(obs["joints"]) @ c["T"] if c["frame"] == "flange" else c["T"]
                viz.camera_frustum(f"world/cameras/{cam}", c, T)
                pts = depth_to_base(c, d, stride=2, joints=obs["joints"] if c["frame"] == "flange" else None)
                rr.log(f"world/points/{cam}", rr.Points3D(pts, colors=viz.height_colors(pts), radii=0.0007))
                fused.append(pts)
            if fused:
                viz.heightmap(height_map(np.concatenate(fused)))
            if obs.get("actions"):
                viz.path(obs["actions"], name="sent_actions")
            if truth:
                viz.objects(truth, cams, obs, kind="truth")
            n = len(obs["actions"]) if obs.get("actions") else 0
            act = f", act() took {obs['act_s'] * 1000:.0f} ms" if obs.get("act_s") is not None else ""
            viz.narrate(f"step {k}: {obs.get('note') or 'observation'}"
                        + (f"; sent {n} targets" if n else "") + act)
            step += 1
    rr.disconnect()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="a run directory (default: the newest in runs/)")
    p.add_argument("--no-spawn", action="store_true", help="only write <run>/replay.rrd")
    a = p.parse_args()
    run = Path(a.run) if a.run else max((d for d in RUNS.iterdir() if d.is_dir()), key=lambda d: d.name)
    print(f"replaying {run}")
    print(f"wrote {replay(run, spawn=not a.no_spawn)}")


if __name__ == "__main__":
    main()
