"""Replay a recorded run (real cell, mock, or sim) into the viewer, step by step, through the same Session as live runs.

    uv run python -m lab.replay                 # the newest run in runs/
    uv run python -m lab.replay runs/<dir>      # a specific one
    uv run python -m lab.replay --no-spawn ...  # only write <dir>/replay.rrd

Runs with truth/ (sim) show the true objects (as they were at the start) in the World view; real runs leave it empty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rerun as rr
import trimesh

from calibration.geometry import load
from lab import viz
from lab.recorder import RUNS, load_step
from lab.session import Session


def _truth(run: Path) -> list[dict]:
    f = run / "truth" / "objects.json"
    if not f.exists():
        return []
    objs = json.loads(f.read_text())
    for o in objs:
        o["mesh"] = trimesh.load(run / "truth" / o["mesh"])
        o.setdefault("rgb", (200, 200, 200))
    return objs


def replay(run: Path, spawn: bool = True) -> Path:
    run = Path(run)
    out = run / "replay.rrd"
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    viz.start(f"replay {run.name}", save_to=out, spawn=spawn)
    truth = _truth(run)
    for rdir in sorted(run.glob("rollout_*")):
        s = Session(load(), truth=truth or None)
        viz.set_step(0)
        viz.narrate(f"{rdir.name} of {run.name} (source {meta.get('source', meta.get('tag', '?'))}, commit {meta.get('commit', '?')})")
        for f in sorted(rdir.glob("step_*.json")):
            obs = load_step(f)
            s.step(obs, note=obs.get("note", ""), actions=obs.get("actions"))
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
