"""Run a trial in a simulated cell: build the scene from an order of objects, dump or spread them, let a mind work,
and watch it live.

    uv run python -m lab.trial list                       # the trials in lab/trials/
    uv run python -m lab.trial kinds                      # every object kind you can order
    uv run python -m lab.trial run sort_cubes             # run one with the mapper (opens the viewer)
    uv run python -m lab.trial run sort_mixed --cell isaac --no-view
    uv run python -m lab.trial run sort_cubes --clean     # perfect depth (default: realistic sensor noise)

The minds are in lab/minds/ (v1's sort policy is archived: git checkout v1-sort).

A trial is a small YAML file in lab/trials/:
    instruction: Sort the cubes by color.
    layout: spread            # spread (apart, flat) | dump (poured from a tube into a pile) | tile (flush rows; cols: N)
    gap: 0.012                # spread only: least gap between things (m); half: [x, y] half-size of the area (m)
    seed: 0
    objects:
      - {kind: cube_10, color: red, count: 2}
    fixtures:                 # optional: things set down at fixed spots before the objects arrive (boxes to sort into)
      - {kind: box_60, color: red, at: [0.30, -0.24], yaw: 0}
    expect: color             # optional: what a correct sort groups by (color | kind | name), for the truth check

Every run is recorded to runs/ (replay.rrd for Rerun, narration.txt, summary.json, truth/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from calibration.geometry import SURFACE_Z
from lab import viz
from lab.catalog import COLORS, KINDS, kind_mesh
from lab import primitives as P
from lab.minds import MINDS
from lab.recorder import Recorder
from lab.simcell import Instance, SimCell, dump_layout, spread_layout, tile_layout

TRIALS = Path(__file__).resolve().parent / "trials"


def load_trial(name: str) -> dict:
    f = TRIALS / f"{name}.yaml"
    if not f.exists():
        raise SystemExit(f"no trial {name!r}; see: uv run python -m lab.trial list")
    t = yaml.safe_load(f.read_text())
    t["name"] = name
    for o in t["objects"] + t.get("fixtures", []):
        if o["kind"] not in KINDS:
            raise SystemExit(f"unknown kind {o['kind']!r}; see: uv run python -m lab.trial kinds")
        o.setdefault("color", KINDS[o["kind"]].color)
        if o["color"] not in COLORS:
            raise SystemExit(f"unknown color {o['color']!r}; colors: {', '.join(COLORS)}")
    return t


def build(trial: dict, seed: int) -> SimCell:
    order = [(o["kind"], o["color"]) for o in trial["objects"] for _ in range(o.get("count", 1))]
    rng = np.random.default_rng(seed)
    fixed = lambda inst: inst + fixtures(trial, start=len(inst) + 1)
    if trial.get("layout") == "tile":
        cell = SimCell(fixed(tile_layout(order, rng, cols=trial.get("cols", len(order)))), trial.get("instruction", ""), seed)
        cell.settle(max_s=2.0)
    elif trial.get("layout", "spread") == "dump":
        inst, chute = dump_layout(order, rng)
        cell = SimCell(fixed(inst), trial.get("instruction", ""), seed, chute=chute)
        cell.dump()
    else:
        kw = {k: (tuple(trial[k]) if isinstance(trial[k], list) else trial[k]) for k in ("gap", "half") if k in trial}
        cell = SimCell(fixed(spread_layout(order, rng, **kw)), trial.get("instruction", ""), seed)
        cell.settle(max_s=2.0)
    return cell


def fixtures(trial: dict, start: int) -> list[Instance]:
    """Boxes and the like, set down where the trial says."""
    out = []
    for k, f in enumerate(trial.get("fixtures", [])):
        m = kind_mesh(KINDS[f["kind"]])
        yaw = np.radians(f.get("yaw", 0))
        out.append(Instance(start + k, f["kind"], f["color"], np.array([*f["at"], SURFACE_Z - m.bounds[0][2] + 0.0005]),
                            np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
    return out


def objects_only(truth: list[dict]) -> list[dict]:
    """The things to sort, without the boxes."""
    return [o for o in truth if "container" not in KINDS[o["kind"]].tags]


def sort_check(truth: list[dict], expect: str | None) -> dict:
    """Where every object ended up: in which box (by box colour), or still on the table; and whether each box holds
    one kind of thing (by the trial's `expect`: color, kind or name)."""
    boxes = [o for o in truth if "container" in KINDS[o["kind"]].tags]
    where, contents = {}, {b["color"] + " box": [] for b in boxes}  # where: by object index (names repeat)
    objs = objects_only(truth)
    for k, o in enumerate(objs):
        c = o["mesh"].centroid
        inside = [b for b in boxes if np.all(np.abs(c[:2] - b["mesh"].centroid[:2]) < b["mesh"].extents[:2] / 2 - 0.004)]
        dest = inside[0]["color"] + " box" if inside else "table"
        where[k] = dest
        if inside:
            contents[dest].append(o)
    key = {"color": lambda o: o["color"], "kind": lambda o: o["kind"], "name": lambda o: o["name"]}.get(expect or "")
    mixed = [b for b, os_ in contents.items() if key and len({key(o) for o in os_}) > 1]
    split = []
    if key:
        for v in {key(o) for o in objs}:
            dests = {where[k] for k, o in enumerate(objs) if key(o) == v} - {"table"}
            if len(dests) > 1:
                split.append(v)
    n_in = sum(len(v) for v in contents.values())
    return {"in_boxes": n_in, "objects": len(objects_only(truth)), "mixed_boxes": mixed, "split_groups": split,
            "contents": {b: [o["name"] for o in os_] for b, os_ in contents.items()},
            "boxes": {b["color"] + " box": [round(float(v) * 1000) for v in b["mesh"].centroid] for b in boxes},
            "on_table": [f'{o["name"]} at ({o["mesh"].centroid[0] * 1000:.0f}, {o["mesh"].centroid[1] * 1000:.0f}) mm'
                         for k, o in enumerate(objs) if where[k] == "table"]}


def pile_check(truth: list[dict], expect: str | None, touch_mm: float = 30.0) -> dict:
    """No boxes: the piles on the table (objects within touch_mm of each other, chained), and whether each holds one
    kind (by `expect`) and each kind is in one pile."""
    objs = objects_only(truth)
    xy = np.array([o["mesh"].centroid[:2] for o in objs])
    pile = list(range(len(objs)))
    def root(i):
        while pile[i] != i:
            i = pile[i]
        return i
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if np.linalg.norm(xy[i] - xy[j]) * 1000 < touch_mm:
                pile[root(i)] = root(j)
    key = {"color": lambda o: o["color"], "kind": lambda o: o["kind"], "name": lambda o: o["name"]}.get(expect or "color")
    groups: dict = {}
    for i, o in enumerate(objs):
        groups.setdefault(root(i), []).append(o)
    piles = [{"at": [round(float(v) * 1000) for v in np.mean([o["mesh"].centroid[:2] for o in os_], 0)],
              "holds": sorted(key(o) for o in os_)} for os_ in groups.values()]
    mixed = [p_ for p_ in piles if len(set(p_["holds"])) > 1]
    split = sorted(v for v in {key(o) for o in objs} if sum(v in p_["holds"] for p_ in piles) > 1)
    return {"objects": len(objs), "piles": len(piles), "mixed_piles": mixed, "split": split,
            "clean": not mixed and not split}


def run(name: str, seed: int | None = None, mind: str = "map", view: bool = True, where: str = "mujoco",
        max_steps: int = 400, noise: bool = True) -> Path:
    """Two timelines in the viewer: "step" (one per decision / observation) and "sim_time" (the motion, every 0.1 s)."""
    trial = load_trial(name)
    seed = trial.get("seed", 0) if seed is None else seed
    t0 = time.time()
    cell = build(trial, seed)
    if where == "isaac":  # the same layout, handed to Isaac Sim on the 4090 box: it does the physics and cameras
        from lab.isaaccell import IsaacCell
        cell = IsaacCell(cell)
    cell.noise = noise
    rec = Recorder(f"{where}_{name}_s{seed}_{mind}", meta={"source": where, "trial": trial, "seed": seed, "mind": mind, "noise": noise})
    cell.save_truth(rec.dir / "truth")
    viz.start(f"{name} (seed {seed}, {mind})", save_to=rec.dir / "replay.rrd", spawn=view)
    m = MINDS[mind](instruction=trial.get("instruction", ""))
    m.reset()
    max_steps = max(max_steps, getattr(m, "MAX_STEPS", 0))  # a mind that waits on another (the ventral stream) idles in steps
    cell.image_size = tuple(getattr(m, "IMAGE_SIZE", cell.image_size))
    viz.truth_setup(cell)
    anim = viz.Animator(cell)
    viz.set_step(0)
    anim(cell)
    viz.narrate(f"trial {name}, seed {seed}: {len(cell.instances)} objects, layout {trial.get('layout', 'spread')}, "
                f"depth {'with realistic noise' if noise else 'perfect'}, mind: {mind}")
    obs = cell.observe()
    for k in range(max_steps):
        viz.set_step(k)
        viz.set_sim_time(cell.t)
        viz.images(obs)
        chunk = m.act(obs)
        if getattr(m, "done", False):
            break
        P.send(cell, chunk, on_frame=anim)
        obs = cell.observe()
    if hasattr(m, "stop"):
        m.stop()
    score = {"robot_s": round(cell.t), "wall_s": round(time.time() - t0)}
    if hasattr(m, "score"):
        score |= m.score(cell.object_meshes())
    if trial.get("fixtures"):
        score["sort"] = sort_check(cell.object_meshes(), trial.get("expect"))
    elif trial.get("expect"):
        score["piles"] = pile_check(cell.object_meshes(), trial.get("expect"))
    viz.narrate(f"done: {cell.t:.0f} s of robot time, {time.time() - t0:.0f} s on this computer. {score}")
    (rec.dir / "summary.json").write_text(json.dumps({"trial": name, "seed": seed, "mind": mind, "score": score}))
    print(f"recorded {rec.dir}\n{score}")
    return rec.dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("kinds")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--seed", type=int)
    r.add_argument("--mind", default="map", choices=list(MINDS))
    r.add_argument("--no-view", action="store_true")
    r.add_argument("--print", action="store_true", help="also print the narration here")
    r.add_argument("--clean", action="store_true", help="perfect depth (default: realistic sensor noise)")
    r.add_argument("--cell", default="mujoco", choices=["mujoco", "isaac"],
                   help="isaac: physics and cameras in Isaac Sim on the 4090 box (isaac/server.py must be running there)")
    a = p.parse_args()
    if a.cmd == "list":
        for f in sorted(TRIALS.glob("*.yaml")):
            t = yaml.safe_load(f.read_text())
            n = sum(o.get("count", 1) for o in t["objects"])
            print(f"{f.stem:16s} {n:3d} objects  {t.get('layout', 'spread'):7s}  \"{t.get('instruction', '')}\"")
    elif a.cmd == "kinds":
        for k in KINDS.values():
            print(f"{k.name:18s} default {k.color:7s} {k.notes}")
        print(f"\ncolors: {', '.join(COLORS)}")
    else:
        viz.PRINT = a.print
        run(a.name, a.seed, a.mind, view=not a.no_view, where=a.cell, noise=not a.clean)


if __name__ == "__main__":
    main()
