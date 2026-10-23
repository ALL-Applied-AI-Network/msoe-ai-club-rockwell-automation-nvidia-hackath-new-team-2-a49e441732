"""Run a trial in the simulated cell: build the scene from an order of objects, dump or spread them, let a policy work,
and watch it live.

    uv run python -m lab.trial list                       # the trials in lab/trials/
    uv run python -m lab.trial kinds                      # every object kind you can order
    uv run python -m lab.trial run six_cubes              # run one (opens the viewer)
    uv run python -m lab.trial run junk_drawer --seed 3   # same order, different dump
    uv run python -m lab.trial run six_cubes --no-view
    uv run python -m lab.trial run cube_tile --policy scan   # just look, don't touch

A trial is a small YAML file in lab/trials/:
    instruction: Sort the cubes by color.
    layout: spread            # spread (apart, flat) | dump (poured from a tube into a pile) | tile (flush rows; cols: N)
    seed: 0
    objects:
      - {kind: cube_10, color: red, count: 2}

Every run is recorded to runs/ like a real one (plus truth/), so `python -m lab.replay` shows it again.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from calibration.geometry import load
from lab import viz, vlm
from lab.catalog import COLORS, KINDS
from lab import perception
from lab.policies.explore import ExplorePolicy
from lab.policies.scan import ScanPolicy
from lab.recorder import Recorder
from lab.session import Session
from lab.simcell import SimCell, dump_layout, spread_layout, tile_layout

TRIALS = Path(__file__).resolve().parent / "trials"
POLICIES = {"scan": ScanPolicy, "explore": ExplorePolicy}


def load_trial(name: str) -> dict:
    f = TRIALS / f"{name}.yaml"
    if not f.exists():
        raise SystemExit(f"no trial {name!r}; see: uv run python -m lab.trial list")
    t = yaml.safe_load(f.read_text())
    t["name"] = name
    for o in t["objects"]:
        if o["kind"] not in KINDS:
            raise SystemExit(f"unknown kind {o['kind']!r}; see: uv run python -m lab.trial kinds")
        o.setdefault("color", KINDS[o["kind"]].color)
        if o["color"] not in COLORS:
            raise SystemExit(f"unknown color {o['color']!r}; colors: {', '.join(COLORS)}")
    return t


def build(trial: dict, seed: int) -> SimCell:
    order = [(o["kind"], o["color"]) for o in trial["objects"] for _ in range(o.get("count", 1))]
    rng = np.random.default_rng(seed)
    if trial.get("layout") == "tile":
        cell = SimCell(tile_layout(order, rng, cols=trial.get("cols", len(order))), trial.get("instruction", ""), seed)
        cell.settle(max_s=2.0)
    elif trial.get("layout", "spread") == "dump":
        inst, chute = dump_layout(order, rng)
        cell = SimCell(inst, trial.get("instruction", ""), seed, chute=chute)
        cell.dump()
    else:
        cell = SimCell(spread_layout(order, rng), trial.get("instruction", ""), seed)
        cell.settle(max_s=2.0)
    return cell


def run(name: str, seed: int | None = None, policy: str = "explore", view: bool = True, use_vlm: bool = False,
        max_steps: int = 400, vlm_model: str | None = None) -> Path:
    """Two timelines in the viewer: "step" (one per decision / observation) and "sim_time" (the motion, every 0.1 s)."""
    trial = load_trial(name)
    seed = trial.get("seed", 0) if seed is None else seed
    t0 = time.time()
    cell = build(trial, seed)
    rec = Recorder(f"sim_{name}_s{seed}_{policy}", meta={"source": "sim", "trial": trial, "seed": seed, "policy": policy})
    cell.save_truth(rec.dir / "truth")
    viz.start(f"{name} (seed {seed}, {policy})", save_to=rec.dir / "replay.rrd", spawn=view)
    pol = POLICIES[policy]()
    pol.reset()
    own_mind = hasattr(pol, "world")
    session = Session(load(), live_truth=cell.object_meshes, perceive=not own_mind, show_truth_meshes=False)
    viz.truth_setup(cell)
    anim = viz.Animator(cell)
    viz.set_step(0)
    anim(cell)
    viz.narrate(f"trial {name}, seed {seed}: {len(cell.instances)} objects, layout {trial.get('layout', 'spread')}. "
                f"Instruction: \"{trial.get('instruction', '')}\" (scene built in {time.time() - t0:.1f} s)")
    rec.new_rollout()
    obs, fresh, version, score = cell.observe(), True, -1, {}
    for k in range(max_steps):
        viz.set_step(k)
        viz.set_sim_time(cell.t)
        a0 = time.time()
        chunk = pol.act(obs)
        act_s = time.time() - a0
        note = getattr(pol, "note", "")
        if fresh:
            rec.log(obs, chunk, act_s=act_s, note=note)
            session.step(obs, note=note, actions=chunk, step=k, narrate=not own_mind)
            if k == 0 and use_vlm:
                _gist(obs, trial, vlm_model)
        if own_mind and pol.version != version:
            version = pol.version
            truth = cell.object_meshes()
            score = perception.score(list(pol.world.things.values()), truth)
            mq = _model_quality(pol.world, truth)
            if mq:
                score["models"] = mq
            viz.narrate(f"(truth check: {score['found']}/{score['objects']} objects found, {score['separated']} on their own, "
                        f"{score['merged']} still clumped" + (f"; {mq['n']} models, surface off by {mq['surface_mm']:.1f} mm, "
                        f"volume ×{mq['volume_ratio']:.2f} (median)" if mq else "") + ")", level="DEBUG")
        elif not own_mind:
            score = session.last_score
        if getattr(pol, "done", False):
            break
        cell.play(chunk, on_frame=anim)
        fresh = getattr(pol, "wants_obs", True)
        if fresh:
            obs = cell.observe()
    viz.narrate(f"done: {cell.t:.0f} s of robot time, {time.time() - t0:.0f} s on this computer. Truth check: {score}")
    (rec.dir / "summary.json").write_text(json.dumps({"trial": name, "seed": seed, "policy": policy, "robot_s": round(cell.t),
                                                      "wall_s": round(time.time() - t0), "score": score}))
    print(f"recorded {rec.dir}\nrobot time {cell.t:.0f} s, wall {time.time() - t0:.0f} s\nfinal: {score}")
    return rec.dir


def _model_quality(world, truth) -> dict:
    """Median of how far the robot's models are from the true objects (only models holding exactly one true object)."""
    from lab import objects
    rows = []
    for t in world.things.values():
        if getattr(t, "model", None) is None:
            continue
        inside = [o for o in truth if np.min(np.linalg.norm(t.points[:, :2] - o["mesh"].centroid[:2], axis=1)) < 0.004]
        if len(inside) == 1:
            rows.append(objects.compare(t.model, inside[0]["mesh"]))
    if not rows:
        return {}
    return {"n": len(rows), "surface_mm": float(np.median([r["surface_mm"] for r in rows])),
            "volume_ratio": float(np.median([r["volume_ratio"] for r in rows]))}


def _gist(obs: dict, trial: dict, model: str | None = None) -> None:
    if not vlm.configured(model):
        viz.vlm("(no VLM configured: put VLM_API_KEY and a VLM_<NAME>=<model id> in cell/.env)", level="WARN")
        return
    viz.vlm(f"glancing at the table and side cameras with {vlm.model_id(model)} ...")
    try:
        r = vlm.gist({k: obs["images"][k] for k in ("table", "side", "wrist")}, trial.get("instruction", ""), model)
    except Exception as e:  # a bad key or model shouldn't stop the run
        viz.vlm(f"VLM call failed: {e}", level="ERROR")
        return
    j = r["json"]
    if r.get("thinking"):
        viz.vlm("thinking: " + r["thinking"][:3000], level="DEBUG")
    if j:
        viz.vlm(f"pile: {j.get('pile')}")
        viz.vlm("objects: " + "; ".join(f"{o.get('count')}× {o.get('color', '')} {o.get('what')}" for o in j.get("objects", [])))
        viz.vlm(f"rule: {j.get('rule')} because {j.get('why')}")
        if j.get("shortcut"):
            viz.vlm(f"shortcut: {j.get('shortcut')}")
        if j.get("unsure"):
            viz.vlm(f"unsure about: {j.get('unsure')}")
    else:
        viz.vlm(r["text"])
    viz.vlm(f"({r['seconds']:.1f} s)", level="DEBUG")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("kinds")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--seed", type=int)
    r.add_argument("--policy", default="explore", choices=list(POLICIES))
    r.add_argument("--no-view", action="store_true")
    r.add_argument("--vlm-glance", action="store_true", help="ask the VLM for a first glance (off by default)")
    r.add_argument("--print", action="store_true", help="also print the narration here")
    r.add_argument("--vlm", help="which VLM_<NAME> from cell/.env, e.g. nemotron or cosmos")
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
        run(a.name, a.seed, a.policy, view=not a.no_view, use_vlm=a.vlm_glance, vlm_model=a.vlm)


if __name__ == "__main__":
    main()
