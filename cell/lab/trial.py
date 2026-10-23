"""Run a trial in the simulated cell: build the scene from an order of objects, dump or spread them, let a policy work,
and watch it live.

    uv run python -m lab.trial list                       # the trials in lab/trials/
    uv run python -m lab.trial kinds                      # every object kind you can order
    uv run python -m lab.trial run six_cubes              # run one (opens the viewer)
    uv run python -m lab.trial run junk_drawer --seed 3   # same order, different dump
    uv run python -m lab.trial run six_cubes --no-view --no-vlm

A trial is a small YAML file in lab/trials/:
    instruction: Sort the cubes by color.
    layout: spread            # spread (apart, flat) | dump (poured from a tube into a pile)
    seed: 0
    objects:
      - {kind: cube_10, color: red, count: 2}

Every run is recorded to runs/ like a real one (plus truth/), so `python -m lab.replay` shows it again.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from calibration.geometry import load
from lab import viz, vlm
from lab.catalog import COLORS, KINDS
from lab.policies.scan import ScanPolicy
from lab.recorder import Recorder
from lab.session import Session
from lab.simcell import SimCell, dump_layout, spread_layout

TRIALS = Path(__file__).resolve().parent / "trials"
POLICIES = {"scan": ScanPolicy}


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
    if trial.get("layout", "spread") == "dump":
        inst, chute = dump_layout(order, rng)
        cell = SimCell(inst, trial.get("instruction", ""), seed, chute=chute)
        cell.dump()
    else:
        cell = SimCell(spread_layout(order, rng), trial.get("instruction", ""), seed)
        cell.settle(max_s=2.0)
    return cell


def run(name: str, seed: int | None = None, policy: str = "scan", view: bool = True, use_vlm: bool = True,
        max_steps: int = 40, vlm_model: str | None = None) -> Path:
    trial = load_trial(name)
    seed = trial.get("seed", 0) if seed is None else seed
    t0 = time.time()
    cell = build(trial, seed)
    rec = Recorder(f"sim_{name}_s{seed}", meta={"source": "sim", "trial": trial, "seed": seed, "policy": policy})
    cell.save_truth(rec.dir / "truth")
    viz.start(f"{name} (seed {seed})", save_to=rec.dir / "replay.rrd", spawn=view)
    session = Session(load(), live_truth=cell.object_meshes)
    viz.narrate(f"trial {name}, seed {seed}: {len(cell.instances)} objects, layout {trial.get('layout', 'spread')}. "
                f"Instruction: \"{trial.get('instruction', '')}\" (scene built in {time.time() - t0:.1f} s)")
    pol = POLICIES[policy]()
    pol.reset()
    rec.new_rollout()
    for k in range(max_steps):
        obs = cell.observe()
        a0 = time.time()
        chunk = pol.act(obs)
        act_s = time.time() - a0
        note = getattr(pol, "note", "")
        rec.log(obs, chunk, act_s=act_s, note=note)
        score = session.step(obs, note=note, actions=chunk)
        if k == 0 and use_vlm:
            _gist(obs, trial, vlm_model)
        if getattr(pol, "done", False):
            break
        cell.play(chunk)
    viz.narrate(f"done: {session.k} steps in {time.time() - t0:.0f} s wall time. Final truth check: {score}")
    print(f"recorded {rec.dir}\nfinal: {score}")
    return rec.dir


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
    r.add_argument("--policy", default="scan", choices=list(POLICIES))
    r.add_argument("--no-view", action="store_true")
    r.add_argument("--no-vlm", action="store_true")
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
        run(a.name, a.seed, a.policy, view=not a.no_view, use_vlm=not a.no_vlm, vlm_model=a.vlm)


if __name__ == "__main__":
    main()
