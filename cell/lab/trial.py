"""Run a trial in the simulated cell: build the scene from an order of objects, dump or spread them, let a policy work,
and watch it live.

    uv run python -m lab.trial list                       # the trials in lab/trials/
    uv run python -m lab.trial kinds                      # every object kind you can order
    uv run python -m lab.trial run six_cubes              # run one (opens the viewer)
    uv run python -m lab.trial run junk_drawer --seed 3   # same order, different dump
    uv run python -m lab.trial run six_cubes --no-view
    uv run python -m lab.trial run cube_tile --policy scan   # just look, don't touch
    uv run python -m lab.trial run mixed_pile --label   # also name things (VLM: lab/labels.py FALLBACKS, or --vlm <model>)

A trial is a small YAML file in lab/trials/:
    instruction: Sort the cubes by color.
    layout: spread            # spread (apart, flat) | dump (poured from a tube into a pile) | tile (flush rows; cols: N)
    seed: 0
    objects:
      - {kind: cube_10, color: red, count: 2}
    fixtures:                 # optional: things set down at fixed spots before the objects arrive (boxes to sort into)
      - {kind: box_60, color: red, at: [0.30, -0.24], yaw: 0}
    expect: color             # optional: what a correct sort groups by (color | kind | name), for the truth check

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
from calibration.geometry import SURFACE_Z
from lab.catalog import COLORS, KINDS, kind_mesh
from lab import perception
from lab.policies.explore import ExplorePolicy
from lab.policies.scan import ScanPolicy
from lab.policies.sort import SortPolicy
from lab.recorder import Recorder
from lab.session import Session
from lab.simcell import Instance, SimCell, dump_layout, spread_layout, tile_layout

TRIALS = Path(__file__).resolve().parent / "trials"
POLICIES = {"scan": ScanPolicy, "explore": ExplorePolicy, "sort": SortPolicy}


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
        cell = SimCell(fixed(spread_layout(order, rng)), trial.get("instruction", ""), seed)
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


def run(name: str, seed: int | None = None, policy: str = "explore", view: bool = True, use_vlm: bool = False,
        max_steps: int = 400, vlm_model: str | None = None, name_things: bool = False, where: str = "mujoco") -> Path:
    """Two timelines in the viewer: "step" (one per decision / observation) and "sim_time" (the motion, every 0.1 s)."""
    trial = load_trial(name)
    seed = trial.get("seed", 0) if seed is None else seed
    t0 = time.time()
    cell = build(trial, seed)
    if where == "isaac":  # the same layout, handed to Isaac Sim on the 4090 box: it does the physics and cameras
        from lab.isaaccell import IsaacCell
        cell = IsaacCell(cell)
    rec = Recorder(f"{'isaac' if where == 'isaac' else 'sim'}_{name}_s{seed}_{policy}",
                   meta={"source": where, "trial": trial, "seed": seed, "policy": policy})
    cell.save_truth(rec.dir / "truth")
    viz.start(f"{name} (seed {seed}, {policy})", save_to=rec.dir / "replay.rrd", spawn=view)
    if policy == "explore":
        pol = ExplorePolicy(label_model=vlm_model, name_things=name_things, instruction=trial.get("instruction", ""))
    elif policy == "sort":
        pol = SortPolicy(label_model=vlm_model, instruction=trial.get("instruction", ""))
    else:
        pol = POLICIES[policy]()
    pol.reset()
    cell.image_size = tuple(getattr(pol, "IMAGE_SIZE", cell.image_size))  # the cell serves the size the policy asks for
    if getattr(pol, "labeler", None) is not None:
        pol.labeler.save_dir = rec.dir / "labels"
        if not pol.labeler.on:
            viz.vlm("(naming is on but no VLM is configured: put VLM_API_KEY and a model in cell/.env)", level="WARN")
    own_mind = hasattr(pol, "world")
    session = Session(load(), live_truth=cell.object_meshes, perceive=not own_mind, show_truth_meshes=False)
    viz.truth_setup(cell)
    anim = viz.Animator(cell)
    viz.set_step(0)
    anim(cell)
    viz.narrate(f"trial {name}, seed {seed}: {len(cell.instances)} objects, layout {trial.get('layout', 'spread')}. "
                f"Instruction: \"{trial.get('instruction', '')}\" (scene built in {time.time() - t0:.1f} s)")
    rec.new_rollout()
    last, n_flung = {}, [0]

    def flung(c, note):
        """Sim sanity: the gripper never moves faster than 150 mm/s (15 mm per frame); a part moving 40 mm in one
        frame was thrown by the physics (parts squeezed into each other), which a real table wouldn't do."""
        for i, b in c.body_positions().items():
            if i in last and np.linalg.norm(b[:2] - last[i][:2]) > 0.04:
                n_flung[0] += 1
                o = next(x for x in c.instances if x.id == i)
                viz.narrate(f"(sim check: the {o.color} {o.kind} was flung at {np.linalg.norm(b[:2] - last[i][:2]) * 10:.1f} m/s "
                            f"during \"{note}\": the physics glitching, not something a real table would do)", level="WARN")
            last[i] = b
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
            truth = objects_only(cell.object_meshes())
            score = perception.score([t for t in pol.world.things.values() if t.status != "bin"], truth)
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
        cell.play(chunk, on_frame=lambda c: (anim(c), flung(c, note)))
        fresh = getattr(pol, "wants_obs", True)
        if fresh:
            obs = cell.observe()
    if trial.get("fixtures"):
        sc = score["sort"] = sort_check(cell.object_meshes(), trial.get("expect"))
        viz.narrate(f"(truth check of the sort: {sc['in_boxes']}/{sc['objects']} in boxes; "
                    + "; ".join(f"{b}: {', '.join(c) or 'empty'}" for b, c in sc["contents"].items())
                    + (f"; mixed: {', '.join(sc['mixed_boxes'])}" if sc["mixed_boxes"] else "")
                    + (f"; split across boxes: {', '.join(sc['split_groups'])}" if sc["split_groups"] else "") + ")")
    if getattr(pol, "labeler", None) is not None:
        score["labels"] = _label_check(pol, cell.object_meshes(), rec.dir / "labels")
        lc = score["labels"]
        viz.narrate(f"(truth check of the names: {lc['right']}/{lc['asked']} right, {lc['colour_right']}/{lc['asked']} colours right; "
                    f"contact sheet in labels/sheet.jpg)", level="DEBUG")
    viz.narrate(f"done: {cell.t:.0f} s of robot time, {time.time() - t0:.0f} s on this computer. Truth check: {score}")
    score["sim_flung"] = n_flung[0]
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


# words that count as naming each kind right (sim parts are plain primitives: a d6 has no pips, so "cube" is fair)
KIND_WORDS = {"cube_10": ["cube", "block"], "cube_16": ["cube", "block"], "screw_m6x30": ["screw", "bolt"],
              "jack": ["jack"], "ring": ["ring", "torus", "washer"], "d4": ["d4", "die", "dice", "pyramid", "tetra"],
              "d6": ["die", "dice", "cube", "d6"], "d8": ["d8", "die", "dice", "octahedr"],
              "d20": ["d20", "die", "dice", "icosahedr"], "lego_1x2": ["lego", "brick"], "lego_2x4": ["lego", "brick"],
              "soldier": ["soldier", "figure", "figurine"], "dowel": ["dowel", "rod", "peg", "stick", "cylinder", "pin"],
              "marker": ["marker", "pen"], "tongue_depressor": ["depressor", "stick", "popsicle"], "cup": ["cup"],
              "bin": ["bin", "box"], "box_60": ["box", "bin", "container", "tray"]}


def _label_check(pol, truth: list[dict], out: Path) -> dict:
    """Each named thing next to what it really is (when it holds exactly one true object), plus a contact sheet."""
    import cv2
    rows, tiles = [], []
    for t in sorted(pol.world.things.values(), key=lambda t: t.id):
        lab = getattr(t, "label", None)
        if lab is None:
            continue
        inside = [o for o in truth if np.min(np.linalg.norm(t.points[:, :2] - o["mesh"].centroid[:2], axis=1)) < 0.004]
        real = f"{inside[0]['color']} {inside[0]['kind']}" if len(inside) == 1 else f"{len(inside)} objects"
        said = f"{lab.get('color', '')} {lab.get('name', '')}".lower()
        right = len(inside) == 1 and any(w in said for w in KIND_WORDS.get(inside[0]["kind"], [inside[0]["kind"]]))
        colour_right = len(inside) == 1 and inside[0]["color"] in str(lab.get("color", "")).lower()
        rows.append({"id": t.id, "said": said.strip(), "truth": real, "right": bool(right), "colour_right": bool(colour_right),
                     "confidence": lab.get("confidence"), "group": getattr(t, "group", None)})
        f = out / f"{t.id:02d}.jpg"
        if f.exists():
            im = cv2.imread(str(f))
            im = cv2.resize(im[:, :im.shape[0]], (300, 300))  # the close-up half
            band = np.zeros((70, 300, 3), np.uint8)
            cv2.putText(band, f"#{t.id} said: {said.strip()[:30]}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(band, f"truth: {real}", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 220, 90) if right else (80, 80, 240), 1)
            tiles.append(np.vstack([im, band]))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / "sheet.jpg"), sheet)
        rr_img = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
        viz.rr.log("vlm_crop", viz.rr.Image(rr_img).compress(jpeg_quality=85))
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(rows, indent=1, default=float))
    if pol.sort_plan:
        (out / "sort_plan.json").write_text(json.dumps(pol.sort_plan, indent=1))
    return {"asked": len(rows), "right": sum(r["right"] for r in rows), "colour_right": sum(r["colour_right"] for r in rows),
            "groups": [g.get("name") for g in (pol.sort_plan or {}).get("groups", [])]}


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
    r.add_argument("--vlm", help="which VLM_<NAME> from cell/.env (e.g. nemotron), or a full model id")
    r.add_argument("--label", action="store_true", help="name each confirmed thing with the VLM, then plan the sort")
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
        run(a.name, a.seed, a.policy, view=not a.no_view, use_vlm=a.vlm_glance, vlm_model=a.vlm, name_things=a.label,
            where=a.cell)


if __name__ == "__main__":
    main()
