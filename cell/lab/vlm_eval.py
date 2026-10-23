"""Which VLM sees best? Show each model the same simulated piles and compare what it says with the truth.

    uv run python -m lab.vlm_eval                          # every VLM_<NAME> in cell/.env, on a few trials
    uv run python -m lab.vlm_eval --models cosmos --trials dice --seeds 0 1 2

Prints a table and saves every answer (and the image it was shown) to runs/vlm_eval_<time>/.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import cv2
import numpy as np

from lab import vlm
from lab.recorder import RUNS
from lab.synth import Renderer
from lab.trial import build, load_trial

SIZE = (1280, 720)


def overview(cell) -> dict[str, np.ndarray]:
    """Bigger overview images than the policy's default: table and side cameras, and the wrist camera from home."""
    cell.observe(rng_noise=False)  # updates the arm pose used by the wrist camera
    r = Renderer([*cell.object_meshes(), *cell.finger_meshes()], cell.cams)
    return {c: r.rgb(c, joints=cell.q, size=SIZE) for c in ("table", "side", "wrist")}


def truth_counts(cell) -> Counter:
    return Counter(i.name for i in cell.instances)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", help="names from cell/.env (default: all)")
    p.add_argument("--trials", nargs="*", default=["six_cubes", "dice", "junk_drawer"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    a = p.parse_args()
    names = a.models or [k for k in vlm.models() if k != "default"]
    out = RUNS / f"vlm_eval_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    rows = []
    for t in a.trials:
        trial = load_trial(t)
        for seed in a.seeds:
            cell = build(trial, seed)
            ims = overview(cell)
            cv2.imwrite(str(out / f"{t}_s{seed}.jpg"), cv2.cvtColor(vlm.tile(ims, width=2048), cv2.COLOR_RGB2BGR))
            truth = truth_counts(cell)
            for m in names:
                try:
                    r = vlm.gist(ims, trial.get("instruction", ""), model=m)
                except Exception as e:
                    r = {"text": f"ERROR: {e}", "json": None, "seconds": 0.0, "model": vlm.model_id(m)}
                j = r["json"] or {}
                said = j.get("total_count")
                rows.append({"trial": t, "seed": seed, "model": m, "truth_total": sum(truth.values()), "said_total": said,
                             "rule": j.get("rule"), "pile": j.get("pile"), "seconds": round(r["seconds"], 1),
                             "parsed": r["json"] is not None, "objects": j.get("objects"), "truth": dict(truth),
                             "raw": r["text"][:4000], "thinking": r.get("thinking", ""), "finish": r.get("finish")})
                print(f"{t:12s} s{seed} {m:9s} truth {sum(truth.values()):2d}  said {said!s:>4}  {r['seconds']:5.1f}s  "
                      f"{'json' if r['json'] else 'NO JSON (' + str(r.get('finish')) + ')'}  rule: {str(j.get('rule'))[:70]}")
    (out / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
