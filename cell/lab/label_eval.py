"""Which VLM names things best? Re-ask several models with the exact crops and prompts a run sent (runs/<run>/labels/).

    uv run python -m lab.label_eval runs/<run> [--models a,b,c]

Prints, per model: names right, colours right, median seconds, failures; writes labels/eval.json and labels/eval.jpg
(one row per crop, one column per model, green = right, red = wrong).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from lab import vlm
from lab.trial import KIND_WORDS

MODELS = ["meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-90b-vision-instruct", "google/gemma-4-31b-it",
          "google/gemma-3-12b-it", "nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
          "nvidia/cosmos-reason2-8b", "moonshotai/kimi-k2.6", "nvidia/vila"]


def judge(answer: dict | None, truth: str) -> tuple[bool, bool]:
    """(name right, colour right) against a truth like "red cube_10"."""
    if not answer or " " not in truth or truth.endswith("objects"):
        return False, False
    colour, kind = truth.split(" ", 1)
    said = f"{answer.get('color', '')} {answer.get('name', '')}".lower()
    return any(w in said for w in KIND_WORDS.get(kind, [kind])), colour in str(answer.get("color", "")).lower()


def one(model: str, item: dict) -> dict:
    t0 = time.time()
    try:
        r = vlm.ask_image(item["image"], item["prompt"], model=model, max_tokens=4000)
        a, err = r["json"], None if r["json"] else (r["text"][:200] or "empty")
    except Exception as e:
        a, err = None, str(e)[:200]
    name_ok, colour_ok = judge(a, item["truth"])
    return {"model": model, "id": item["id"], "answer": a, "error": err, "seconds": time.time() - t0,
            "right": name_ok, "colour_right": colour_ok}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--models", help="comma-separated model ids (default: every vision model we know of)")
    a = p.parse_args()
    d = Path(a.run) / "labels"
    truth = {r["id"]: r["truth"] for r in json.loads((d / "check.json").read_text())}
    items = []
    for f in sorted(d.glob("[0-9][0-9].json")):
        q = json.loads(f.read_text())
        if q["id"] in truth:
            img = cv2.cvtColor(cv2.imread(str(f.with_suffix(".jpg"))), cv2.COLOR_BGR2RGB)
            items.append({"id": q["id"], "prompt": q["prompt"], "image": img, "truth": truth[q["id"]]})
    models = a.models.split(",") if a.models else MODELS
    with ThreadPoolExecutor(12) as pool:
        res = list(pool.map(lambda mi: one(*mi), [(m, it) for m in models for it in items]))
    print(f"{len(items)} crops from {a.run}\n")
    print(f"{'model':48s} names  colours  median s  failed")
    for m in models:
        rs = [r for r in res if r["model"] == m]
        print(f"{m:48s} {sum(r['right'] for r in rs):2d}/{len(rs):<3d} {sum(r['colour_right'] for r in rs):2d}/{len(rs):<5d} "
              f"{np.median([r['seconds'] for r in rs]):7.1f}   {sum(r['answer'] is None for r in rs)}")
    (d / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    # one row per crop: the close-up, then each model's answer
    W, H = 220, 300
    rows = []
    for it in items:
        cells = [cv2.resize(it["image"][:, :it["image"].shape[0]], (H, H))]
        band = np.zeros((H, W, 3), np.uint8)
        cv2.putText(band, f"#{it['id']} truth:", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(band, it["truth"][:24], (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cells.append(band)
        for m in models:
            r = next(r for r in res if r["model"] == m and r["id"] == it["id"])
            c = np.zeros((H, W, 3), np.uint8)
            c[:] = (30, 70, 30) if r["right"] else (30, 30, 80)
            cv2.putText(c, m.split("/")[1][:26], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            txt = (f"{r['answer'].get('color', '')} {r['answer'].get('name', '')}" if r["answer"] else f"FAILED: {r['error']}")
            for k in range(6):
                cv2.putText(c, txt[k * 24:(k + 1) * 24], (6, 60 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(c, f"{r['seconds']:.0f} s", (6, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cells.append(c)
        rows.append(np.hstack(cells))
    cv2.imwrite(str(d / "eval.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"\nwrote {d / 'eval.jpg'} and eval.json")


if __name__ == "__main__":
    main()
