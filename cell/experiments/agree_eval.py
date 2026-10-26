"""Do the two streams agree? Map a MuJoCo scene (depth -> segments), ask SAM 3.1 about the table camera's picture,
compare (lab.agree), and save a picture of it: masks filled, depth segments outlined in the colour of their case.

    uv run python -m experiments.agree_eval [trial] [seed] [prompt ...]
    e.g. uv run python -m experiments.agree_eval sort_touching 0 cube box
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from lab import agree, primitives as P, sam
from lab.minds import MINDS
from lab.things import Things
from lab.trial import build, load_trial

OUT = Path(__file__).resolve().parent




def picture(img, ag, things, cam) -> np.ndarray:
    out = img.copy()
    for m in ag.masks:
        col = np.array(agree.CASE_COLOUR.get(m["case"], (90, 140, 255)) if m["case"] in ("no_segment", "merged") else (90, 140, 255))
        out[m["mask"]] = (0.5 * out[m["mask"]] + 0.5 * col).astype(np.uint8)
    by = {t.id: t for t in things}
    for r in ag.rows:
        s = agree.silhouette(by[r.seg], cam, ag.image_wh).astype(np.uint8)
        cs, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, agree.CASE_COLOUR[r.case], 1)
        x, y = cs[0][:, 0, :].min(0)
        cv2.putText(out, f"#{r.seg}", (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, agree.CASE_COLOUR[r.case], 1)
    return out


def run(trial: str, seed: int, prompts: list[str]):
    cell = build(load_trial(trial), seed)
    cell.noise = True
    cell.image_size = (1920, 1080)
    m = MINDS["map"]()
    m.reset()
    obs = cell.observe()
    img = obs["images"]["table"]
    for _ in range(40):
        ch = m.act(obs)
        if m.done:
            break
        P.send(cell, ch)
        obs = cell.observe()
    th = Things()
    th.update(m.map, img, m.cams)
    things = th.known + th.maybe
    cam = m.cams["table"]
    wh = (img.shape[1], img.shape[0])
    crop, scale = agree.focus(things, cam, wh)
    masks = [d for p in prompts for d in sam.segment(img, p, crop=crop, scale=scale)]
    ag = agree.compare(things, masks, cam, wh)
    print(f"{trial} s{seed}: {len(cell.instances)} objects really there; depth: {len(th.known)} things + {len(th.maybe)} maybe; "
          f"SAM ({', '.join(prompts)}): {len(ag.masks)} masks; cases {ag.case_counts()}")
    print(ag.text({t.id: t for t in things}))
    pic = picture(img, ag, things, cam)
    x0, y0, x1, y1 = crop
    path = OUT / f"agree_{trial}_s{seed}.png"
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(pic[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    print("picture:", path)
    return ag


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "sort_touching", int(a[1]) if len(a) > 1 else 0, a[2:] or ["cube", "box"])
