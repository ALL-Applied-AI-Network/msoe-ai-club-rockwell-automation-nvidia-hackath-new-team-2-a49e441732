"""Naming things: once a thing has moved as one rigid piece, ask a VLM what it is.

The physical side decides what counts as one object (common fate); the VLM only names it. A blob that might be three
parts is never sent: the question "what is this?" has no right answer until we know it is one thing.

For each confirmed thing:
    1. find the image that shows it best: of the camera views since the last time something moved, the one where its
       3D points project biggest and fully inside the frame (usually the wrist camera, ~240 mm away)
    2. crop around it (square, with a margin, scaled up) and put the full view next to it with the thing outlined
    3. tell the model what the depth camera measured (size, height, volume): from a crop alone it can't judge scale
    4. ask in the background (a thread per call), so the arm keeps working; answers are picked up by poll()
       (the free NVIDIA endpoint often answers 500 or times out: each call is retried, then tried on the next model;
       thinking is off: Nemotron gave the same labels in ~3 s instead of ~90 s)

The label sticks to the thing (its id), not to a picture: the thing's 3D model is tracked when it gets bumped, so the
name follows it. When all things are named, plan_sort() asks once, in text only, how to group them for the instruction.

Every crop is saved with its question and answer (runs/<run>/labels/), so models can be compared on the same crops
(lab/label_eval.py).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z
from lab import vlm
from lab.viz import project_to_image

CROP = 448  # px, the close-up's side

LABEL = """This is a photo of something on a robot's sorting table. Left: a close-up (the outlined thing is in the
middle; other things may show at the edges). Right: the camera view it came from, with it outlined in white.
The robot measured it with a depth camera: footprint about {L:.0f} x {W:.0f} mm, {H:.0f} mm tall, volume about {V:.0f} mm³.
{status}
Answer in JSON only:
{{
  "name": "short common name, as specific as you can tell (e.g. '10 mm cube', 'M6 screw', 'six-sided die')",
  "category": "the kind of thing, for sorting (e.g. 'fastener', 'building block', 'game piece')",
  "color": "main colour",
  "material": "best guess",
  "count": 1,
  "confidence": 0.0,
  "attached": false,
  "why": "one sentence: what in the image tells you"
}}
"count": how many separate objects are inside the outline (a clump of touching things is more than 1).
"confidence": 0 to 1, how sure you are of BOTH the name and the count. Be honest: 0.9+ only if it's a familiar
object you'd bet on; low if it's odd, partly hidden, or might be two things.
"attached": true if it looks like separate parts joined together (a screw in a nut, something stuck to it)."""

SORT = """A robot has laid out these objects on a table, one at a time, and named each from a close-up photo:
{things}
The instruction is: "{instruction}".
Group them for that instruction. Every id goes in exactly one group. Answer in JSON only:
{{
  "rule": "the sorting rule, one sentence",
  "groups": [{{"name": "group name", "ids": [1, 2]}}],
  "unsure": [{{"id": 0, "why": "why this one is hard to place"}}]
}}"""


def measure(points: np.ndarray) -> dict:
    """Footprint (long x short side, mm, from the tightest rectangle round it), height and a rough volume."""
    p = points[(points[:, 2] - SURFACE_Z) * 1000 > 2.5, :3]
    (_, _), (a, b), _ = cv2.minAreaRect((p[:, :2] * 1000).astype(np.float32))
    return {"L": max(a, b), "W": min(a, b), "H": float((p[:, 2].max() - SURFACE_Z) * 1000)}


def best_view(pts: np.ndarray, views: list[dict], cams: dict) -> tuple[dict, str, np.ndarray] | None:
    """(obs, camera, pixels) where the thing looks biggest and is fully in frame; None if no view shows it."""
    best, best_size = None, 0.0
    for obs in views:
        for cam in ("wrist", "table"):
            img = obs["images"][cam]
            h, w = img.shape[:2]
            uv = project_to_image(cam, cams, pts, (w, h), joints=obs.get("joints"))
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if inside.mean() < 0.95:
                continue
            size = float(np.ptp(uv[:, 0]) * np.ptp(uv[:, 1]))
            if size > best_size:
                best, best_size = (obs, cam, uv), size
    return best


def compose(img: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the picture sent: close-up | full view with outline, the close-up alone)."""
    h, w = img.shape[:2]
    lo, hi = uv.min(0), uv.max(0)
    c, half = (lo + hi) / 2, max(float((hi - lo).max()) * 0.8, 24.0)  # 60 % margin each side, at least 48 px across
    x0, y0 = int(np.clip(c[0] - half, 0, max(w - 2 * half, 0))), int(np.clip(c[1] - half, 0, max(h - 2 * half, 0)))
    s = int(min(2 * half, w, h))
    close = cv2.resize(img[y0:y0 + s, x0:x0 + s], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
    full = img.copy()
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.polylines(full, [hull], True, (255, 255, 255), 2)
    cv2.rectangle(full, (x0, y0), (x0 + s, y0 + s), (255, 255, 255), 1)
    full = cv2.resize(full, (int(w * CROP / h), CROP))
    return np.hstack([close, full]), close


# best on our crops (lab/label_eval.py, Oct 23): nano omni 4/5 names in ~16 s; super 3/5 ~15 s; llama 90b 4/5 but 1-5 min
FALLBACKS = ["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nvidia/nemotron-3.5-super-vl-120b-a12b",
             "meta/llama-3.2-90b-vision-instruct"]


def usable(answer) -> bool:
    """A real answer, not the question's template echoed back (small models do that)."""
    return isinstance(answer, dict) and bool(answer.get("name")) and "short common name" not in str(answer.get("name"))


class Labeler:
    def __init__(self, cams: dict, model: str | None = None, save_dir: Path | None = None):
        self.cams, self.model, self.save_dir = cams, model, save_dir
        self.models = list(dict.fromkeys(([vlm.model_id(model)] if model else []) + FALLBACKS))
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.pending: dict[int, Future] = {}
        self.asked: dict[int, tuple[str, int]] = {}  # id -> (status, footprint mm²) when last asked
        self.views: list[dict] = []  # observations since the last time something moved

    @property
    def on(self) -> bool:
        return vlm.configured(self.model)

    def saw(self, obs: dict) -> None:
        self.views = self.views[-7:] + [obs]

    def moved(self) -> None:
        """Something is about to move: older views no longer show where things are."""
        self.views = []

    def due(self, t) -> bool:
        """Never asked, or it has changed since: it was a blob then and is confirmed now (a cleaner picture of one
        object), or its footprint grew or shrank by a third (part of it was carried off: the old name is stale)."""
        if t.id in self.pending:
            return False
        if t.id not in self.asked:
            return True
        status, area = self.asked[t.id]
        return (status != "one" and t.status == "one") or not (0.67 < t.area / max(area, 1) < 1.5)

    def stale(self, t) -> bool:
        """Its name is out of date or a new one is on the way."""
        return t.id in self.pending or self.due(t)

    def needs_look(self, t) -> bool:
        """Due a question, and no recent picture shows it whole: the arm should go and look at it."""
        return self.on and self.due(t) and best_view(self._pts(t), self.views, self.cams) is None

    @staticmethod
    def _pts(t) -> np.ndarray:
        return t.model.samples(600) if t.model is not None else t.points[:, :3]

    def ask(self, t) -> str | None:
        """Start naming a thing in the background. Returns what to narrate, or None if it can't be asked yet."""
        if not self.on or not self.due(t):
            return None
        v = best_view(self._pts(t), self.views, self.cams)
        if v is None:
            return None
        obs, cam, uv = v
        picture, close = compose(obs["images"][cam], uv)
        m = measure(t.points)
        m["V"] = t.model.world_mesh().volume * 1e9 if t.model is not None else m["L"] * m["W"] * m["H"]
        status = {"one": "The robot has moved it and it moved as one rigid piece: it IS one object.",
                  "blob": "Nobody has touched it yet: it might be one object or several touching.",
                  "bin": "It looks like a container."}.get(t.status, "")
        prompt = LABEL.format(**m, status=status)
        again = t.id in self.asked
        self.asked[t.id] = (t.status, t.area)
        self.pending[t.id] = self.pool.submit(self._call, t.id, picture, close, cam, prompt, m)
        return (f"Asking the VLM {'again ' if again else ''}what #{t.id} is (close-up from the {cam} camera, "
                f"{m['L']:.0f} × {m['W']:.0f} × {m['H']:.0f} mm)" + (": it changed since the last answer." if again else "."))

    def _call(self, tid, picture, close, cam, prompt, m) -> dict:
        t0, tries = time.time(), []
        r, model = None, None
        for model in self.models:
            for k in range(2):
                try:
                    r = vlm.ask_image(picture, prompt, model=model, max_tokens=2000, think=False)
                    err = None if usable(r and r["json"]) else "no usable answer: " + (r or {}).get("text", "")[:120]
                except Exception as e:  # a failed call is an answer too: shown, not fatal
                    r, err = None, str(e)[:200]
                tries.append(f"{model.split('/')[-1]}: {err or 'ok'}")
                if err is None:
                    break
                time.sleep(2 + 3 * k)
            if err is None:
                break
        out = {"id": tid, "camera": cam, "measured": m, "seconds": time.time() - t0, "model": model, "tries": tries,
               "answer": r["json"] if err is None else None, "text": (r or {}).get("text", ""), "error": err,
               "close": close, "picture": picture}
        if self.save_dir is not None:
            d = Path(self.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{tid:02d}.jpg"), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
            (d / f"{tid:02d}.json").write_text(json.dumps({k: v for k, v in out.items() if k not in ("close", "picture")}
                                                          | {"prompt": prompt}, indent=1, default=float))
        return out

    def poll(self, wait: float = 0.0) -> list[dict]:
        """Answers that have come back (waiting up to `wait` seconds for the rest)."""
        end = time.time() + wait
        done = []
        for tid, f in list(self.pending.items()):
            try:
                done.append(f.result(timeout=max(0.0, end - time.time())) if wait else f.result(timeout=0) if f.done() else None)
            except Exception:
                done.append(None)
            if done[-1] is None:
                done.pop()
            else:
                del self.pending[tid]
        return done

    def plan_sort(self, things: list, instruction: str) -> dict | None:
        """One text-only call: group the named things for the instruction."""
        rows = []
        for t in things:
            lab = getattr(t, "label", None) or {}
            m = measure(t.points)
            rows.append(f'  #{t.id}: "{lab.get("name", "unnamed")}" ({lab.get("color", "?")}, {lab.get("category", "?")}), '
                        f'{m["L"]:.0f} x {m["W"]:.0f} x {m["H"]:.0f} mm')
        prompt = SORT.format(things="\n".join(rows), instruction=instruction or "Sort these.")
        for model in self.models:
            try:
                r = vlm.ask_text(prompt, model=model, max_tokens=4000, think=False)
            except Exception:
                continue
            if r and r["json"]:
                return r | {"prompt": prompt}
        return None
