"""Save every observation the policy receives (and what it answered), so any run can be replayed into the viewer.

    runs/<YYYYmmdd-HHMMSS>_<tag>/
        meta.json
        rollout_000/
            step_0000.json             joints, tcp_pose, t, trial, the actions returned, how long act() took
            step_0000_rgb_table.jpg    each colour image
            step_0000_depth_wrist.png  each depth image (16-bit PNG, mm, lossless)

Plain files on purpose: you can open any of them without our code.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=RUNS.parent).stdout.strip()
    except OSError:
        return ""


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class Recorder:
    def __init__(self, tag: str = "run", meta: dict | None = None, root: Path = RUNS):
        self.dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps({"tag": tag, "commit": _git_commit(), **_jsonable(meta or {})}, indent=2))
        self.rollout, self.step, self.rdir = -1, 0, None

    def new_rollout(self) -> None:
        self.rollout += 1
        self.step = 0
        self.rdir = self.dir / f"rollout_{self.rollout:03d}"
        self.rdir.mkdir(exist_ok=True)

    def log(self, obs: dict, actions=None, act_s: float | None = None, note: str = "", extra: dict | None = None) -> None:
        if self.rdir is None:
            self.new_rollout()
        stem = self.rdir / f"step_{self.step:04d}"
        for cam, img in (obs.get("images") or {}).items():
            cv2.imwrite(f"{stem}_rgb_{cam}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        for cam, d in (obs.get("depth") or {}).items():
            cv2.imwrite(f"{stem}_depth_{cam}.png", np.asarray(d, np.uint16))
        rec = {"joints": obs.get("joints"), "tcp_pose": obs.get("tcp_pose"), "t": obs.get("t"), "trial": obs.get("trial"),
               "actions": actions, "act_s": act_s, "note": note, **(extra or {})}
        Path(f"{stem}.json").write_text(json.dumps(_jsonable(rec)))
        self.step += 1


def load_step(json_path: Path) -> dict:
    """One recorded step back as an observation dict (images RGB, depth uint16) plus what was recorded with it."""
    rec = json.loads(Path(json_path).read_text())
    stem = str(json_path)[: -len(".json")]
    obs = {k: v for k, v in rec.items()}
    obs["joints"] = np.asarray(rec["joints"], float) if rec.get("joints") is not None else None
    obs["images"], obs["depth"] = {}, {}
    for f in sorted(Path(json_path).parent.glob(Path(stem).name + "_rgb_*.jpg")):
        obs["images"][f.stem.split("_rgb_")[1]] = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    for f in sorted(Path(json_path).parent.glob(Path(stem).name + "_depth_*.png")):
        obs["depth"][f.stem.split("_depth_")[1]] = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    return obs
