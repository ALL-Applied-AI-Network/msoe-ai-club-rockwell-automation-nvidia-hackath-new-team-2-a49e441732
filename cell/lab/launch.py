"""A tiny menu: open a recorded run in Rerun, or start a trial (it opens live). Throwaway convenience.

    uv run python -m lab.launch
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from lab.recorder import RUNS
from lab.trial import TRIALS

CELL = Path(__file__).resolve().parents[1]


def runs() -> list[Path]:
    return sorted((d for d in RUNS.iterdir() if (d / "replay.rrd").exists()), key=lambda d: d.name, reverse=True)


def describe(d: Path) -> str:
    f = d / "summary.json"
    if not f.exists():
        return d.name
    s = json.loads(f.read_text())
    sc = s.get("score") or {}
    res = f"{sc.get('separated', '?')}/{sc.get('objects', '?')} separated" if sc else "no score"
    return f"{s['trial']:15s} seed {s['seed']}  {res:15s} robot {s['robot_s']:4d} s   ({d.name[:15]})"


def main() -> None:
    while True:
        rs, ts = runs()[:20], sorted(f.stem for f in TRIALS.glob("*.yaml"))
        print("\nRUNS (newest first)")
        for i, d in enumerate(rs, 1):
            print(f"  {i:2d}  {describe(d)}")
        print("TRIALS (start one; it opens live)")
        for i, t in enumerate(ts):
            n = sum(o.get("count", 1) for o in yaml.safe_load((TRIALS / f"{t}.yaml").read_text())["objects"])
            print(f"  {chr(97 + i)}   {t} ({n} objects)")
        c = input("\nnumber = open a run, letter = start a trial, q = quit: ").strip().lower()
        if c in ("q", ""):
            return
        if c.isdigit() and 1 <= int(c) <= len(rs):
            subprocess.Popen([sys.executable, "-m", "rerun", str(rs[int(c) - 1] / "replay.rrd")], cwd=CELL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        elif len(c) == 1 and c.isalpha() and ord(c) - 97 < len(ts):
            subprocess.Popen([sys.executable, "-m", "lab.trial", "run", ts[ord(c) - 97]], cwd=CELL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            print(f"started {ts[ord(c) - 97]}: the viewer opens in a moment")


if __name__ == "__main__":
    main()
