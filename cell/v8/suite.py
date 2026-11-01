"""The overnight suite: every menu task, headless, on the Isaac twin - each one recorded for replay and
scored against the simulator's ground truth. The sim half of the sim-to-real comparison.

    cd cell && SAM_REAL=1 uv run python -m v8.suite                # the full list, then repeats
    cd cell && SAM_REAL=1 uv run python -m v8.suite ladder01_move  # just these

Writes runs/<ts>_suite/: one folder per run (replay.rrd, chat.jsonl, before.jpg, after.jpg,
result.json), then summary.json and summary.html (pass/fail, times, before/after pictures, and the
command that opens each recording in Rerun for screen capture).
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CELL = HERE.parent

TASKS = ["ladder32_census", "ladder01_move", "ladder06_pen", "ladder02_bin", "ladder05_bolt",
         "ladder08_pick_named", "ladder03_match_block", "ladder04_match_bin", "ladder31_tallest",
         "ladder21_block_bin", "ladder20_pen_move", "ladder07_shapes", "ladder23_match",
         "ladder24_tools_sort", "ladder30_pens", "ladder11_sorted_piles", "ladder09_sort_ignore",
         "ladder10_sort_pile", "ladder12_sorted_bins"]
REPEATS = ["ladder01_move", "ladder02_bin", "ladder03_match_block"] * 2   # success rates, not anecdotes
CAP_S = 540           # a run that hasn't finished in 9 minutes is stopped (and scored as it stands)
RESTART_EVERY = 4     # fresh Isaac stage every few runs: parked bodies pile up and destabilize physics


def restart_isaac(log) -> None:
    r = subprocess.run(["zsh", str(CELL / "isaac" / "restart.sh")], capture_output=True, text=True, timeout=900)
    ok = "ready on" in (r.stdout + r.stderr)
    log(f"isaac restart: {'ready' if ok else 'FAILED - ' + (r.stdout + r.stderr)[-300:]}")
    if not ok:
        raise RuntimeError("the Isaac server would not come up")


def table_jpg(cell, path: Path) -> None:
    img = cell.observe(cams=("table",))["images"]["table"]
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_RGB2BGR))


def truth_now(cell) -> tuple[dict, dict]:
    """(name -> mesh in the base frame, id -> pose) from the simulator itself."""
    import trimesh
    poses = cell._post("sim/truth", {})["poses"]
    meshes = {}
    for i in cell.instances:
        v = poses.get(str(i.id))
        if v is None or i.id not in cell.local:
            continue
        T = trimesh.transformations.quaternion_matrix(v[3:7])
        T[:3, 3] = v[:3]
        meshes[f"{i.color} {i.kind} #{i.id}"] = cell.local[i.id].copy().apply_transform(T)
    return meshes, poses


def run_one(task: str, out: Path, log) -> dict:
    import rerun as rr
    from lab import ladder as LD
    from lab import primitives as P
    from lab.isaaccell import IsaacCell
    from lab.robot import SPEEDS
    from v8 import view as V
    from v8.agent import Deps, Mind
    from v8.eyes import Eyes
    from v8.stage import staged

    out.mkdir(parents=True, exist_ok=True)
    t = LD.load_trial(task)
    rr.init(f"v8_{task}", recording_id=out.name)
    rr.set_sinks(rr.FileSink(str(out / "replay.rrd")))
    V.setup()
    P.TRAVEL, P.APPROACH = SPEEDS["sim"]
    cell = IsaacCell(staged(task, log))
    eyes = Eyes(cell, V.View())
    mind = Mind()
    chat = []

    def say(line: str) -> None:
        chat.append({"t": time.time(), "who": "robot", "text": line})
        with open(out / "chat.jsonl", "a") as f:
            f.write(json.dumps(chat[-1]) + "\n")

    table_jpg(cell, out / "before.jpg")
    truth0, before = truth_now(cell)
    with open(out / "chat.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "who": "you", "text": t["instruction"]}) + "\n")
    timer = threading.Timer(CAP_S, lambda: setattr(eyes, "stop_now", True))
    timer.start()
    t0 = time.time()
    reply, crash = "", None
    try:
        reply = mind.say(t["instruction"], Deps(eyes=eyes, say=say,
                                                 recent=[c["text"] for c in chat]))
    except Exception as e:   # the suite's one boundary: record the whole cause, keep the night going
        crash = f"{type(e).__name__}: {e}"
        (out / "crash.txt").write_text(traceback.format_exc())
    finally:
        timer.cancel()
    secs = time.time() - t0
    say(reply or f"(no reply: {crash})")
    time.sleep(2.0)   # let things settle before the truth is read
    table_jpg(cell, out / "after.jpg")
    _, after = truth_now(cell)
    goals = t.get("goal") or []
    lines = LD.check(goals, truth0, before, after) if goals else []
    acts = [c["text"] for c in chat if c["text"].startswith("(")]
    res = {"task": task, "instruction": t["instruction"], "scored": bool(goals),
           "ok": (all(x for x, _ in lines) if goals else None),
           "checks": [{"ok": bool(x), "says": s} for x, s in lines],
           "seconds": round(secs), "capped": eyes.stop_now and not crash, "crash": crash,
           "grabs": sum("grabbing" in a for a in acts), "drops": sum("dropping" in a for a in acts),
           "inspects": sum("inspecting" in a for a in acts), "looks": sum("look" in a for a in acts),
           "refusals": sum("refused" in a for a in acts), "misses": sum("MISS" in a for a in acts),
           "moved_ok": sum("confirmed at" in a for a in acts), "inside_ok": sum("INSIDE" in a for a in acts),
           "blobs": len(eyes.blobs), "reply": reply[:600]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def write_summary(root: Path, results: list[dict]) -> None:
    (root / "summary.json").write_text(json.dumps(results, indent=1))
    scored = [r for r in results if r.get("scored") and r.get("ok") is not None]
    passed = sum(r["ok"] for r in scored)
    cards = []
    for r in results:
        d = r["dir"]
        state = ("CRASH" if r.get("crash") else "unscored" if not r.get("scored")
                 else "PASS" if r.get("ok") else "FAIL")
        checks = "<br>".join(f"{'✓' if c['ok'] else '✗'} {html.escape(c['says'])}" for c in r.get("checks", []))
        cmd = f"cd ~/yaad/team-repo-dev/cell && uv run rerun {root.relative_to(CELL)}/{d}/replay.rrd"
        cards.append(f'''<section class="card {state.lower()}"><header><b>{html.escape(r["task"])}</b>
 · <span class="st">{state}</span> · {r.get("seconds", "?")} s{" (capped)" if r.get("capped") else ""}
 · grabs {r.get("grabs", 0)}, drops {r.get("drops", 0)}, confirmed moves {r.get("moved_ok", 0)}, inside {r.get("inside_ok", 0)},
 refusals {r.get("refusals", 0)}, misses {r.get("misses", 0)}</header>
<div class="task">“{html.escape(r.get("instruction", ""))}”</div>
<div class="row"><figure><img src="{d}/before.jpg"><figcaption>before</figcaption></figure>
<figure><img src="{d}/after.jpg"><figcaption>after</figcaption></figure>
<div class="checks">{checks or html.escape(r.get("crash") or "no goal to score (perception task)")}
<div class="reply">{html.escape(r.get("reply", ""))}</div>
<button class="cp" data-cmd="{html.escape(cmd)}">copy replay command</button></div></div></section>''')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>V8 sim suite</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{background:#101318;color:#dde2e8;font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px}}
.card{{background:#181c22;border:1px solid #2a2f37;border-left:5px solid #555;border-radius:10px;padding:10px 12px;margin-bottom:12px}}
.card.pass{{border-left-color:#2f9a55}} .card.fail{{border-left-color:#b84a3a}} .card.crash{{border-left-color:#d08a20}}
.task{{font-style:italic;margin:4px 0}} .row{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0}}
img{{width:320px;border-radius:6px;display:block}} figcaption{{font-size:11px;color:#8a929c;text-align:center}}
.checks{{flex:1;min-width:260px;font-size:13px}} .reply{{color:#8a929c;font-size:12.5px;margin:6px 0}}
.cp{{background:#222831;color:#dde2e8;border:1px solid #2a2f37;border-radius:5px;padding:3px 9px;cursor:pointer}}
.st{{font-weight:600}}</style></head><body>
<h1 style="font-size:20px;margin:0">V8 on the Isaac twin: {passed}/{len(scored)} scored runs passed</h1>
<div style="color:#8a929c;margin-bottom:12px">{len(results)} runs · scored against the simulator's ground truth by lab.ladder.check · each run is a Rerun recording (copy the command, paste in a terminal, screen-record)</div>
{''.join(cards)}
<script>document.querySelectorAll('.cp').forEach(b=>b.onclick=()=>{{navigator.clipboard&&navigator.clipboard.writeText(b.dataset.cmd);
b.textContent='copied';setTimeout(()=>b.textContent='copy replay command',1200);}});</script></body></html>"""
    (root / "summary.html").write_text(page)


def main() -> None:
    from lab.recorder import RUNS
    tasks = sys.argv[1:] or (TASKS + REPEATS)
    root = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_suite"
    root.mkdir(parents=True)

    def log(s: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        with open(root / "suite.log", "a") as f:
            f.write(line + "\n")

    log(f"suite: {len(tasks)} runs -> {root}")
    results = []
    for k, task in enumerate(tasks):
        d = f"{k:02d}_{task}"
        if k % RESTART_EVERY == 0:
            try:
                restart_isaac(log)
            except Exception as e:
                log(f"restart failed: {e}")
        for attempt in (1, 2):
            try:
                res = run_one(task, root / d, log)
                break
            except Exception as e:   # staging or the twin itself fell over: one fresh restart, then give up
                (root / d).mkdir(parents=True, exist_ok=True)
                (root / d / "crash.txt").write_text(traceback.format_exc())
                log(f"{task}: setup crashed ({type(e).__name__}: {e})" + ("; restarting Isaac" if attempt == 1 else ""))
                res = {"task": task, "crash": f"{type(e).__name__}: {e}", "scored": False, "ok": None}
                if attempt == 1:
                    try:
                        restart_isaac(log)
                    except Exception as e2:
                        log(f"restart failed: {e2}")
        res["dir"] = d
        results.append(res)
        state = "CRASH" if res.get("crash") else "unscored" if not res.get("scored") else ("PASS" if res.get("ok") else "FAIL")
        log(f"[{k + 1}/{len(tasks)}] {task}: {state} in {res.get('seconds', '?')} s "
            f"(grabs {res.get('grabs', 0)}, confirmed {res.get('moved_ok', 0)}, inside {res.get('inside_ok', 0)})")
        write_summary(root, results)
    scored = [r for r in results if r.get("scored") and r.get("ok") is not None]
    log(f"done: {sum(r['ok'] for r in scored)}/{len(scored)} scored runs passed -> {root / 'summary.html'}")


if __name__ == "__main__":
    main()
