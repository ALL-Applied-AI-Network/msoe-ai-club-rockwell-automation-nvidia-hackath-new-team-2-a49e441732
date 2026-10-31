"""V8, step zero: the bare cell console. Connect to a cell, see the table, use your hands. Nothing else.

    uv run --directory cell python -m v8.app        # http://127.0.0.1:8801

Deliberately nothing else: no agent, no robot thread, no fallbacks, no auto-connect. The Rerun pane
is served EMPTY on purpose - it fills only when V8 grows eyes, and we will wire that knowingly.
A button either does its thing or shows the error it actually got, whole.

Kept services: the Isaac cell client (lab/isaaccell.py), the real cell client (lab/realcell.py),
the parts catalog (lab/catalog.py) and the trial scenes (lab/trial.py). Nothing from the old loop.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
PORT = 8801
GRPC, WEB = 9877, 9091   # beside the old app's 9876/9090, so both can run while you switch over


def main() -> None:
    import rerun as rr
    from lab import ladder as LD
    from lab.catalog import COLORS, KINDS

    # The empty viewer: served so the page can embed it, with nothing in it. That is the honest
    # state of V8's head right now.
    rr.init("v8")
    rr.set_sinks(rr.GrpcServerSink(bind_ip="127.0.0.1", port=GRPC))
    uri = f"rerun+http://127.0.0.1:{GRPC}/proxy"
    rr.serve_web_viewer(web_port=WEB, open_browser=False, connect_to=uri)
    viewer = f"http://127.0.0.1:{WEB}/?url={quote(uri, safe='')}"

    build = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, cwd=HERE).stdout.strip() or "?"

    st = {"mode": None, "cell": None, "task": None, "log": [], "chat": [], "busy": False}
    lock = threading.Lock()

    from v8.agent import Deps, Mind
    from v8.eyes import Eyes
    from v8 import view as v8view
    mind = Mind()   # it talks, remembers, and has eyes (look/inspect) once a cell is connected
    mind_lock = threading.Lock()
    st["eyes"] = None
    vw = v8view.View()

    def say(text: str) -> None:
        st["log"] = (st["log"] + [{"t": time.time(), "text": text}])[-30:]
        print(text, flush=True)

    def table() -> list[dict]:
        cell = st["cell"]
        if st["mode"] != "sim" or cell is None:
            return []
        out = []
        for i in cell.instances:
            p = cell.poses.get(i.id)
            if p is None:
                continue
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color,
                        "box": i.kind.startswith("box"),
                        "x": round(float(p[0]) * 1000), "y": round(float(p[1]) * 1000)})
        return out

    names = sorted(f.stem for f in LD.TRIALS.glob("ladder*.yaml"))
    tasks = [{"name": n, "instruction": LD.load_trial(n)["instruction"]} for n in names]

    def wipe() -> None:
        """Reset and Clear both mean ALL of it: the mind's memory, the chat, the blobs, the viewer.
        A fresh start that remembers old views isn't fresh (they lump into the new world)."""
        if st["eyes"] is not None:
            st["eyes"].stop_now = True    # the running job halts at its next motion gate
        mind.forget()
        st["chat"] = []
        v8view.clear()
        if st["cell"] is not None and st["mode"] == "sim":
            st["eyes"] = Eyes(st["cell"], vw)

    def staged(trial_name: str):
        """Build a trial's scene with everything INSIDE the arm's envelope from birth: fixtures
        (bins) get their declared spots clamped in the spec, and randomly-spread objects re-roll
        their layout seed until all land inside. (Editing positions after the physics settles does
        nothing: Isaac imposes the SETTLED body poses, which is how a clamped bolt stayed at 492.)"""
        import copy
        import numpy as np
        from lab.trial import build as build_scene
        t = copy.deepcopy(LD.load_trial(trial_name))
        moved = 0
        for f in t.get("fixtures", []):
            x, y = float(f["at"][0]), float(f["at"][1])
            r = float(np.hypot(x, y)) * 1000
            if r > 440:
                f["at"] = [x * 435.0 / r, y * 435.0 / r]
                moved += 1
        sim = None
        for seed in [t.get("seed", 0), *range(1, 24)]:
            sim = build_scene(t, seed)
            if all(float(np.hypot(i.pos[0], i.pos[1])) * 1000 <= 445 for i in sim.instances):
                break
        else:
            say("staging: no all-inside layout found in 24 rolls; using the last one")
        if moved:
            say(f"staging: pulled {moved} fixture(s) inside the reach ring")
        return sim

    def connect_sim() -> str:
        from lab.isaaccell import IsaacCell
        from lab import primitives as P
        from lab.robot import SPEEDS
        P.TRAVEL, P.APPROACH = SPEEDS["sim"]
        with lock:
            st["cell"] = IsaacCell(staged("chat_table"))
            st["mode"] = "sim"
            st["eyes"] = Eyes(st["cell"], vw)
            v8view.setup()
        return "connected to the Isaac simulator: the starting table is set"

    def connect_real(b: dict) -> str:
        from lab.realcell import RealCell
        from lab.robot import _approved_token
        url, token, ca = b.get("url") or "", (b.get("token") or "").strip(), b.get("ca") or None
        if not token:
            url, token = _approved_token(url, ca)
        cell = RealCell(url, token, ca)
        rep = cell.check()
        from lab import primitives as P
        from lab.robot import SPEEDS
        P.TRAVEL, P.APPROACH = SPEEDS["real"]   # the speeds the real arm was tuned with (150/50 mm/s)
        with lock:
            st["cell"], st["mode"] = cell, "real"
            st["eyes"] = Eyes(cell, vw)
            v8view.setup()
        return (f"connected to the real cell: {rep['round_trip_ms']} ms round trip, a picture in "
                f"{rep['observe_s']} s, cameras {'ok' if rep['sees'] else 'NOT seeing'}")

    def hands(b: dict) -> str:
        if st["mode"] != "sim":
            raise RuntimeError("hands work on the simulator; on the real cell your real hands do that")
        cell = st["cell"]
        a = b.get("action")
        with lock:
            if a == "pour":
                order = [(it["kind"], it["color"]) for it in b.get("items", [])
                         for _ in range(int(it.get("count", 1)))]
                if not order:
                    return "nothing picked to pour"
                new = cell.pour(order)
                return f"poured {len(new)} thing(s) onto the table"
            if a == "take":
                cell.take([int(b["id"])])
                return "took it off the table"
            if a == "tip":
                out = cell.tip_box(int(b["id"]))
                return f"tipped {len(out)} thing(s) out"
            if a == "add_box":
                i = cell.add_box(b.get("color", "white"))
                p = cell.poses[i.id]
                return f"added a {b.get('color', 'white')} box at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f})"
        raise ValueError(f"unknown hands action {a!r}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                data = (HERE / "app.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/state":
                out = {"mode": st["mode"], "build": build, "viewer": viewer, "task": st["task"],
                       "busy": st["busy"],
                       "table": table(), "tasks": tasks, "chat": st["chat"],
                       "model": mind.model_name,
                       "kinds": [k for k in KINDS if not k.startswith("box")],
                       "boxes": [k for k in KINDS if k.startswith("box")],
                       "colors": sorted(COLORS), "log": st["log"]}
                if st["eyes"] is not None:
                    out["objects"] = st["eyes"].roster()
                if st["mode"] == "real":
                    try:
                        out["trials"] = [{"id": t["trial_id"], "status": t["status"]}
                                         for t in st["cell"].trials()]
                    except Exception as e:  # the poll must not kill the page; the error is SHOWN, whole
                        out["trials_error"] = f"{type(e).__name__}: {e}"
                self._json(out)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            try:
                if self.path == "/connect":
                    msg = connect_sim() if body.get("target") == "sim" else connect_real(body)
                elif self.path == "/reset":
                    if st["mode"] != "sim":
                        raise RuntimeError("reset is the simulator's; the real table is reset by hand")
                    msg = connect_sim()
                    wipe()
                    msg = "reset: " + msg + "; mind, chat and world model cleared"
                elif self.path == "/hands":
                    msg = hands(body)
                elif self.path == "/say":
                    text = (body.get("text") or "").strip()
                    if not text:
                        return self._json({"error": "say something"}, 400)
                    st["chat"].append({"who": "you", "text": text})
                    if st["busy"]:   # mid-job: the side-channel voice answers NOW; the hands keep working
                        recent = [c["text"] for c in st["chat"][-14:] if c["who"] == "robot"]
                        reply = mind.comment(text, st["eyes"], recent)
                        st["chat"].append({"who": "robot", "text": f"(answering while I work) {reply}"})
                        return self._json({"reply": reply})
                    st["busy"] = True
                    try:
                        with mind_lock:
                            reply = mind.say(text, Deps(eyes=st["eyes"],
                                                        say=lambda t: st["chat"].append({"who": "robot", "text": t}),
                                                        recent=[f"[{c['who']}] {c['text'][:160]}"
                                                                for c in st["chat"][-30:]]))
                        st["chat"].append({"who": "robot", "text": reply})
                    except Exception as e:   # the boundary: the crash lands IN THE CHAT, whole cause named
                        print(traceback.format_exc(), flush=True)
                        st["chat"].append({"who": "robot",
                                           "text": f"I CRASHED: {type(e).__name__}: {e} (full trace in the app log)"})
                        return self._json({"error": str(e)}, 500)
                    finally:
                        st["busy"] = False
                    return self._json({"reply": reply})
                elif self.path == "/flash":
                    from v8 import view as _v
                    b = next((b for b in (st["eyes"].blobs if st["eyes"] else []) if b.id == int(body["id"])), None)
                    if b is None:
                        return self._json({"error": "no such blob"}, 404)
                    _v.flash(b, st["eyes"].grid)
                    return self._json({"ok": True})
                elif self.path == "/trial/start":
                    if st["mode"] != "real":
                        raise RuntimeError("trials are the real cell's; connect to it first")
                    trials = st["cell"].trials()
                    pick = (next((t for t in trials if t["status"] == "ready"), None)
                            or next((t for t in trials if t["status"] == "waiting"), None))
                    if pick is None:
                        raise RuntimeError("no ready or waiting trial on the cell")
                    t = st["cell"].start_trial(pick["trial_id"])   # the CLIENT starts it: waiting is fair game
                    msg = f"trial {t['trial_id']} is running"
                elif self.path == "/chat/clear":
                    wipe()
                    msg = "cleared: chat, memory and the world model - it knows nothing now"
                elif self.path == "/task":
                    name = body.get("name")
                    t = next((t for t in tasks if t["name"] == name), None)
                    if t is None:
                        return self._json({"error": f"no task {name!r}"}, 404)
                    if st["mode"] == "real":   # the REAL table is staged by hands: a task click must
                        st["task"] = name      # never yank the session back into the simulator
                        say(f"real cell: {name} selected - the instruction is in the chat box; Enter sends it")
                        return self._json({"msg": "instruction loaded (real cell: no staging)",
                                           "instruction": t["instruction"]})
                    from lab.isaaccell import IsaacCell
                    if st["eyes"] is not None:
                        st["eyes"].stop_now = True
                    with lock:
                        st["cell"] = IsaacCell(staged(name))
                        st["mode"] = "sim"
                        st["eyes"] = Eyes(st["cell"], vw)
                    mind.forget()
                    st["chat"] = []
                    v8view.clear()
                    v8view.setup()
                    st["task"] = name
                    msg = f"set up {name}: the table is staged. Instruction: {t['instruction']}"
                    say(msg)
                    return self._json({"msg": msg, "instruction": t["instruction"]})
                else:
                    return self._json({"error": "not found"}, 404)
            except Exception as e:
                # the one boundary: the server must answer. The WHOLE error reaches the page and the log.
                print(traceback.format_exc(), flush=True)
                msg = f"{type(e).__name__}: {e}"
                say(msg)
                return self._json({"error": msg}, 500)
            say(msg)
            self._json({"msg": msg})

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    say(f"v8 console (build {build}): http://127.0.0.1:{PORT} - not connected to anything yet")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
