"""Talk to the robot: a chat page next to its mind (Rerun), with "Joel's hands" to change the table.

    uv run python -m lab.app            # then http://127.0.0.1:8800 (opens by itself)

Frontend: this page (chat, the hands panel, Rerun's web viewer). Backend: lab/robot.py (the body and mind, in its own
thread) and lab/chat.py (its voice). The "real world": the Isaac cell on the GPU box (isaac/run_server.sh there; the
window on that box's monitor is what you'd see standing next to the arm).

The page streams everything (your messages, the robot's replies, what it does, the VLM's view of the table) as
Server-Sent Events from /events.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
PORT = 8800


class Board:
    """Every message on the page, in order; /events streams them."""

    def __init__(self):
        self.items: list[dict] = []
        self.cv = threading.Condition()

    def add(self, who: str, text: str) -> None:
        with self.cv:
            self.items.append({"n": len(self.items), "who": who, "text": text, "t": time.time()})
            self.cv.notify_all()

    def after(self, n: int, wait: float = 15.0) -> list[dict]:
        with self.cv:
            if len(self.items) <= n:
                self.cv.wait(wait)
            return self.items[n:]


def main() -> None:
    from lab.chat import Chat
    from lab.robot import Robot
    board = Board()
    board.add("system", "Connecting to the Isaac cell and setting up the robot...")
    robot = Robot()
    chat = Chat(robot, board.add)
    state = {"distract": False, "last_rule": None}

    def on_event(kind: str, text: str) -> None:
        """The robot's own news: shown as a log line; the ones a person cares about also go to the chat agent."""
        if kind == "plan":
            rule = text.split("Rule: ")[-1] if "Rule: " in text else text
            board.add("log", f"plan: {text}")
            if rule == state["last_rule"]:
                return
            state["last_rule"] = rule
        elif kind != "done":
            board.add("log", text if kind != "move" else f"→ {text}")
        if kind in ("plan", "done", "seen", "check") or (kind == "note" and not text.startswith("Sorted:")):
            threading.Thread(target=chat.event, args=(kind, text), daemon=True).start()

    robot.listeners.append(on_event)
    viewer = f"http://127.0.0.1:9090/?url={quote(robot.viewer_uri, safe='')}"
    board.add("system", "Ready. The Isaac window on the GPU box is the real world; the right side here is what I think.")

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
            elif self.path == "/config":
                self._json({"viewer": viewer, "distract": state["distract"]})
            elif self.path == "/objects":
                self._json({"objects": robot.objects(), "state": robot.state, "t": robot.cell.t})
            elif self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                n = 0
                try:
                    while True:
                        for it in board.after(n):
                            self.wfile.write(f"data: {json.dumps(it)}\n\n".encode())
                            n = it["n"] + 1
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/say":
                text = (body.get("text") or "").strip()
                if text:
                    board.add("you", text)
                    if state["distract"]:
                        robot.distract(2.0)
                    threading.Thread(target=chat.user, args=(text,), daemon=True).start()
                self._json({})
            elif self.path == "/hands":
                try:
                    msg = hands(body)
                except Exception as e:
                    msg = f"couldn't: {e}"
                board.add("hands", msg)
                self._json({"ok": True, "msg": msg})
            elif self.path == "/distract":
                state["distract"] = bool(body.get("on"))
                self._json({"distract": state["distract"]})
            elif self.path == "/native":
                rerun = Path(sys.executable).parent / "rerun"
                subprocess.Popen([str(rerun), robot.viewer_uri], stdin=subprocess.DEVNULL)
                self._json({})
            else:
                self._json({"error": "not found"}, 404)

    def hands(b: dict) -> str:
        a = b.get("action")
        if a == "pour":
            order = [(it["kind"], it["color"]) for it in b.get("items", []) for _ in range(int(it.get("count", 1)))]
            return "You " + robot.pour(order) if order else "nothing picked to pour"
        if a == "take":
            return "You " + robot.take([int(b["id"])])
        if a == "tip":
            return "You " + robot.tip_box(int(b["id"]))
        if a == "add_box":
            return "You " + robot.add_box(b.get("color", "white"))
        return f"unknown action {a}"

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"talk to the robot: http://127.0.0.1:{PORT}", flush=True)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
