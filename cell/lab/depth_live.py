"""The Mac's side of live depth: start the depth service on the GPU box, and feed it every picture with its pose.

    link = DepthLink.start()                 # (re)start isaac/depth_server.py on the box, tunnel to it, reset it
    fwd = Forward(pull, link, T_fc)          # a thread: every picture whose capture time is known goes over, posed
    link.grid()                              # voxels with evidence for something

Pictures are posed here, from lab/pull.py (when each was taken) and the joints then: the box only does depth. The cell's
token never leaves the Mac.
"""
from __future__ import annotations

import base64
import queue
import socket
import subprocess
import threading
import time

import numpy as np
import requests

from calibration.geometry import flange_pose
from lab.isaaccell import SSH

PORT = 8767
REMOTE = "~/cell-isaac"


def _up() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def deploy(restart: bool = True) -> None:
    """Copy isaac/depth_server.py and calibration/ to the box and (re)start the service there."""
    from pathlib import Path
    cell = Path(__file__).resolve().parents[1]
    subprocess.run(["rsync", "-a", "--exclude", "__pycache__", str(cell / "calibration"), str(cell / "isaac"),
                    f"{SSH}:cell-isaac/"], check=True)
    if restart:
        # two calls: a kill pattern in the same command line as the start would match (and kill) that shell too
        subprocess.run(["ssh", SSH, "pkill -f '[d]epth_server.py'; rm -f " + REMOTE + "/isaac/depth.log; true"], check=True)
        time.sleep(1)
        subprocess.Popen(["ssh", "-n", SSH, f"cd {REMOTE} && (setsid nohup ~/cosmos-env/bin/python isaac/depth_server.py "
                          "> isaac/depth.log 2>&1 </dev/null &)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)  # don't wait: ssh can hang on to the detached process
        for _ in range(60):
            time.sleep(1)
            out = subprocess.run(["ssh", SSH, f"grep -m1 -E 'ready on|Traceback' {REMOTE}/isaac/depth.log"],
                                 capture_output=True, text=True).stdout
            if "ready" in out:
                return
            if "Traceback" in out:
                raise RuntimeError(subprocess.run(["ssh", SSH, f"tail -20 {REMOTE}/isaac/depth.log"],
                                                  capture_output=True, text=True).stdout)
        raise RuntimeError("the depth service did not start")


def tunnel() -> None:
    if _up():
        return
    subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                      "-L", f"{PORT}:127.0.0.1:{PORT}", SSH], stdin=subprocess.DEVNULL)
    for _ in range(50):
        if _up():
            return
        time.sleep(0.2)
    raise RuntimeError("no tunnel to the depth service")


class DepthLink:
    def __init__(self):
        self.url = f"http://127.0.0.1:{PORT}/"
        self.s = requests.Session()
        self.q: queue.Queue = queue.Queue()
        self.sent = 0
        threading.Thread(target=self._send_loop, daemon=True).start()

    @classmethod
    def start(cls, lens: dict, restart: bool = True) -> "DepthLink":
        deploy(restart)
        tunnel()
        link = cls()
        link.s.post(link.url + "reset", json=lens, timeout=30).raise_for_status()
        return link

    def send(self, fid, t: float, T: np.ndarray, jpeg: bytes) -> None:
        self.q.put({"id": fid, "t": t, "T": np.asarray(T, float).ravel().tolist(),
                    "jpeg": base64.b64encode(jpeg).decode()})

    def _send_loop(self) -> None:
        s = requests.Session()
        while True:
            body = self.q.get()
            try:
                s.post(self.url + "frame", json=body, timeout=30)
                self.sent += 1
            except requests.RequestException as e:
                print(f"depth: a frame did not go over: {e}")

    def grid(self, sure: float = 2.2) -> dict:
        return self.s.get(self.url + "grid", params={"sure": sure}, timeout=60).json()

    def stats(self) -> dict:
        return self.s.get(self.url + "stats", timeout=30).json()


class Forward:
    """Every EVERY s: refit the camera's beat, and send each picture whose capture time is known and whose joints
    have been heard after it (so the pose is interpolated, not extrapolated). Waits until pull.delay is known."""
    EVERY = 0.5

    def __init__(self, pull, link: DepthLink, T_fc: np.ndarray):
        self.pull, self.link, self.T_fc = pull, link, T_fc
        self.sent: set[str] = set()
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def stop(self) -> None:
        self.stop_ev.set()
        self.th.join(timeout=5)

    def flush(self) -> int:
        """Send what can be sent now; returns how many went."""
        if self.pull.delay is None or len(self.pull.frames) < 20:
            return 0
        self.pull.fit()
        with self.pull.lock:
            t_last = self.pull.jt[-1] if self.pull.jt else -1e18
            todo = [f for f in self.pull.frames if f.hash not in self.sent and f.tc is not None and f.tc < t_last - 0.05]
        todo.sort(key=lambda f: f.tc)
        for f in todo:
            T = flange_pose(self.pull.joints_at(f.tc)[0]) @ self.T_fc
            self.link.send(f.hash, f.tc, T, f.jpeg)
            self.sent.add(f.hash)
        return len(todo)

    def _run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                self.flush()
            except Exception as e:  # keep forwarding
                print(f"forward: {type(e).__name__}: {e}")
            time.sleep(self.EVERY)
