"""The robot as a long-running session: a body (the Isaac cell, standing in for the real one), a mind (the sort policy:
the evidence map; v1's sort policy is archived), and a voice (lab/chat.py talks for it). lab/app.py puts it on a page.

Two worlds, one switch: the Isaac cell on the GPU box ("sim": with a Reset, and Joel's hands to change the table), or the
real cell ("real": a run token, a setup check, then the operator's trial; no truth and no hands, the person's real
hands do that). The same mind drives both.

It runs in its own thread and never waits for the chat: a job (an instruction) starts the sort policy; rules from the
person change its plan between moves; when there's nothing to do it watches the table, and a change there (someone
pouring more parts on, taking something away) is reported, not acted on: the chat decides what to do about it.

Everything it narrates goes to Rerun as before (served live: the page embeds the web viewer, and native viewers can
connect too), and the lines that matter to a person standing next to it become events for the chat.
"""
from __future__ import annotations

import collections
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from lab import armcheck, primitives as P, viz
from lab.isaaccell import IsaacCell
from lab.realcell import RealCell, TrialOver
from lab.minds import MINDS
from lab.trial import build, load_trial

RUNS = Path(__file__).resolve().parents[1] / "runs"
SPEEDS = {"sim": (300.0, 70.0), "real": (150.0, 50.0)}  # travel, approach mm/s: on the real arm, the speeds it was
                                                       # tuned with (it lags its targets; the pause before a look is
                                                       # sized for 0.8 s of lag)
WATCH_S = 8.0          # idle: look at the table this often (wall clock)
CHANGE_PX = 120        # this many table-depth pixels changing by > 5 mm: something on the table changed


class Robot:
    def __init__(self, trial: str = "chat_table", seed: int = 0):
        self.trial_name, self.seed = trial, seed
        self.mode = "sim"
        try:
            self.cell = IsaacCell(build(load_trial(trial), seed))
        except (SystemExit, OSError, requests.RequestException) as e:  # no Isaac (unreachable, or reset): MuJoCo stands in
            print(f"{e}; using the MuJoCo cell instead", flush=True)
            self.cell = build(load_trial(trial), seed)
        self.cell.noise = True  # the simulator's depth made as bad as the real sensors'
        P.TRAVEL, P.APPROACH = SPEEDS["sim"]
        self.cell_lock = threading.Lock()  # held by the robot thread while it uses the cell: a switch waits for it
        self.dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_chat"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.viewer_uri = viz.start_served("robot", save_to=self.dir / "replay.rrd")
        viz.truth_setup(self.cell)
        self.anim = viz.Animator(self.cell)
        self.lock = threading.Lock()
        self.events: collections.deque = collections.deque(maxlen=500)  # (time, kind, text): for the chat and the page
        self.listeners: list = []
        viz.LISTENERS.append(self._heard)
        self.instruction, self.rules = "", []
        self.policy = None  # the mind running the current job (lab/minds)
        self.parked = None  # a job paused or stopped by an error: resume() picks it up again
        self.mind_name = "map"
        self.pending: str | None = None   # "start" | "pause": picked up by the robot thread between moves
        self.state = "idle"
        self.distract_s = 0.0             # hold still this long before the next move (talking to it slows it down)
        self.baseline = None              # the table depth when it last finished: changes are measured from it
        self.step = 0
        self.done_log: list[str] = []      # what it has done, in order: "carried #7 into box #2", "raked #4"
        self.looks: list[dict] = []        # the person's looks asked for before a mind was running (look_at)
        self.facts: list[tuple] = []       # (text, items): what the person says is on the table, for every job
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ commands (from the chat or the page; any thread)

    def start(self, instruction: str | None = None, mind: str | None = None) -> str:
        with self.lock:
            pol = self.policy or self.parked
            if getattr(pol, "hold", False) and mind in (None, "focus"):  # it was only looking: now the sort, same map
                pol.hold = False
                if instruction and instruction != "sort these":
                    self.instruction = instruction
                    pol.board.add_user_rule(instruction)
                if self.policy is None:
                    self.policy, self.parked, self.state = pol, None, "working"
                return "sorting now, from what I've already measured"
            if self.policy is not None and self.pending is None:  # already on it: a second "go" must not wipe the work
                return "already working on it"
            if instruction is not None:
                self.instruction = instruction
            if mind is not None:
                self.mind_name = mind
            self.pending = "start"
        return f"starting: \"{self.full_instruction() or 'sort these'}\""

    def add_rule(self, rule: str) -> str:
        with self.lock:
            self.rules.append(rule)
            pol = self.policy
        if pol is not None and hasattr(pol, "board"):  # two streams: the ventral stream reads it and redoes the rule
            pol.board.add_user_rule(rule)
            return "rule passed to the ventral stream; it will redo the sort rule"
        if pol is not None and hasattr(pol, "director"):  # a mind with a planner replans with it before the next move
            pol.instruction = self.full_instruction()
            pol.director.instruction = pol.instruction
            pol.director.again()
            return "rule added; replanning"
        return "rule added; it applies to the next sort"

    def look_at(self, x_mm: float | None = None, y_mm: float | None = None, thing: int | None = None,
                word: str = "") -> str:
        """The person's "look over there": a close look at a place (mm, base frame), at a thing on the board, or at what
        SAM finds for a word on the table picture (near the place, if one is given). It skips the queue; with no job
        running it starts the focus mind holding (looks only, no sort until asked). The findings come back as a
        "looked" event."""
        xy = None if x_mm is None or y_mm is None else (x_mm / 1000.0, y_mm / 1000.0)
        with self.lock:
            pol = self.policy or self.parked
        if thing is not None:
            objs = getattr(pol, "objects", None)
            if objs is None or int(thing) not in objs.all:
                return f"not done: there's no #{thing} measured"
            xy = tuple(objs.all[int(thing)].xy)
        if xy is None and not word:
            return "not done: give a place (x_mm, y_mm), a thing's number, or a word for SAM"
        if xy is not None and not (0.15 <= xy[0] <= 0.65 and -0.45 <= xy[1] <= 0.45):
            return f"not done: ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm is off the table (x 150-650, y -450-450)"
        ask = {"action": "look_for" if word else "look_closer", "note": word, "xy": xy}
        what = (f"'{word}'" if word else "a close look") + (f" at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm" if xy else "")
        if pol is None:
            with self.lock:
                self.looks.append(ask)
                self.mind_name, self.instruction, self.pending = "focus", "", "start"
            return f"looking: {what} (only looking; no sort until asked)"
        if not hasattr(pol, "objects"):
            return f"not done: the '{self.mind_name}' job can't look at one place; stop it or ask for a sort first"
        self._ask(pol, ask)
        return f"looking: {what}, before anything else" + ("" if self.policy is not None else " (when you say carry on)")

    def _ask(self, pol, ask: dict) -> None:
        def done(r):
            self._emit("looked", f"{r.status}: {r.result}")
        pol.board.request(ask["action"], None, ask["note"], xy=ask["xy"], who="person", on_done=done)

    def tell(self, text: str, items: list[dict]) -> str:
        """What the person says is on the table ("4 screws and 2 washers", "the cubes are 15 mm"): SAM searches with
        their words first, a count it can't reach makes it search again before it says done (and then it says what's
        missing), a size explains a grip it won't make. It never stands in for a measurement."""
        items = [{"word": str(i.get("word", "")).strip(), "count": i.get("count"), "size_mm": i.get("size_mm")}
                 for i in items or [] if str(i.get("word", "")).strip()]
        with self.lock:
            self.facts.append((text, items))
            pol = self.policy or self.parked
        if pol is not None and hasattr(pol, "board"):
            pol.board.add_fact(text, items)
        what = ", ".join(i["word"] + (f" x{i['count']}" if i["count"] else "") + (f" ({i['size_mm']} mm)" if i["size_mm"] else "")
                         for i in items) or "noted as text"
        return f"noted: {what}" + ("" if pol is not None else "; it applies when a job starts")

    def pause(self) -> str:
        with self.lock:
            self.pending = "pause"
        return "pausing after this move"

    def resume(self) -> str:
        """Carry on with the job that was paused or stopped (what it has learned kept), or start it again."""
        with self.lock:
            if self.policy is not None:
                return "already working"
            if self.parked is not None:
                self.policy, self.parked, self.state = self.parked, None, "working"
                self._emit("state", "working")
                return "carrying on where I left off"
        return self.start()

    # ------------------------------------------------------------ the world: reset the simulator, or switch to the real cell

    def _stop_now(self) -> None:
        """Drop the job; wait until the robot thread has let go of the cell."""
        with self.lock:
            for pol in (self.policy, self.parked):
                if hasattr(pol, "stop"):  # a mind with a thread of its own (the ventral stream)
                    pol.stop()
            self.policy, self.pending = None, None
        with self.cell_lock:
            pass

    def reset(self) -> str:
        """From scratch. Simulator: the starting table (three empty boxes), the arm at home, no job, no rules. Real
        cell: the robot's head only (no job, no rules, nothing remembered); the table is reset by hand, the trials
        are the operator's, and the connection stays."""
        if self.mode != "sim":
            self._stop_now()
            viz.clear_scene()
            self.step = viz.STEP_OFFSET[0] + 1
            self.parked, self.instruction, self.rules, self.done_log, self.baseline = None, "", [], [], None
            self.facts, self.looks = [], []
            self.state = "connected"
            self._emit("state", "reset")
            return "cleared: no job, no rules, nothing remembered (still connected to the real cell)"
        self._stop_now()
        with self.cell_lock:
            self.cell = IsaacCell(build(load_trial(self.trial_name), self.seed))
            self.cell.noise = True
            viz.clear_scene()
            self.step = viz.STEP_OFFSET[0] + 1  # draw the new session after the clear, not under it
            self.anim = viz.Animator(self.cell)
            viz.truth_setup(self.cell)
        self.instruction, self.rules, self.done_log, self.baseline, self.state, self.parked = "", [], [], None, "idle", None
        self.facts, self.looks = [], []
        self.events.clear()
        self._emit("state", "reset")
        return "the simulator is back to the starting table"

    def use_sim(self) -> str:
        self._stop_now()
        P.TRAVEL, P.APPROACH = SPEEDS["sim"]
        self.mode = "sim"
        return self.reset()

    def use_real(self, url: str, token: str, ca: str | None = None) -> dict:
        """Connect to the real cell and check it (it doesn't start anything: start_trial does). No token: use the one
        from our approved request (`run.py --submit` wrote .request.json)."""
        if not token:
            url, token = _approved_token(url, ca)
        cell = RealCell(url, token, ca)
        report = cell.check()
        self._stop_now()
        with self.cell_lock:
            self.cell = cell
            self.anim = viz.Animator(self.cell)
        P.TRAVEL, P.APPROACH = SPEEDS["real"]
        self.mode, self.state, self.baseline = "real", "connected", None
        self._emit("state", "connected")
        return report

    def start_trial(self) -> dict:
        if self.mode != "real":
            raise RuntimeError("not connected to the real cell")
        with self.cell_lock:
            t = self.cell.start_trial()
        self.state = "trial"
        self._emit("state", "trial")
        return t

    def full_instruction(self) -> str:
        rules = "".join(f"\nRule from the person, overriding anything above: {r}" for r in self.rules)
        return (self.instruction + rules).strip()

    def distract(self, seconds: float) -> None:
        self.distract_s += seconds

    # ------------------------------------------------------------ what it knows (for the chat)

    def status(self) -> str:
        """A plain account of the robot's state and its world model, for the chat agent."""
        lines = [f"State: {self.state}. Instruction: \"{self.instruction or '(none yet)'}\"."]
        if self.rules:
            lines.append("Standing rules from the person: " + "; ".join(self.rules))
        pol = self.policy or self.parked
        if pol is not None and hasattr(pol, "status"):
            lines.append(pol.status())
        if self.done_log:
            lines.append("What I've done lately (oldest first):\n  " + "\n  ".join(self.done_log[-12:]))
        lines.append(f"Robot time so far: {self.cell.t:.0f} s.")
        return "\n".join(lines)

    def recent(self, n: int = 12) -> list[str]:
        return [f"{kind}: {text}" for _, kind, text in list(self.events)[-n:]]

    # ------------------------------------------------------------ the body's loop

    def _emit(self, kind: str, text: str) -> None:
        self.events.append((time.time(), kind, text))
        for f in list(self.listeners):
            try:
                f(kind, text)
            except Exception:
                pass

    def _heard(self, panel: str, text: str, level: str) -> None:
        """Narration / VLM lines that a person standing there would care about become events."""
        if level == "DEBUG" or text.startswith("("):
            return
        if panel == "vlm" and text.startswith("Director ("):
            self._emit("plan", re.sub(r"^Director \([^)]*\): ", "", text))
        elif panel == "vlm" and text.startswith("Instruction vs what I see"):
            self._emit("check", text)
        elif panel == "narration" and re.match(r"Move \d+\.", text):
            what = re.sub(r"^Move \d+\. ", "", text).split(": ")[0]
            self.done_log.append(f"{time.strftime('%H:%M:%S')} {what}")
            self._emit("move", what)
        elif panel == "narration" and text.startswith(("Spread ", "Test ")):
            self.done_log.append(f"{time.strftime('%H:%M:%S')} {text.split(': ')[0][:120]}")
        elif panel == "narration" and (text.startswith(("Done", "Nothing more", "Sorted:", "Lost")) or level == "WARN"):
            self._emit("note", text)

    def _run(self) -> None:
        obs, fresh = None, True
        last_watch = 0.0
        while True:
            with self.lock:
                cmd, self.pending = self.pending, None
            with self.cell_lock:
                try:
                    if cmd == "start":
                        if self.mode == "real" and not self.cell.running():
                            self._emit("note", "No trial is running on the real cell: press Run the test first.")
                        else:
                            self._new_policy()
                            obs, fresh = self.cell.observe(), True
                    elif cmd == "pause" and self.policy is not None:
                        self.parked, self.policy, self.state = self.policy, None, "paused"
                        self._emit("state", "paused")
                    if self.policy is not None:
                        obs, fresh = self._act(obs, fresh)
                    elif self.mode == "sim" and time.time() - last_watch > WATCH_S:
                        last_watch = time.time()
                        try:
                            self._watch()
                        except Exception:  # the tunnel to the GPU box dropped (a network change): reopen it, quietly
                            from lab.isaaccell import tunnel
                            try:
                                tunnel()
                            except BaseException:
                                pass
                except TrialOver:
                    self.policy, self.state = None, "trial over"
                    self._emit("note", "The trial is over (time's up, or the operator ended it).")
                except Exception as e:  # a crash in the policy must not kill the body
                    self._emit("note", f"I hit an error and stopped: {type(e).__name__}: {e}")
                    self.parked, self.policy, self.state = self.policy, None, "stopped"
            if self.policy is None:
                time.sleep(0.2)

    def _new_policy(self) -> None:
        for pol in (self.policy, self.parked):
            if hasattr(pol, "stop"):
                pol.stop()
        self.parked = None
        pol = MINDS[self.mind_name](instruction=self.full_instruction())
        pol.reset()
        with self.lock:
            looks, self.looks = self.looks, []
        if hasattr(pol, "board"):
            for text, items in list(self.facts):
                pol.board.add_fact(text, items)
        if looks and hasattr(pol, "objects"):  # started by look_at: look where asked, sort only when asked
            pol.hold = True
            for ask in looks:
                self._ask(pol, ask)
        if hasattr(pol, "board"):
            pol.board.real = self.mode == "real"
        else:  # say why the ventral panel stays empty
            import rerun as rr
            rr.log("ventral", rr.TextLog(f"No ventral stream in this job: '{self.mind_name}' works from depth alone. "
                                         "Think + sort runs the LLM + SAM beside it.", level="WARN"))
        self.cell.image_size = tuple(getattr(pol, "IMAGE_SIZE", self.cell.image_size))
        self.cell.instruction = self.full_instruction()
        self.policy, self.state = pol, "working"
        self._emit("state", "working")

    def _act(self, obs, fresh):
        pol = self.policy
        viz.set_step(self.step)
        viz.set_sim_time(self.cell.t)
        if fresh:
            self._record(obs)
        self.step += 1
        chunk = pol.act(obs)
        if getattr(pol, "done", False):
            self.policy, self.state = None, "idle"
            self._emit("done", self.status())
            self.baseline = self._table_depth()
            return obs, True
        if self.distract_s > 0:  # someone's talking to it: it slows down
            n = int(min(self.distract_s, 6.0) / P.DT)
            self.distract_s = 0.0
            o = self.cell.observe()
            self.cell.play(P.hold(np.asarray(o["tcp_pose"][:3], float), float(o["tcp_pose"][5]), int(o["joints"][6] >= 0.5), n),
                           on_frame=self.anim)
        blocked = self._precheck(chunk, obs)
        if blocked:  # only a fingertip into the table is held back (real cell); skip it, carry on with the job
            self._emit("note", f"I skipped a move: {blocked}.")
            return self.cell.observe(), True
        self._emit("motion", self._describe(P.tcp_rows(chunk) if isinstance(chunk, P.JointRows) else chunk, obs))
        P.send(self.cell, chunk, on_frame=self.anim)
        fresh = getattr(pol, "wants_obs", True)
        return (self.cell.observe() if fresh else obs), fresh

    def _record(self, obs) -> None:
        """What the robot was given: the raw images and depth to Rerun (the sensor strip), and on the real cell also to
        disk (runs/<session>/obs/NNNN.npz: depth, joints, pose; .jpg images): the real sensors' data, to tune on."""
        viz.images(obs)
        viz.arm("mind", obs["joints"])
        if self.mode != "real":
            return
        import cv2
        d = self.dir / "obs"
        d.mkdir(exist_ok=True)
        np.savez_compressed(d / f"{self.step:04d}.npz", joints=np.asarray(obs["joints"], float),
                            tcp_pose=np.asarray(obs["tcp_pose"], float), t=float(obs["t"]),
                            **{f"depth_{c}": v for c, v in (obs.get("depth") or {}).items()})
        for c, img in (obs.get("images") or {}).items():
            cv2.imwrite(str(d / f"{self.step:04d}_{c}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    def _describe(self, chunk, obs) -> str:
        """One line for the actions panel: where the fingertip goes, how it's turned, the gripper, how long."""
        from calibration.kinematics import wpr_matrix
        rows = np.atleast_2d(np.asarray(chunk, float))
        if rows.shape[1] != 7:
            return f"joint motion, {len(rows)} rows ({len(rows) * P.DT:.1f} s)"
        up = lambda r: f"({r[0]:.0f}, {r[1]:.0f}, {r[2] - P.TABLE_UF0:.0f} up)"
        a, b = np.asarray(obs["tcp_pose"], float) if obs is not None else rows[0], rows[-1]
        tilt = np.degrees(np.arccos(np.clip(-wpr_matrix(*b[3:6])[2, 2], -1, 1)))
        grips = np.diff(rows[:, 6] >= 0.5).nonzero()[0]
        g = "" if not len(grips) else " · " + ", ".join("close" if rows[k + 1, 6] >= 0.5 else "open" for k in grips)
        still = int(np.sum(np.all(np.abs(np.diff(rows[:, :3], axis=0)) < 1e-6, axis=1)))
        return (f"{up(a)} → {up(b)} mm · {'straight down' if tilt < 3 else f'tilted {tilt:.0f}°'}{g} · "
                f"{len(rows) * P.DT:.1f} s" + (f" (holds {still * P.DT:.1f} s to look)" if still >= 5 else ""))

    def _precheck(self, chunk, obs) -> str | None:
        """What should hold a tcp chunk back: on the real cell, a fingertip below the table. The arm model's fuller
        check (IK continuity, the arm against the table) is only logged: on the real cell it blocked moves the arm
        could make (its IK is stricter than the controller's, and its fingers fatter than the real ones), and the
        controller enforces its own limits."""
        joints = isinstance(chunk, P.JointRows)
        rows = np.atleast_2d(np.asarray(P.tcp_rows(chunk) if joints else chunk, float))
        if rows.shape[1] != 7 or obs is None:
            return None
        lowest = float((rows[:, 2] - P.TABLE_UF0).min())  # the fingertip's height over the table, mm
        if self.mode == "real" and lowest < 0.5:
            return f"it would put the fingertips {lowest:.1f} mm into the table"
        try:
            rep = (armcheck.check_joints(np.asarray(chunk)[:, :6], 0.0) if joints
                   else armcheck.check(rows, np.asarray(obs["joints"][:6], float), []))
            if not rep.ok:
                viz.narrate(f"(arm check, not blocking: {rep.problem})", level="WARN")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------ watching the table when idle

    def _table_depth(self):
        return self.cell.observe()["depth"]["table"].astype(np.int32)

    def _watch(self) -> None:
        d = self._table_depth()
        if self.baseline is None:
            self.baseline = d
            return
        both = (d > 0) & (self.baseline > 0)
        changed = both & (np.abs(d - self.baseline) > 5)
        if changed.sum() > CHANGE_PX:
            nearer = (changed & (d < self.baseline)).sum() > (changed & (d > self.baseline)).sum()
            self._emit("seen", "Something changed on the table while I was idle: "
                       + ("there's more on it than before." if nearer else "something's gone from it."))
            self.baseline = d

    # ------------------------------------------------------------ Joel's hands (the page's panel)

    def objects(self) -> list[dict]:
        out = []
        poses = getattr(self.cell, "poses", None)  # the Isaac cell's truth; the MuJoCo stand-in has none
        if poses is None:
            return out
        for i in self.cell.instances:
            p = poses.get(i.id)
            if p is None:
                continue
            out.append({"id": i.id, "name": i.name, "kind": i.kind, "color": i.color, "box": i.kind.startswith("box"),
                        "x": round(float(p[0]) * 1000), "y": round(float(p[1]) * 1000)})
        return out

    def pour(self, order: list[tuple[str, str]]) -> str:
        new = self.cell.pour(order)
        viz.truth_setup(self.cell)
        return f"poured {len(new)} things onto the table"

    def take(self, ids) -> str:
        self.cell.take(ids)
        return f"took {len(ids)} thing(s) off the table"

    def tip_box(self, box_id: int) -> str:
        out = self.cell.tip_box(box_id)
        return f"tipped box {box_id} out: {len(out)} things back on the table"

    def add_box(self, color: str) -> str:
        b = self.cell.add_box(color, at=(0.30, 0.22))
        viz.truth_setup(self.cell)
        return f"put a {color} box down at (300, 220) mm"


def _approved_token(url: str, ca: str | None) -> tuple[str, str]:
    """(url, token) of our submitted request, once the organizers have approved it."""
    import json
    from cell_client import CellClient
    req_file = Path(__file__).resolve().parents[1] / ".request.json"
    if not req_file.exists():
        raise RuntimeError("no token given, and no submitted request (.request.json) to take one from")
    req = json.loads(req_file.read_text())
    st = CellClient(req.get("url") or url, "", ca).request_status(req["request_id"], req["secret"])
    if st.get("status") != "approved":
        raise RuntimeError(f"request {req['request_id']} is {st.get('status')}"
                           + (f": {st['reason']}" if st.get("reason") else "") + "; paste a run token, or wait for approval")
    return req.get("url") or url, st["token"]
