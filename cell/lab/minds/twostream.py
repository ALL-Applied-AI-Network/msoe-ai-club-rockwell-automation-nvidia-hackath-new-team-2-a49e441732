"""Two streams at once: this mind is the dorsal stream (depth, the map, the arm); lab.ventral runs beside it in its own
thread (an LLM with SAM 3.1). They talk through a board (lab.streams).

    uv run python -m lab.trial run mixed_pile --mind streams

The dorsal stream's loop:
    1. map the table (lab.minds.mapper); after every few looks, put the things and a clear table picture on the board.
       Between looks, a request from the ventral stream goes first (it skips the queue).
    2. then, until the job is done: a request if there is one; else the next thing the sort rule gives a destination
       (only things both streams agree on: one 3D segment, one outline); else wait for the ventral stream.

Requests are semantic ("break up #4: I see 4 things in it, keep them close"); the dorsal stream turns them into motion:
    break_up    two or three things -> pick the top one and set it down just clear of the rest;
                a pile -> a rake through it toward free table; then look again so the pieces become things
    look_closer -> the wrist camera straight above it, two pictures
    put_away    -> carry it to a box or a table spot
It refuses what it can't do safely (no clear grip, no free room, out of reach) and says why.
"""
from __future__ import annotations

import time

import numpy as np

from calibration.geometry import SURFACE_Z
from calibration.kinematics import tcp_uf0, tool_ik, tool_pose, tool_r, wpr_of
from lab import agree, armcheck, grasp, mindview, primitives as P, viz
from lab.minds.sorter import DROP_ABOVE_RIM_MM, GRIP_SHARE, SorterMind
from lab.streams import Board
from lab.things import _inside

PUBLISH_EVERY = 4      # looks between updates of the board while mapping
IDLE_ROWS = 20         # 2 s of standing still while the ventral stream thinks
IDLE_MAX_S = 240.0     # nothing to do and no word from the ventral stream for this long (wall clock): stop
STALE_M = 0.05         # after a pick or a rake, names and rule places this close to where the arm worked are dropped
CLEAR_MM = 14.0        # a free spot: no evidence of anything within this of its centre
GROUP_GAP_MM = 22.0    # things in one table group, centre to centre
RAKE_ABOVE_MM = 3.0
TRIES = 2              # failed picks at one spot before it's left to the ventral stream (ids change; spots don't)


class TwoStreamMind(SorterMind):
    JOB = "two streams"
    IMAGE_SIZE = (1920, 1080)
    MAX_STEPS = 2500  # SAM needs the pixels: a 10 mm part is ~8 px across in the whole table picture

    def __init__(self, instruction: str = "", budget: int = 24, ventral: bool = True, **kw):
        super().__init__(instruction, budget=budget, **kw)
        self.with_ventral = ventral

    def reset(self) -> None:
        super().reset()
        self.board = Board()
        self.ventral = None
        self.placed: list[tuple[np.ndarray, str]] = []  # (where, destination) of everything put away
        self.table_slots: dict[str, list] = {}          # table group name -> spots used
        self.idle_since = None
        self.failed: set[int] = set()
        self.misses: list[np.ndarray] = []               # where picks came up empty (one entry per miss)
        self.boxes = {}
        self._started = False

    def stop(self) -> None:
        self.board.stop()

    # ------------------------------------------------------------ the board

    def _publish(self, obs, kind: str, text: str = "") -> None:
        arm = agree.arm_mask(np.asarray(obs["joints"][:6], float), self.cams["table"],
                             (obs["images"]["table"].shape[1], obs["images"]["table"].shape[0]))
        self.board.set_things(self.things.known, self.things.maybe, obs["images"]["table"], arm)
        self._draw_board()
        self.board.post(kind, text)

    def _draw_board(self) -> None:
        import rerun as rr
        b = self.board
        rule = b.rule
        lines = ["### The board", f"**Sort rule** ({rule.source if rule else ''}): {rule.text if rule else '*none yet*'}"]
        if rule:
            lines += [f"- {g['name']}: {', '.join('#' + str(i) for i in g['ids'])} → {g['to']}" for g in rule.groups]
        if b.user_rules:
            lines.append("**You said**: " + "; ".join(b.user_rules))
        lines.append(f"**Requests**: {b.queue_text()}")
        if b.agreement is not None:
            lines.append("**Streams**: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in b.agreement.case_counts().items()))
        lines.append(f"**Put away**: {len(self.placed)}")
        rr.log("board", rr.TextDocument("\n\n".join(lines), media_type=rr.MediaType.MARKDOWN))

    # ------------------------------------------------------------ the loop

    def _loop(self, obs):
        self.table_rgb = obs["images"]["table"]
        self.q_home = np.asarray(obs["joints"][:6], float)
        if self.with_ventral:
            from lab.ventral import Ventral
            self.ventral = Ventral(self.board, self.cams, self.instruction)
            self.ventral.start()
        self._published_at = 0
        yield from self._map_with_requests(obs)
        obs = self.obs
        self.sorting = True
        self._refresh()
        self.boxes = {t.id: t for t in self.things.known if t.kind == "box"}
        self._publish(obs, "map_done", f"{len(self.things.known)} things, {len(self.things.maybe)} maybe, "
                                       f"{len(self.boxes)} boxes after {self.looks} looks")
        while True:
            r = self.board.take_request()
            if r is not None:
                obs = yield from self._do_request(r, obs)
                continue
            job = self._next_job()
            if job is not None:
                t, dest = job
                obs = yield from self._put(t, dest, obs)
                continue
            if self._finished():
                self._say("Done: every thing the rule names is where it goes.")
                self.board.post("done", "the rule is carried out")
                break
            if self.idle_since is None:
                self.idle_since = time.time()
                waiting = [t for t in self.things.known if t.kind == "part" and not self._is_placed(t) and not self._in_box(t)
                           and not self._given_up(t)]
                self.board.post("idle", f"{len(waiting)} part(s) on the table I can't move yet: "
                                + ", ".join(f"#{t.id}" + (" (no destination in the rule)" if self._dest_of(t.id) is None else " (streams don't agree on it yet: segment again)")
                                            for t in waiting))
            elif time.time() - max(self.idle_since, self._last_ventral()) > IDLE_MAX_S:
                self._say("Nothing to do and no word from the ventral stream for a while: stopping.")
                self.board.post("done", "idle too long")
                break
            obs = yield self._still(obs, IDLE_ROWS)
        self.board.stop()
        self.obs = obs

    def _map_with_requests(self, obs):
        yield from super(SorterMind, self)._loop(obs)

    def _show(self, obs, line: str, t0: float) -> None:
        super()._show(obs, line, t0)
        if not self._started:  # the table camera's depth is in: the ventral stream has things and a picture to start on
            self._started = True
            self._publish(obs, "started", "mapping the table; first things from the table camera are on the board")

    def _between_looks(self, obs):
        if self.looks - self._published_at >= PUBLISH_EVERY:
            self._published_at = self.looks
            self._publish(obs, "map_progress", f"{self.looks} looks; {len(self.things.known)} things so far")
        r = self.board.take_request()
        while r is not None:  # the ventral stream's requests go first, even mid-map
            self.sorting = True
            self._refresh()
            self.boxes = {t.id: t for t in self.things.known if t.kind == "box"}
            obs = yield from self._do_request(r, obs)
            self.sorting = False
            r = self.board.take_request()
        return obs

    def _refresh(self):
        found = self.things.update(self.map, self.table_rgb, self.cams)
        self._draw_things(found)
        return found

    # ------------------------------------------------------------ what to do next, from the rule

    def _last_ventral(self) -> float:
        """When the ventral stream last did something (a request, a rule)."""
        with self.board.cond:
            ts = [t for _, t, k, _ in self.board.events if k in ("request", "rule")]
        return max(ts) if ts else 0.0

    def _dest_of(self, tid: int) -> str | None:
        rule = self.board.rule
        if rule is None:
            return None
        return next((g["to"] for g in rule.groups if tid in g["ids"]), None)

    def _group_of(self, tid: int) -> str:
        rule = self.board.rule
        return next((g["name"] for g in rule.groups if tid in g["ids"]), "?") if rule else "?"

    def _next_job(self):
        ag = self.board.agreement
        cases = {r.seg: r.case for r in ag.rows} if ag else {}
        for t in sorted(self.things.known, key=lambda t: t.xy[1]):
            if t.kind != "part" or t.id in self.failed or self._is_placed(t) or self._in_box(t) or self._given_up(t):
                continue
            dest = self._dest_of(t.id)
            if dest is None or (cases.get(t.id) != "agree" and t.id not in self.board.vouched):
                continue  # no destination yet, or the streams don't (yet) agree it's one thing: wait
            return t, dest
        return None

    def _finished(self) -> bool:
        rule = self.board.rule
        if rule is None:
            return False
        if any(r.status in ("queued", "doing") for r in self.board.requests):
            return False
        left = [t for t in self.things.known if t.kind == "part" and not self._is_placed(t) and not self._in_box(t)
                and t.id not in self.failed and not self._given_up(t)]
        return not left and len(self.placed) > 0

    def _given_up(self, t) -> bool:
        return sum(np.linalg.norm(t.xy - p) < 0.010 for p in self.misses) >= TRIES

    def _is_placed(self, t) -> bool:
        return any(np.linalg.norm(t.xy - p) < 0.012 for p, _ in self.placed)

    def _in_box(self, t) -> bool:
        return any(_inside(t.xy, b, margin_mm=3) for b in self.boxes.values())

    # ------------------------------------------------------------ requests: words -> motion

    def _do_request(self, r, obs):
        self.idle_since = None
        if r.xy is not None:  # a place, not a thing (an outline with no 3D segment under it)
            self._say(f"Request {r.id} from the ventral stream: look closer at ({r.xy[0] * 1000:.0f}, {r.xy[1] * 1000:.0f}) mm. It goes first.")
            obs = yield from self._look_twice(r.xy, obs)
            self._refresh()
            self.board.finish(r, "done", "looked twice from 20 cm straight above")
            self._publish(obs, "skill_done", f"after request {r.id}: {len(self.things.known)} things")
            return obs
        t = self.board.things.get(r.target) if isinstance(r.target, int) else None
        t = next((x for x in self.things.known + self.things.maybe if t is not None and np.linalg.norm(x.xy - t.xy) < 0.012), t)
        if t is None:
            self.board.finish(r, "refused", f"#{r.target} isn't on the table any more (things moved: re-segment)")
            return obs
        self._say(f"Request {r.id} from the ventral stream: {r.action.replace('_', ' ')} #{t.id}"
                  + (f" ({r.note})" if r.note else "") + ". It goes first.")
        if r.action == "look_closer":
            obs = yield from self._look_twice(t.xy, obs)
            self._refresh()
            self.board.finish(r, "done", "looked twice from 20 cm straight above")
        elif r.action == "break_up":
            obs = yield from self._break_up(r, t, obs)
        elif r.action == "put_away":
            obs = yield from self._put(t, r.to or self._dest_of(t.id) or "table", obs, request=r)
        self._publish(obs, "skill_done", f"after request {r.id}: {len(self.things.known)} things")
        return obs

    def _break_up(self, r, t, obs):
        before = self._near(t.xy, t.size_mm[0] / 2000 + 0.01)
        n = r.expect or 2
        keep_close = "far" not in (r.note or "")
        if n >= 3 and t.size_mm[0] > 30:
            plan = self._rake_rows(t, obs, keep_close)
            how = "rake"
        else:
            plan = self._apart_rows(t, obs, keep_close)
            how = "pick apart"
        if plan is None:
            # the other way, before giving up
            plan = self._apart_rows(t, obs, keep_close) if how == "rake" else self._rake_rows(t, obs, keep_close)
            how = "pick apart" if how == "rake" else "rake"
        if plan is None:
            self.board.finish(r, "refused", "no safe pick or rake for it from what I've seen (no grip, no free room, or out of reach)")
            return obs
        rows_list, where, what = plan
        self._say(f"Breaking up #{t.id} ({n} things, says the ventral stream): {what}.")
        for rows in rows_list:
            obs = yield rows
        for xy in where:  # what we knew there is stale
            self.map.forget(xy, 0.03 if how == "rake" else 0.015)
        obs = yield from self._look_twice(t.xy, obs)
        for xy in where[1:]:
            if np.linalg.norm(np.asarray(xy) - t.xy) > 0.02:
                obs = yield from self._look_twice(xy, obs)
        self._refresh()
        for xy in where:
            self.board.forget_near(xy, STALE_M)
        after = self._near(t.xy, t.size_mm[0] / 2000 + 0.06)
        self.board.finish(r, "done", f"{how}: {before} segment(s) there before, {after} in and around it now")
        return obs

    def _near(self, xy, radius: float) -> int:
        return sum(np.linalg.norm(x.xy - xy) < radius for x in self.things.known if x.kind == "part")

    def _look_twice(self, xy, obs):
        look = self._look_rows(tuple(xy), obs)
        if look is None:
            return obs
        obs = yield look
        self._fuse_wrist(obs)
        obs = yield self._still(obs, 3)
        self._fuse_wrist(obs)
        return obs

    # ------------------------------------------------------------ free table

    def _occupied(self) -> np.ndarray:
        m = self.map
        return ((m.solid() | m.leaning()) & m.clear_of_table()).any(2)

    def _free(self, xy, occ=None, clear_mm: float = CLEAR_MM) -> bool:
        """Seen table, nothing on it within clear_mm, reachable, not in or by a box."""
        m = self.map
        occ = self._occupied() if occ is None else occ
        if not P.reachable(*xy, margin_mm=25) or any(_inside(xy, b, margin_mm=25) for b in self.boxes.values()):
            return False
        c = ((np.asarray(xy) - m.origin[:2]) / m.v).astype(int)
        r = int(np.ceil(clear_mm / 1000 / m.v))
        if np.any(c - r < 0) or np.any(c + r >= occ.shape):
            return False
        if occ[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1].any():
            return False
        seen = m.table_found(sure=False)[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1]
        return seen.mean() > 0.6 and not any(np.linalg.norm(p - xy) < clear_mm / 1000 for p, _ in self.placed)

    def _spot_near(self, xy, r_min: float, r_max: float, occ=None):
        occ = self._occupied() if occ is None else occ
        for r in np.arange(r_min, r_max + 1e-6, 0.008):
            for a in np.radians(np.arange(0, 360, 20)):
                p = np.asarray(xy) + r * np.array([np.cos(a), np.sin(a)])
                if self._free(p, occ):
                    return p
        return None

    def _table_spot(self, group: str):
        """The next spot for a thing of this table group: a row of spots per group, groups side by side."""
        occ = self._occupied()
        used = self.table_slots.get(group)
        if used:
            last = used[-1]
            d = np.array([0.0, GROUP_GAP_MM / 1000]) if len(used) < 2 else used[-1] - used[-2]
            for step in (d, -d, d[::-1], -d[::-1]):
                p = last + step
                if self._free(p, occ):
                    return p
            return self._spot_near(last, 0.02, 0.08, occ)
        anchors = [s[0] for s in self.table_slots.values() if s]
        if anchors:  # a new group: next to the others (a cluster of clusters)
            return self._spot_near(anchors[-1], 0.045, 0.12, occ)
        # the first group: free table toward the robot's left, clear of the work
        for x in np.arange(0.30, 0.56, 0.02):
            for y in np.arange(0.30, 0.08, -0.02):
                if self._free((x, y), occ, clear_mm=30):
                    return np.array([x, y])
        return None

    # ------------------------------------------------------------ moves

    def _approach(self, xy, yaw, safe_mm, obs):
        """A joint-space move to straight above xy, fingers down, jaws at yaw; (rows, q there) or None."""
        q_now = np.asarray(obs["joints"][:6], float)
        T_now = tool_pose(q_now)
        over = np.array([xy[0], xy[1], SURFACE_Z + safe_mm / 1000])
        wpr = np.array([-180.0, 0.0, tool_r(yaw)])
        try:
            q_over = tool_ik(over, wpr, q_now)
        except ValueError:
            return None
        rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q_over, over, wpr, height_mm=safe_mm, settle=0)
        if rows is None or not armcheck.check_joints(np.asarray(rows)[:, :6], 0.0).ok:
            return None
        return rows, q_over

    def _carry_rows(self, t, place_xy, drop_mm, safe_top_mm, obs, grip=None):
        """Grip t, carry it to place_xy (drop_mm: let go that high over the table; None: set it down)."""
        scene, target = self._surface(t)
        grips = [grip] if grip is not None else grasp.find(scene, target, max_height_mm=GRIP_SHARE * t.top_mm)[0]
        for g in grips:
            safe = max(P.SAFE_MM, safe_top_mm + 15.0 + g.height_mm)
            a = self._approach((g.x, g.y), g.yaw, safe, obs)
            if a is None:
                continue
            approach, q_over = a
            rows, _ = P.pick_and_place(tcp_uf0(q_over), (g.x, g.y), g.height_mm, g.yaw, tuple(place_xy),
                                       drop_mm=drop_mm, safe_mm=safe)
            rep = armcheck.check(rows, q_over)
            if rep.ok:
                self._draw_grip(g)
                return approach, rows, g
        return None

    def _apart_rows(self, t, obs, keep_close: bool):
        """Pick the top-most graspable piece of t and set it down just clear of the rest."""
        scene, target = self._surface(t)
        grips, _ = grasp.find(scene, target, max_height_mm=t.top_mm)
        if not grips:
            return None
        g = max(grips[:4], key=lambda g: g.height_mm)  # the top piece first: it's the one sitting on others
        r0 = t.size_mm[0] / 2000 + (0.02 if keep_close else 0.05)
        spot = self._spot_near(t.xy, r0, r0 + 0.06)
        if spot is None:
            return None
        plan = self._carry_rows(t, spot, None, t.top_mm, obs, grip=g)
        if plan is None:
            return None
        approach, rows, g = plan
        return [approach, rows], [(g.x, g.y), tuple(spot)], (f"pick the top piece (grip {g.height_mm:.0f} mm up) and set it down "
                                                             f"{np.linalg.norm(spot - t.xy) * 1000:.0f} mm away")

    def _rake_rows(self, t, obs, keep_close: bool):
        """A rake through the middle of t, toward free table: what's in the blade's path moves on, the rest stays."""
        occ = self._occupied()
        half = t.size_mm[0] / 2000
        run = 0.03 if keep_close else 0.06
        best = None
        for a in np.radians(np.arange(0, 360, 30)):
            d = np.array([np.cos(a), np.sin(a)])
            start, end = t.xy - d * (half + 0.018), t.xy + d * (half + run)
            if not (P.reachable(*start, margin_mm=20) and P.reachable(*end, margin_mm=20)):
                continue
            if any(_inside(p, b, margin_mm=20) for b in self.boxes.values() for p in (start, end)):
                continue
            if not self._free(start, occ, clear_mm=8):  # the blade comes down on bare table
                continue
            room = sum(self._free(t.xy + d * (half + s), occ, clear_mm=8) for s in np.arange(0.01, run + 0.03, 0.01))
            if best is None or room > best[0]:
                best = (room, start, end, a)
        if best is None:
            return None
        _, start, end, a = best
        appr = self._approach(start, np.degrees(a), P.SAFE_MM, obs)
        if appr is None:
            return None
        approach, q_over = appr
        rows, what = P.rake(tcp_uf0(q_over), start, end, height_mm=RAKE_ABOVE_MM)
        rep = armcheck.check(rows, q_over)
        if not rep.ok:
            viz.narrate(f"(rake skipped: {rep.problem})", level="DEBUG")
            return None
        mid = (t.xy + end) / 2
        return [approach, rows], [tuple(t.xy), tuple(mid), tuple(end)], what

    def _put(self, t, dest: str, obs, request=None):
        """Carry t to dest ("box #N" or "table"), then check the spot it left."""
        self.idle_since = None
        box = None
        if dest.strip().lower().startswith("box"):
            try:
                box = self.boxes.get(int(dest.split("#")[-1]))
            except ValueError:
                box = None
            if box is None:
                self._say(f"#{t.id}: the rule says {dest}, and there's no such box; asking the ventral stream.")
                self.board.forget_near(t.xy, 0.001)  # out of the rule: the ventral stream gives it a real destination
                self.board.post("gave_up", f"#{t.id}: the rule says {dest}, which isn't a box on the board")
                if request:
                    self.board.finish(request, "refused", f"no {dest}")
                return obs
            place, drop, top, where = tuple(box.xy), box.top_mm + DROP_ABOVE_RIM_MM, box.top_mm, f"the {box.color} box #{box.id}"
        else:
            group = self._group_of(t.id)
            spot = self._table_spot(group)
            if spot is None:
                self._say(f"#{t.id}: no free table for the '{group}' group; leaving it.")
                self.failed.add(t.id)
                return obs
            place, drop, top, where = tuple(spot), None, t.top_mm, f"the table, with the '{group}' group"
        plan = self._carry_rows(t, place, drop, top, obs)
        if plan is None:
            self._say(f"#{t.id}: no clear grip from what I've seen; leaving it for now.")
            self.failed.add(t.id)
            if request:
                self.board.finish(request, "refused", "no clear grip")
            return obs
        approach, rows, g = plan
        label = getattr(t, "label", None) or t.name
        self._say(f"Put #{t.id} ({label}) in {where}.")
        obs = yield approach
        obs = yield rows
        self.map.forget((g.x, g.y), 0.012)
        if box is None:
            self.map.forget(place, 0.012)
            self.table_slots.setdefault(self._group_of(t.id), []).append(np.asarray(place))
        self.placed.append((np.asarray(place), dest))
        obs = yield from self._look_twice((g.x, g.y), obs)
        if box is None:
            obs = yield from self._look_twice(place, obs)
        self._refresh()
        self.board.forget_near((g.x, g.y), STALE_M)
        # still there: a part where it was, standing as tall as it did (a touching neighbour is lower or elsewhere)
        still = [x for x in self.things.known if x.kind == "part" and np.linalg.norm(x.xy - t.xy) < 0.006
                 and x.top_mm >= 0.75 * t.top_mm]
        if still:
            self.placed.pop()
            self.misses.append(np.asarray(t.xy))
            self.picks["failed"] += 1
            if self._given_up(t):
                self._say(f"#{t.id}: still there after {TRIES} picks; leaving it to the ventral stream (break it up, or name it again).")
                self.board.post("gave_up", f"#{still[0].id} at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) mm: {TRIES} picks didn't lift it")
            else:
                self._say(f"#{t.id}: still there after the pick.")
        else:
            self.picks["verified"] += 1
        self._publish(obs, "skill_done", f"put #{t.id} ({label}) in {where}" + (" (it didn't come up)" if still else ""))
        if request:
            self.board.finish(request, "done", f"put in {where}")
        return obs
