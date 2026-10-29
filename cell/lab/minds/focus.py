"""Look where it matters: the ventral stream (LLM + SAM) picks the regions; the arm looks closely only there, and picks
from a fresh close look. No blind mapping of the whole table.

    uv run python -m lab.trial run sort_cups --mind focus

    1. glance: the table camera's picture; the ventral stream chooses SAM prompts ("cube", "cup"), and each outline
       lands on the table (calibration): the regions worth a look. The table camera's depth goes into the map once,
       only to know what stands where (clearance for the fingers), not to find parts.
    2. close looks at the regions, as low as the fingers allow (the camera rides 118 mm above the fingertips, 106 mm to
       the side): SAM outlines objects in the wrist picture and depth inside each outline is pooled against the table
       ring around it (lab.focus): position, top face, height, colour per object.
    3. the sort rule (the ventral stream's) gives each object a cup; for each: a close look right above it, re-measure,
       grip at the fresh position, drop in the cup, look to check.
Requests from the ventral stream (look closer at an outline or an object) skip the queue, as before.
"""
from __future__ import annotations

import time

import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z
from calibration.kinematics import tcp_uf0, tool_ik, tool_pose, wpr_of
from lab import agree, armcheck, focus, grasp, primitives as P, viz
from lab.minds.twostream import IDLE_ROWS, TwoStreamMind
from lab.views import _cam_in_tool, _look_at

CAM_HEIGHTS = (0.14, 0.17, 0.21, 0.24)  # camera above the table for a close look, lowest first (fingertips 118 mm lower)
TIP_CLEAR_MM = 12.0               # the fingertips stay this far above anything the map knows under them
PROMPT_WAIT_S = 60.0              # wall time to wait for the ventral stream's prompts before using the defaults
DEFAULT_PROMPTS = ["cube", "cup", "small object"]
LOOK_R = 0.035                    # a close look covers this well around its centre (objects there not found = gone)
MAX_RELOOKS = 6                   # second looks in a whole job: after that it acts on what it has
RELOOKS = 2                       # second looks at one spot where the streams disagree, then it's dropped
MAX_PROMPTS = 6                   # SAM words per look (each is a call)
DROP_ABOVE_RIM_MM = 4.0
GRIP_SHARE = 0.45                 # fingertips close at most this far up the part
COLOUR_SURE = ("yellow", "green", "blue", "purple")  # hues the cameras keep apart (not red and orange)
TABLE_EVERY = 8                   # ask for the table camera at least every this many observations (else wrist only)
PILE_GAP = 0.07                   # m between parts of different piles on the table (the piles never touch)
REAL_PART_MM = 15.0               # the real cubes (measured through the Oct 28 calibration)
UNSEEN_CAP_MM = 90.0              # space no depth has seen counts as solid up to this high, for the fingertips
CUP_TOP_MIN_MM = 70.0             # a cup's rim for clearance: never lower than this (Oct 28: the table camera read a real
                                  # cup as 41 mm, the wrist 45-52, and the hand caught the rim on the way in)


def cup_top(b) -> float:
    """A cup's height for planning clearance: its measured top, or CUP_TOP_MIN_MM if that's higher (depth sees the thin
    rim poorly and reads cups short)."""
    return max(float(b.top_mm), CUP_TOP_MIN_MM)


BOX_WORDS = ("cup", "box", "container", "bin", "tray", "bowl", "basket", "bucket")
BOX_ASSUME_MM = 90.0              # an unmeasured cup or box near a close look: the fingertips stay above this
SIZE_TOL_MM = 4.0                 # measured this much wider than the person said: say so when the grip fails


class FocusMind(TwoStreamMind):
    JOB = "focus"

    def act(self, obs):
        """Fill in the table camera's picture and depth when this observation left them out (cams_next): the last ones."""
        if "table" in (obs.get("images") or {}):
            self._table_obs = (obs["images"]["table"], (obs.get("depth") or {}).get("table"))
            self._need_table, self._since_table = False, 0
        elif getattr(self, "_table_obs", None) is not None:
            obs["images"] = {**obs["images"], "table": self._table_obs[0]}
            if self._table_obs[1] is not None:
                obs["depth"] = {**(obs.get("depth") or {}), "table": self._table_obs[1]}
            self._since_table = getattr(self, "_since_table", 0) + 1
        return super().act(obs)

    def cams_next(self) -> tuple:
        """The pictures to ask for next: the wrist alone, except when the table camera is needed (a glance from home,
        the final check) or hasn't been asked for in TABLE_EVERY observations (the ventral stream's view of the table)."""
        if getattr(self, "_need_table", True) or getattr(self, "_since_table", 0) >= TABLE_EVERY:
            return ("table", "side", "wrist")
        return ("wrist",)

    def reset(self) -> None:
        super().reset()
        self.objects = focus.Objects()
        self.board.mode = "focus"
        self.prompts: list[str] | None = None
        self.T_ct = _cam_in_tool(self.cams["wrist_depth"])
        self.final_checks = 0
        self.depth_only = focus.DepthOnly()   # raised depth no outline covers, until a second look finds it again
        self.relooks: dict = {}               # spot (1 cm grid) -> second looks spent there
        self.to_look: list = []               # (xy, why, extra SAM words): disagreements to look at again
        self.hold = False                     # only the person's looks until a sort is asked for (lab.robot.look_at)
        self._look_ok = True                  # the last close look could be made
        self._count_search = False            # one more search with the person's words when the count is short

    # ------------------------------------------------------------ the board, with measured objects

    def _publish(self, obs, kind: str, text: str = "") -> None:
        img = obs["images"]["table"]
        arm = agree.arm_mask(np.asarray(obs["joints"][:6], float), self.cams["table"], (img.shape[1], img.shape[0]))
        self.board.set_things([o for o in self.objects.all.values() if o.kind != "unknown"], [], img, arm)
        with self.board.cond:
            self.board.unknown = [o.describe() for o in self.objects.all.values() if o.kind == "unknown"]
        self._draw_objects()
        self._draw_board()
        self.board.post(kind, text)

    def _draw_objects(self) -> None:
        rr.log("mind/7_things", rr.Clear(recursive=True))
        for o in self.objects.all.values():
            h = max(o.top_mm, 2.0) / 1000
            rr.log(f"mind/7_things/o{o.id}", rr.Boxes3D(
                centers=[[*o.xy, SURFACE_Z + h / 2]], half_sizes=[[o.size_mm[0] / 2000, o.size_mm[1] / 2000, h / 2]],
                rotation_axis_angles=[rr.RotationAxisAngle([0, 0, 1], degrees=o.angle)], colors=[list(o.rgb)],
                labels=[f"#{o.id} {o.label or o.what} · {o.size_mm[0]:.0f}×{o.size_mm[1]:.0f}×{o.top_mm:.0f} mm"]))
            if o.outline is not None:  # the outline the grip is planned on, at the height of its top
                ring = np.column_stack([o.outline, np.full(len(o.outline), SURFACE_Z + h)])
                rr.log(f"mind/7_things/o{o.id}/outline", rr.LineStrips3D([np.vstack([ring, ring[:1]])],
                                                                         colors=[[255, 255, 255]], radii=0.0003))

    # ------------------------------------------------------------ the loop

    def _loop(self, obs):
        self.table_rgb = obs["images"]["table"]
        self.q_home = np.asarray(obs["joints"][:6], float)
        if self.with_ventral:
            from lab.ventral import Ventral
            self.ventral = Ventral(self.board, self.cams, self.instruction)
            self.ventral.start()
        self._started = True  # (the two-stream mind would announce itself on the first fused look)
        self.map.integrate(obs["depth"]["table"], self.cams["table_depth"])  # what stands where: clearance only
        cups = focus.cups_from_table(obs["depth"]["table"], obs["images"]["table"], self.cams)  # every cup, at once
        self.objects.merge(cups)
        if cups:
            self._say(f"The table camera's depth shows {len(cups)} cup(s): " + "; ".join(o.describe() for o in self.objects.boxes()) + ".")
        self._draw()
        self._publish(obs, "started", "the table camera's picture is on the board: choose SAM prompts (segment)")
        while self.hold and not self.board.stopped:  # asked to look, not to sort: only the person's looks
            r = self.board.take_request(who="person")
            obs = (yield from self._do_request(r, obs)) if r is not None else (yield self._still(obs, IDLE_ROWS))
        # 1. the ventral stream's prompts and regions (it segments the table picture)
        t0 = time.time()
        while self.board.regions is None and time.time() - t0 < PROMPT_WAIT_S and not self.board.stopped:
            obs = yield self._still(obs, 5)
        regs = self.board.regions
        if not regs:
            self._say("No regions from the ventral stream: looking where SAM finds anything with default prompts.")
            regs = self._glance(obs, DEFAULT_PROMPTS)
        self.prompts = self.board.prompts or DEFAULT_PROMPTS
        cups = self.objects.boxes()  # measured already: no close look needed at a cup
        regs = [r for r in regs if not any(np.linalg.norm(np.asarray(r["xy"]) - b.xy) * 1000 < max(b.size_mm) / 2 + 8 for b in cups)]
        spots = focus.spots(regs)
        self._say(f"{len(regs)} outlines on the table picture outside the cups -> {len(spots)} close looks.")
        # 2. close looks only there
        for xy in spots:
            obs = yield from self._requests_first(obs)
            obs = yield from self._close_look(xy, obs)
        if self._next_job() is None:  # second looks only while there's nothing to pick yet
            obs = yield from self._look_again(obs)
        self.boxes = {o.id: o for o in self.objects.boxes()}
        self._publish(obs, "map_done", f"measured {len(self.objects.parts())} parts and {len(self.boxes)} cups "
                                       f"in {len(spots)} close looks")
        # 3. the rule's work, requests first
        while True:
            job = self._next_job()  # a job first: act on the best measurement, and a miss is the check
            r = self.board.take_request(looks=job is None)  # the ventral stream's "look closer" waits for the picks
            if r is not None:
                obs = yield from self._do_request(r, obs)
                continue
            if job is not None:
                obs = yield from self._put(*job, obs)
                continue
            if self.to_look:
                obs = yield from self._look_again(obs)
                continue
            if self._finished() and self.final_checks < 2:
                # anything left the looks never covered (pushed by a rake, dropped)? the table picture says
                self.final_checks += 1
                obs = yield from self._park(obs)
                regs = self._glance(obs, self.prompts or DEFAULT_PROMPTS)
                cups = self.objects.boxes()
                left = [r for r in regs if not any(np.linalg.norm(np.asarray(r["xy"]) - b.xy) < 0.045 for b in cups)
                        and not any(np.linalg.norm(np.asarray(r["xy"]) - o.xy) < 0.012 for o in self.objects.all.values())]
                spots = focus.spots(left)
                self._say(f"Final check of the table picture: {len(left)} outline(s) outside the cups -> {len(spots)} close look(s).")
                if not spots:
                    self.final_checks = 99
                for xy in spots:
                    obs = yield from self._close_look(xy, obs, label=" (final check)")
                self.board.post("skill_done", f"final check: {len(spots)} looks")
                continue
            if self._finished():
                short = self._short()
                if short and not self._count_search:  # fewer than the person said: one more search, their words
                    self._count_search = True
                    obs = yield from self._search(obs, [w for w, _, _ in short])
                    continue
                gap = "; ".join(f"I found {n} {w}{'' if n == 1 or w.endswith('s') else 's'}, you said {c}" for w, n, c in short)
                self._say("Done: every part the rule names is in its cup." + (f" But {gap}." if gap else ""))
                self.board.post("done", "the rule is carried out" + (f"; short: {gap}" if gap else ""))
                break
            if self.idle_since is None:
                self.idle_since = time.time()
                left = [o for o in self.objects.parts() if not self._done_with(o)]
                self._say(f"Waiting on the ventral stream: {len(left)} part(s) without a cup in the rule.")
                self.board.post("idle", f"{len(left)} measured part(s) have no cup in the rule: "
                                + ", ".join(f"#{o.id}" for o in left)
                                + ("" if self.objects.boxes() else ". No cup or box is measured: set the rule with "
                                   "'table' piles, or request look_closer at a cup's outline if you see one"))
            elif time.time() - max(self.idle_since, self._last_ventral()) > 240:
                self._say("Nothing to do and no word from the ventral stream for a while: stopping.")
                self.board.post("done", "idle too long")
                break
            obs = yield self._still(obs, IDLE_ROWS)
        self.board.stop()
        self.obs = obs

    def _park(self, obs):
        """Back to where the arm started, out of the table camera's view of the table (and ask for its picture)."""
        self._need_table = True
        q_now = np.asarray(obs["joints"][:6], float)
        T_now, T_home = tool_pose(q_now), tool_pose(self.q_home)
        rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), self.q_home, T_home[:3, 3], wpr_of(T_home[:3, :3]))
        if rows is not None:
            obs = yield rows
        return obs

    def _requests_first(self, obs):
        r = self.board.take_request()
        while r is not None:
            obs = yield from self._do_request(r, obs)
            r = self.board.take_request()
        return obs

    def _glance(self, obs, prompts):
        from lab import sam
        img = obs["images"]["table"]
        tp = agree.TablePixels(self.cams["table"], (img.shape[1], img.shape[0]))
        try:
            masks = sam.segment_many(img, list(prompts), crop=self._table_crop(img), scale=1.5)
        except Exception as e:
            viz.narrate(f"(SAM failed on {', '.join(prompts)}: {e})", level="WARN")
            masks = []
        return focus.regions(masks, tp)

    def _table_crop(self, img):
        from calibration.geometry import project
        cam = self.cams["table"]
        W, H = img.shape[1], img.shape[0]
        c = np.array([[x, y, SURFACE_Z] for x in (0.15, 0.65) for y in (-0.45, 0.45)])
        uv = project(cam, c) * [W / cam["size"][0], H / cam["size"][1]]
        x0, y0 = np.clip(uv.min(0) - 30, 0, None).astype(int)
        x1, y1 = np.minimum(uv.max(0) + 30, [W, H]).astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    # ------------------------------------------------------------ a close look

    def _close_look(self, xy, obs, label: str = "", extra: tuple = ()):
        """Look straight down at xy; keep what SAM and depth agree on; queue another look where they don't."""
        rows = self._close_rows(xy, obs)
        self._look_ok = rows is not None
        if rows is None:
            self._say(f"No safe close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm.")
            return obs
        obs = yield rows
        obs = yield self._still(obs, 12)  # the arm settles: the picture and the joints agree
        self._fuse_wrist(obs)
        prompts = list(dict.fromkeys(list(extra) + self._told_words() + list(self.prompts or DEFAULT_PROMPTS)
                                     + list(self.board.prompts or [])))  # the person's words first: they know
        found, masks, issues = focus.measure(obs, self.cams, prompts[:MAX_PROMPTS])
        depth_only = [o for o in found if o.kind == "unknown"]
        agreed = [o for o in found if o.kind != "unknown"]
        confirmed = self.depth_only.update(depth_only, xy, xy, LOOK_R)  # the camera is straight above xy
        self.objects.merge(agreed + confirmed, seen_xy=xy, seen_r=LOOK_R)
        self._draw_wrist_masks(masks, obs)
        for i in issues:
            if i.kind in ("flat", "partial", "no_depth"):
                self._queue_look(i.xy, i.text)
        noise = next((i.text for i in issues if i.kind == "noise"), "")
        told = [i.text for i in issues if i.kind in ("flat", "partial", "no_depth")]
        self._say(f"Close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm{label}: {len(masks)} outlines; "
                  f"{len(agreed)} where SAM and depth agree: " + ("; ".join(f"{o.describe()} ({o.agree:.0%} agree)" for o in agreed) or "none")
                  + (f". They disagree: {'; '.join(told)}" if told else "")
                  + f". Depth only: {len(depth_only)} patch(es), {len(confirmed)} found again from another look"
                  + (f"; {noise}" if noise else "") + ".")
        self._publish(obs, "skill_done", f"close look: {len(agreed)} objects both streams agree on"
                      + (f", {len(confirmed)} things only depth sees (try another word?)" if confirmed else ""))
        return obs

    def _queue_look(self, xy, why: str, extra: tuple = ()) -> None:
        key = tuple(np.round(np.asarray(xy) / 0.01).astype(int))
        if sum(self.relooks.values()) >= MAX_RELOOKS:  # real depth never runs out of doubts: stop asking
            return
        if self.relooks.get(key, 0) >= RELOOKS or any(np.linalg.norm(np.asarray(q) - xy) < 0.01 for q, _, _ in self.to_look):
            return
        self.relooks[key] = self.relooks.get(key, 0) + 1
        self.to_look.append((np.asarray(xy, float), why, tuple(extra)))

    def _look_again(self, obs):
        """The disagreements, looked at again from straight above each: they agree, or it's dropped."""
        while self.to_look:
            xy, why, extra = self.to_look.pop(0)
            obs = yield from self._close_look(xy, obs, label=f" (again: {why})", extra=extra)
        return obs

    def _close_rows(self, xy, obs):
        """A joint move putting the wrist camera straight above xy, as low as the fingertips allow."""
        q_now = np.asarray(obs["joints"][:6], float)
        T_now = tool_pose(q_now)
        heights = self._heights()
        for cam_h in CAM_HEIGHTS:
            for roll in np.radians([0, 90, 180, 270]):
                Tt = _look_at(np.array([xy[0], xy[1], SURFACE_Z]), np.array([0, 0, -1.0]), cam_h, roll) @ np.linalg.inv(self.T_ct)
                tip, wpr = Tt[:3, 3], wpr_of(Tt[:3, :3])
                if not P.reachable(*tip[:2], margin_mm=15):
                    continue
                under = max(heights(tip[:2], 0.06), self._boxy(tip[:2]), self._boxy(xy))  # the hand is wider than the fingers
                if (tip[2] - SURFACE_Z) * 1000 < under + TIP_CLEAR_MM:
                    continue  # the fingers would come down on something
                try:
                    q = tool_ik(tip, wpr, q_now)
                except ValueError:
                    continue
                rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q, tip, wpr)
                if rows is not None and armcheck.check_joints(np.asarray(rows)[:, :6], 0.0).ok:
                    return rows
        return None

    def _boxy(self, xy) -> float:
        """A cup or box SAM outlined on the table picture near xy and not measured yet: assume it's tall (the table
        camera's depth misses thin walls and sees the floor through the opening, so the map calls it low; the fingers
        went into a cup on the real cell)."""
        regs = self.board.regions or []
        for r in regs:
            if any(w in str(r.get("prompt", "")).lower() for w in BOX_WORDS) and np.linalg.norm(np.asarray(r["xy"]) - xy) < 0.09:
                if not any(np.linalg.norm(b.xy - np.asarray(r["xy"])) < 0.03 for b in self.objects.boxes()):
                    return BOX_ASSUME_MM
        return max([cup_top(b) + 10 for b in self.objects.boxes() if np.linalg.norm(b.xy - xy) < 0.09], default=0.0)

    def _heights(self):
        """height_under(xy, r): the tallest thing (mm) the map or the measured objects put within r of xy."""
        m = self.map
        occ = (m.solid() | m.leaning()) & m.clear_of_table()
        pts = m.centers(*np.nonzero(occ))
        # never seen is not empty: unseen space near the table may hold something (capped so the highest look stays
        # possible); seen-empty and seen-solid are what the depth said
        unseen = m.unknown() & m.above_table()
        unseen[:, :, m.k_table + int(UNSEEN_CAP_MM / 1000 / m.v):] = False
        if unseen.any():
            pts = np.vstack([pts, m.centers(*np.nonzero(unseen))])
        objs = list(self.objects.all.values())

        def under(xy, r):
            h = 0.0
            if len(pts):
                near = np.linalg.norm(pts[:, :2] - xy, axis=1) < r
                if near.any():
                    h = (pts[near, 2].max() - SURFACE_Z) * 1000
            for o in objs:
                if np.linalg.norm(o.xy - xy) < r + max(o.size_mm) / 2000:
                    h = max(h, cup_top(o) if o.kind == "box" else o.top_mm)
            return h
        return under

    def _draw_wrist_masks(self, masks, obs) -> None:
        img = obs["images"]["wrist"]
        f = viz.view_scale(img.shape[1])
        import cv2
        w, h = int(img.shape[1] * f), int(img.shape[0] * f)
        seg = np.zeros((h, w), np.uint8)
        for j, m in enumerate(masks[:250]):
            seg[cv2.resize(m["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)] = j + 1
        rr.log("cams/wrist/rgb/sam", rr.AnnotationContext([rr.AnnotationInfo(id=0, label="", color=(0, 0, 0, 0))]
                                                          + [rr.AnnotationInfo(id=j + 1, label=m["prompt"], color=(90, 200, 120, 140))
                                                             for j, m in enumerate(masks[:250])]))
        rr.log("cams/wrist/rgb/sam", rr.SegmentationImage(seg, opacity=0.4))

    # ------------------------------------------------------------ the rule's work

    def _done_with(self, o) -> bool:
        return self._is_placed(o) or self._given_up(o) or any(
            np.linalg.norm(o.xy - b.xy) * 1000 < max(b.size_mm) / 2 + 3 for b in self.objects.boxes())

    def _colour_rule(self) -> None:
        """A plain colour sort ("sort these into the cups", "by colour") needs no deliberation: each part goes to the one
        cup of its colour. Set it, or keep it up to date, while the rule is the arm's own; the ventral stream's rule (or
        the person's words) replace it. Red and orange are left to the ventral stream: on the real cameras an orange cup
        reads hue 4-7 and a red cube hue 2."""
        import re
        from lab.streams import Rule
        rule = self.board.rule
        if rule is not None and rule.source != "dorsal":
            return
        words = (self.instruction or "").lower()
        if self.board.user_rules or not re.search(r"colou?r|cups?|boxes", words):
            return
        said = re.findall(r"\b(red|orange|yellow|green|blue|purple|pink)\b[^.;,]*?\bin(?:to)? the (red|orange|yellow|green|blue|purple|pink)\b", words)
        if any(a != b for a, b in said):
            return  # "red in the yellow cup": the person's mapping, for the ventral stream to carry out
        cups: dict = {}
        for b in self.objects.boxes():
            cups.setdefault(b.color, []).append(b)
        groups = []
        for c, bs in cups.items():
            if c in COLOUR_SURE and len(bs) == 1:
                ids = sorted(o.id for o in self.objects.parts() if o.color == c and not self._done_with(o))
                if ids:
                    groups.append({"name": f"{c} parts", "ids": ids, "to": f"box #{bs[0].id}"})
        if not groups or (rule is not None and rule.groups == groups):
            return
        self.board.set_rule(Rule("each part into the one cup of its colour (the arm's own rule: "
                                 + ", ".join(f"{g['name']} -> {g['to']}" for g in groups) + ")", groups, source="dorsal"))
        self._say("Sort rule (mine, by colour): " + "; ".join(f"{g['name']} {g['ids']} -> {g['to']}" for g in groups) + ".")

    def _next_job(self):
        self._colour_rule()
        rule = self.board.rule
        for o in sorted(self.objects.parts(), key=lambda o: o.xy[1]):
            if self._done_with(o) or o.id in self.failed:
                continue
            dest = self._dest_of(o.id)
            if dest is not None:
                return o, dest
        return None

    def _finished(self) -> bool:
        if self.board.rule is None or any(r.status in ("queued", "doing") for r in self.board.requests):
            return False
        return all(self._done_with(o) or o.id in self.failed for o in self.objects.parts()) and len(self.placed) > 0

    def _do_request(self, r, obs):
        self.idle_since = None
        if r.action == "look_for":
            return (yield from self._look_for(r, obs))
        xy = r.xy if r.xy is not None else (self.objects.all[r.target].xy if r.target in self.objects.all else None)
        if xy is None:
            self.board.finish(r, "refused", f"#{r.target} isn't on the board any more")
            return obs
        if r.action == "break_up" and r.target in self.objects.all:
            t = self.objects.all[r.target]
            plan = self._rake_rows(t, obs, keep_close="far" not in (r.note or ""))
            if plan is None:
                self.board.finish(r, "refused", "no safe rake through it (no free table beyond it, or out of reach)")
                return obs
            rows_list, where, what = plan
            self._say(f"Request {r.id}: break up #{t.id} ({r.note or ''}): {what}.")
            for rows in rows_list:
                obs = yield rows
            for xy in where:
                self.map.forget(xy, 0.03)
            self.misses = [m for m in self.misses if np.linalg.norm(m - t.xy) > 0.04]  # a fresh chance after a rake
            obs = yield from self._close_look(np.asarray(where[1]), obs, label=" (after the rake)")
            self.board.finish(r, "done", f"raked: {len(self.objects.parts())} parts measured now")
            return obs
        if r.action == "put_away" and r.target in self.objects.all:
            return (yield from self._put(self.objects.all[r.target], r.to or self._dest_of(r.target) or "", obs, request=r))
        self._say(f"Request {r.id} from the {'person' if r.who == 'person' else 'ventral stream'}: close look at "
                  f"({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm. It goes first.")
        obs = yield from self._close_look(np.asarray(xy), obs)
        if not self._look_ok:
            self.board.finish(r, "refused", f"no safe close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm "
                                            "(out of reach, or the fingers would hit something)")
        elif r.who == "person":
            self.board.finish(r, "done", self._near_text(xy))
        else:
            self.board.finish(r, "done", f"close look: {len(self.objects.parts())} parts measured so far")
        return obs

    def _look_for(self, r, obs):
        """The person's "look at the screw": SAM finds the word on the table picture (near xy if given), close looks there."""
        word = r.note or "object"
        obs = yield from self._park(obs)  # out of the table camera's way
        regs = self._glance(obs, [word])
        if r.xy is not None:
            regs = [g for g in regs if np.linalg.norm(np.asarray(g["xy"]) - r.xy) < 0.08]
        spots = focus.spots(regs)[:3]
        if not spots:
            self.board.finish(r, "refused", f"SAM finds no '{word}' on the table camera's picture"
                                            + (" near there" if r.xy is not None else ""))
            return obs
        self._say(f"Looking for '{word}': {len(regs)} outline(s) on the table picture -> {len(spots)} close look(s).")
        said = []
        for xy in spots:
            obs = yield from self._close_look(xy, obs, label=f" (looking for '{word}')", extra=(word,))
            said.append(self._near_text(xy) if self._look_ok else
                        f"no safe close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm")
        self.board.finish(r, "done", f"'{word}': " + " | ".join(said))
        return obs

    def _pile_of(self, o) -> str:
        """Its group's pile. The ventral stream renames groups as it goes ("red", "red parts", "red bricks"): a colour
        in the name keeps one pile per colour, whatever it's called this time."""
        from lab.ventral import COLOURS
        name = self._group_of(o.id).lower()
        said = [c for c in COLOURS if c in name.split()]
        return said[0] if len(said) == 1 else name

    def _pile_spot(self, group: str):
        """The next spot in this group's pile: a tight cluster round the pile's first spot; a new pile goes on free
        table at least PILE_GAP from every other pile (so piles never touch), toward the robot's left first."""
        occ = self._occupied()
        mine = self.table_slots.get(group, [])
        others = [q for g, qs in self.table_slots.items() if g != group for q in qs]
        def ok(q, clear_mm):
            return (self._free(q, occ, clear_mm=clear_mm) and all(np.linalg.norm(q - m) > 0.019 for m in mine)
                    and all(np.linalg.norm(q - m) > PILE_GAP for m in others))
        if mine:
            c = mine[0]
            for r in (0.022, 0.038):
                for a in np.radians(np.arange(0, 360, 30)):
                    q = c + r * np.array([np.cos(a), np.sin(a)])
                    if ok(q, 12):
                        return q
            return None
        for x in np.arange(0.30, 0.54, 0.02):
            for y in np.arange(0.32, -0.33, -0.02):
                q = np.array([x, y])
                if ok(q, 30):
                    return q
        return None

    def _put_on_table(self, o, obs, request=None):
        """No cups (or the person asked for piles): its group's pile on free table, a row of spots per group."""
        group = self._pile_of(o)
        self.boxes = {b.id: b for b in self.objects.boxes()}  # _table_spot keeps clear of these
        spot = self._pile_spot(group)
        if spot is None:
            self.failed.add(o.id)
            self._say(f"#{o.id}: no free table for the '{group}' pile; leaving it.")
            self.board.post("gave_up", f"#{o.id}: no free table for the '{group}' pile")
            if request:
                self.board.finish(request, "refused", "no free table")
            return obs
        old = o.xy.copy()
        obs = yield from self._close_look(old, obs, label=f" (before picking #{o.id})")
        fresh = [x for x in self.objects.parts() if np.linalg.norm(x.xy - old) < 0.015]
        if not fresh:
            self.board.post("gave_up", f"#{o.id} not found on the close look before the pick")
            return obs
        o = min(fresh, key=lambda x: np.linalg.norm(x.xy - old))
        g = self._grip_for(o)
        if g is None:
            self.misses.append(np.asarray(o.xy))
            self.board.post("gave_up", f"#{o.id}: neighbours too close for the fingers (ask to break it up)")
            return obs
        under = self._heights()
        safe = max(P.SAFE_MM, max(under(o.xy, 0.03), under(spot, 0.03)) + 15.0 + g.height_mm)
        a = self._approach((g.x, g.y), g.yaw, safe, obs)
        if a is None:
            self.failed.add(o.id)
            self._say(f"#{o.id}: no safe way above it.")
            return obs
        approach, q_over = a
        rows, _ = P.pick_and_place(tcp_uf0(q_over), (g.x, g.y), g.height_mm, g.yaw, tuple(spot), drop_mm=None, safe_mm=safe)
        rep = armcheck.check(rows, q_over)
        if not rep.ok:
            self.failed.add(o.id)
            self._say(f"#{o.id}: the move failed the arm check ({rep.problem}).")
            return obs
        self._draw_grip(g)
        label = o.label or o.what
        self._say(f"Put #{o.id} ({label}) on the table with the '{group}' pile at ({spot[0] * 1000:.0f}, "
                  f"{spot[1] * 1000:.0f}) mm: grip at ({g.x * 1000:.0f}, {g.y * 1000:.0f}) mm, jaws at {g.yaw:.0f} deg.")
        obs = yield approach
        obs = yield rows
        self.map.forget((g.x, g.y), 0.012)
        obs = yield from self._close_look(np.array([g.x, g.y]), obs, label=" (checking the pick)")
        still = [x for x in self.objects.parts() if np.linalg.norm(x.xy - [g.x, g.y]) < 0.006]
        if still:
            self.misses.append(np.array([g.x, g.y]))
            self.picks["failed"] += 1
            self._say(f"#{o.id}: still there after the pick.")
        else:
            self.picks["verified"] += 1
            self.placed.append((np.asarray(spot), "table"))
            self.table_slots.setdefault(group, []).append(np.asarray(spot))
        self._publish(obs, "skill_done", f"put #{o.id} ({label}) in the '{group}' pile on the table"
                      + (" (it didn't come up)" if still else ""))
        if request:
            self.board.finish(request, "done", "put on the table")
        return obs

    # ------------------------------------------------------------ what the person told it

    def _told_words(self) -> list[str]:
        return list(self.board.told())

    @staticmethod
    def _matches(word: str, text: str) -> bool:
        """'red cube' matches 'red cube #3' and 'cubes'; every word of it, plural or not."""
        have = {t.rstrip("s") for t in text.lower().replace("#", " ").replace("_", " ").split()}
        return all(t.rstrip("s") in have for t in word.lower().split())

    def _told_size(self, o) -> float | None:
        for w, d in self.board.told().items():
            if d["size_mm"] and self._matches(w, f"{o.color} {o.what} {o.label or ''}"):
                return float(d["size_mm"])
        return None

    def _short(self) -> list[tuple[str, int, int]]:
        """(word, found, said) for each count the person gave that the table and the cups don't reach."""
        with self.board.cond:
            put = [n for c in self.board.contents.values() for n in c]
        out = []
        for w, d in self.board.told().items():
            if not d["count"]:
                continue
            n = sum(self._matches(w, f"{o.color} {o.what} {o.label or ''}") for o in self.objects.parts()
                    if not any(np.linalg.norm(o.xy - b.xy) * 1000 < max(b.size_mm) / 2 + 3 for b in self.objects.boxes()))
            n += sum(self._matches(w, name) for name in put)
            if n < int(d["count"]):
                out.append((w, n, int(d["count"])))
        return out

    def _search(self, obs, words):
        """The table picture with the person's words; a close look at every outline nothing measured covers."""
        obs = yield from self._park(obs)
        regs = self._glance(obs, words)
        cups = self.objects.boxes()
        left = [r for r in regs if not any(np.linalg.norm(np.asarray(r["xy"]) - b.xy) < 0.045 for b in cups)
                and not any(np.linalg.norm(np.asarray(r["xy"]) - o.xy) < 0.012 for o in self.objects.all.values())]
        spots = focus.spots(left)
        self._say(f"Fewer {', '.join(w if w.endswith('s') else w + 's' for w in words)} than you said: searching the table picture for them; "
                  f"{len(left)} new outline(s) -> {len(spots)} close look(s).")
        for xy in spots:
            obs = yield from self._close_look(xy, obs, label=" (searching with your words)", extra=tuple(words))
        return obs

    def _near_text(self, xy, r: float = LOOK_R) -> str:
        """What the board holds around xy after a look, for the person: what both streams agree on, what only depth sees."""
        near = sorted((o for o in self.objects.all.values() if np.linalg.norm(o.xy - xy) < r),
                      key=lambda o: np.linalg.norm(o.xy - xy))
        seen = [f"{o.describe()}, SAM and depth agree {o.agree:.0%}" for o in near if o.kind != "unknown"]
        only = [o.describe() for o in near if o.kind == "unknown"]
        queued = sum(np.linalg.norm(q - xy) < r for q, _, _ in self.to_look)
        return (f"looked at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm: " + ("; ".join(seen) or "nothing both streams agree on")
                + (f". Only depth sees: {'; '.join(only)}" if only else "")
                + (f". {queued} spot(s) there where they disagree, queued for another look" if queued else ""))

    # ------------------------------------------------------------ what it knows, for the chat

    def status(self) -> str:
        b = self.board
        lines = ["Mind: focus (the table picture with SAM picks where to look; close looks keep only what SAM's outline "
                 "and the wrist depth agree on). Coordinates: mm in the robot's base frame, x out from the base, "
                 "+y to the robot's left."]
        if self.hold:
            lines.append("Holding: only looking where the person asks; not sorting until asked.")
        with b.cond:
            labels, rule, contents = dict(b.labels), b.rule, {k: list(v) for k, v in b.contents.items()}
            events = list(b.events)
        parts = [o for o in self.objects.parts()]
        boxes = self.objects.boxes()
        unknown = [o for o in self.objects.all.values() if o.kind == "unknown"]
        def where(o):
            if o.id in self.failed or self._given_up(o):
                return "gave up on it"
            if self._done_with(o):
                return "put away"
            return f"for {self._dest_of(o.id)}" if self._dest_of(o.id) else "no cup in the rule yet"
        lines.append(f"Parts measured ({len(parts)}):" + "".join(
            f"\n  {o.describe()}" + (f", called '{labels[o.id]}'" if o.id in labels else "")
            + f", SAM and depth agree {o.agree:.0%}, {where(o)}" for o in parts))
        lines.append(f"Cups and boxes ({len(boxes)}):" + "".join(
            f"\n  {o.describe()}" + (f", holds: {', '.join(contents[o.id])}" if contents.get(o.id) else "") for o in boxes))
        if unknown:
            lines.append("Only depth sees these (found again from a second look, no SAM outline; steered around, never "
                         "picked):" + "".join(f"\n  {o.describe()}" for o in unknown))
        if rule is not None:
            lines.append(f"Sort rule ({rule.source}): {rule.text}" + "".join(
                f"\n  {g['name']}: {', '.join('#' + str(i) for i in g['ids'])} -> {g['to']}" for g in rule.groups))
        else:
            lines.append("Sort rule: none yet.")
        lines.append(f"Requests: {b.queue_text()}. Second looks queued: {len(self.to_look)}"
                     + "".join(f"\n  ({q[0] * 1000:.0f}, {q[1] * 1000:.0f}) mm: {why}" for q, why, _ in self.to_look[:5]))
        trouble = [f"{k}: {t}" for _, _, k, t in events if k in ("gave_up", "request_refused")][-6:]
        if trouble:
            lines.append("Gave up or refused lately (oldest first):\n  " + "\n  ".join(trouble))
        told = self.board.told()
        if told:
            lines.append("What the person told me is on the table: " + "; ".join(
                f"{w}" + (f" x{d['count']}" if d["count"] else "") + (f", {d['size_mm']} mm" if d["size_mm"] else "")
                for w, d in told.items()))
            short = self._short()
            if short:
                lines.append("Fewer than they said, so far: " + "; ".join(f"{w}: {n} of {c}" for w, n, c in short))
        lines.append(f"Put away so far: {len(self.placed)}.")
        return "\n".join(lines)

    def _grip_for(self, o):
        """Jaws placed from o's outline (any shape): narrow enough between them, both fingers clear of o and of every
        other object taller than the fingertips' height (lab.grasp.from_outline)."""
        h = max(2.5, min(GRIP_SHARE * o.top_mm, o.top_mm - 3.0))
        others = [x.footprint() for x in self.objects.all.values() if x.id != o.id and x.top_mm > h]
        grips = grasp.from_outline(o.footprint(), others, h)
        if not grips and o.kind == "part":
            # real outlines come out 20-27 mm for ~15 mm cubes (smeared edges): try the outline shrunk to the part's
            # real size round its centre (the person's size if they said one)
            fp = o.footprint()
            c = fp.mean(0)
            span = np.ptp(fp, 0).max() * 1000
            real = self._told_size(o) or REAL_PART_MM
            if span > real:
                grips = grasp.from_outline(c + (fp - c) * (real / span), others, h)
                if grips:
                    self._say(f"#{o.id}: its outline is {span:.0f} mm across; gripping it as {real:.0f} mm round its centre.")
        return grips[0] if grips else None

    def _draw_grip(self, g) -> None:
        """Where the two fingers come down (their footprints), at the fingertips' height."""
        a = np.radians(g.yaw)
        d, e = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        z = SURFACE_Z + g.height_mm / 1000
        strips = []
        for side in (-1, 1):
            near, far = side * grasp.GAP / 2000, side * (grasp.GAP / 2 + grasp.FINGER) / 2000
            c = [np.array([g.x, g.y]) + s0 * d + s1 * e * grasp.WIDTH / 2000 for s0, s1 in
                 ((near, -1), (far, -1), (far, 1), (near, 1), (near, -1))]
            strips.append([[*p, z] for p in c])
        rr.log("mind/plan/grip", rr.LineStrips3D(strips, colors=[[60, 220, 90]], radii=0.0006))

    def _put(self, o, dest: str, obs, request=None):
        """Look closely right above o, grip it at its fresh position, drop it in the cup (or set it down in its group's
        pile on the table), look to check."""
        self.idle_since = None
        box = None
        if dest.strip().lower() == "table":
            return (yield from self._put_on_table(o, obs, request))
        if dest.strip().lower().startswith("box"):
            try:
                box = self.objects.all.get(int(dest.split("#")[-1]))
            except ValueError:
                box = None
        if box is None or box.kind != "box":
            self._say(f"#{o.id}: the rule says {dest}, which isn't a measured cup; asking the ventral stream.")
            self.board.forget_near(o.xy, 0.001)
            self.board.post("gave_up", f"#{o.id}: {dest} isn't a measured cup")
            if request:
                self.board.finish(request, "refused", f"no {dest}")
            return obs
        old = o.xy.copy()
        obs = yield from self._close_look(old, obs, label=f" (before picking #{o.id})")
        fresh = [x for x in self.objects.parts() if np.linalg.norm(x.xy - old) < 0.015]
        if not fresh:
            self._say(f"#{o.id} isn't where it was any more; leaving it for the ventral stream.")
            self.board.post("gave_up", f"#{o.id} not found on the close look before the pick")
            return obs
        o = min(fresh, key=lambda x: np.linalg.norm(x.xy - old))
        g = self._grip_for(o)
        said = self._told_size(o)
        if g is None and said and max(o.size_mm) > said + SIZE_TOL_MM:
            self.failed.add(o.id)
            viz.narrate(f"#{o.id} measures {o.size_mm[0]:.0f} x {o.size_mm[1]:.0f} mm and you said {said:.0f} mm: my "
                        f"outline is too big for the fingers, so I'm not gripping it blind. Can you nudge it, or check it?",
                        level="WARN")
            self.board.post("gave_up", f"#{o.id}: measures {max(o.size_mm):.0f} mm, the person said {said:.0f}: too wide to grip")
            return obs
        if g is None:
            self._say(f"#{o.id}: no grip with both fingers clear of its neighbours; the others first.")
            self.misses.append(np.asarray(o.xy))  # tried again later (after neighbours leave), twice at most
            self.board.post("gave_up", f"#{o.id}: neighbours too close for the fingers (ask to break it up)")
            return obs
        safe = max(P.SAFE_MM, cup_top(box) + 15.0 + g.height_mm)
        a = self._approach((g.x, g.y), g.yaw, safe, obs)
        if a is None:
            self.failed.add(o.id)
            self._say(f"#{o.id}: no safe way above it.")
            return obs
        approach, q_over = a
        rows, _ = P.pick_and_place(tcp_uf0(q_over), (g.x, g.y), g.height_mm, g.yaw, tuple(box.xy),
                                   drop_mm=cup_top(box) + DROP_ABOVE_RIM_MM, safe_mm=safe)
        rep = armcheck.check(rows, q_over)
        if not rep.ok:
            self.failed.add(o.id)
            self._say(f"#{o.id}: the pick failed the arm check ({rep.problem}).")
            return obs
        self._draw_grip(g)
        label = o.label or o.what
        self._say(f"Put #{o.id} ({label}) in the {box.color} cup #{box.id}: grip at ({g.x * 1000:.0f}, {g.y * 1000:.0f}) mm, "
                  f"{g.height_mm:.0f} mm up, jaws at {g.yaw:.0f} deg (from a look {o.pixels} depth px strong).")
        obs = yield approach
        obs = yield rows
        self.map.forget((g.x, g.y), 0.012)
        obs = yield from self._close_look(np.array([g.x, g.y]), obs, label=" (checking the pick)")
        still = [x for x in self.objects.parts() if np.linalg.norm(x.xy - [g.x, g.y]) < 0.006]
        if still:
            self.misses.append(np.array([g.x, g.y]))
            self.picks["failed"] += 1
            self._say(f"#{o.id}: still there after the pick.")
        else:
            self.picks["verified"] += 1
            self.placed.append((np.array(box.xy), dest))
            with self.board.cond:
                self.board.contents.setdefault(box.id, []).append(label if o.color in label else f"{o.color} {label}")
        self._publish(obs, "skill_done", f"put #{o.id} ({label}) in the {box.color} cup" + (" (it didn't come up)" if still else ""))
        if request:
            self.board.finish(request, "done", "put away")
        return obs
