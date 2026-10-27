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

CAM_HEIGHTS = (0.14, 0.17, 0.21)  # camera above the table for a close look, lowest first (fingertips 118 mm lower)
TIP_CLEAR_MM = 12.0               # the fingertips stay this far above anything the map knows under them
PROMPT_WAIT_S = 60.0              # wall time to wait for the ventral stream's prompts before using the defaults
DEFAULT_PROMPTS = ["cube", "cup", "small object"]
LOOK_R = 0.035                    # a close look covers this well around its centre (objects there not found = gone)
DROP_ABOVE_RIM_MM = 4.0
GRIP_SHARE = 0.45                 # fingertips close at most this far up the part
NEIGHBOUR_MM = 2.0                # a finger keeps this much air from any other object


class FocusMind(TwoStreamMind):
    JOB = "focus"

    def reset(self) -> None:
        super().reset()
        self.objects = focus.Objects()
        self.board.mode = "focus"
        self.prompts: list[str] | None = None
        self.T_ct = _cam_in_tool(self.cams["wrist_depth"])
        self.final_checks = 0

    # ------------------------------------------------------------ the board, with measured objects

    def _publish(self, obs, kind: str, text: str = "") -> None:
        img = obs["images"]["table"]
        arm = agree.arm_mask(np.asarray(obs["joints"][:6], float), self.cams["table"], (img.shape[1], img.shape[0]))
        self.board.set_things(list(self.objects.all.values()), [], img, arm)
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
        self._draw()
        self._publish(obs, "started", "the table camera's picture is on the board: choose SAM prompts (segment)")
        # 1. the ventral stream's prompts and regions (it segments the table picture)
        t0 = time.time()
        while self.board.regions is None and time.time() - t0 < PROMPT_WAIT_S and not self.board.stopped:
            obs = yield self._still(obs, 5)
        regs = self.board.regions
        if not regs:
            self._say("No regions from the ventral stream: looking where SAM finds anything with default prompts.")
            regs = self._glance(obs, DEFAULT_PROMPTS)
        self.prompts = self.board.prompts or DEFAULT_PROMPTS
        spots = focus.spots(regs)
        self._say(f"{len(regs)} outlines on the table picture -> {len(spots)} close looks.")
        # 2. close looks only there
        for xy in spots:
            obs = yield from self._requests_first(obs)
            obs = yield from self._close_look(xy, obs)
        self.boxes = {o.id: o for o in self.objects.boxes()}
        self._publish(obs, "map_done", f"measured {len(self.objects.parts())} parts and {len(self.boxes)} cups "
                                       f"in {len(spots)} close looks")
        # 3. the rule's work, requests first
        while True:
            r = self.board.take_request()
            if r is not None:
                obs = yield from self._do_request(r, obs)
                continue
            job = self._next_job()
            if job is not None:
                obs = yield from self._put(*job, obs)
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
                self._say("Done: every part the rule names is in its cup.")
                self.board.post("done", "the rule is carried out")
                break
            if self.idle_since is None:
                self.idle_since = time.time()
                left = [o for o in self.objects.parts() if not self._done_with(o)]
                self._say(f"Waiting on the ventral stream: {len(left)} part(s) without a cup in the rule.")
                self.board.post("idle", f"{len(left)} measured part(s) have no cup in the rule: "
                                + ", ".join(f"#{o.id}" for o in left))
            elif time.time() - max(self.idle_since, self._last_ventral()) > 240:
                self._say("Nothing to do and no word from the ventral stream for a while: stopping.")
                self.board.post("done", "idle too long")
                break
            obs = yield self._still(obs, IDLE_ROWS)
        self.board.stop()
        self.obs = obs

    def _park(self, obs):
        """Back to where the arm started, out of the table camera's view of the table."""
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
        masks = []
        for p in prompts:
            try:
                masks += sam.segment(img, p, crop=self._table_crop(img), scale=1.5)
            except Exception as e:
                viz.narrate(f"(SAM failed on '{p}': {e})", level="WARN")
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

    def _close_look(self, xy, obs, label: str = ""):
        rows = self._close_rows(xy, obs)
        if rows is None:
            self._say(f"No safe close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm.")
            return obs
        obs = yield rows
        obs = yield self._still(obs, 12)  # the arm settles: the picture and the joints agree
        self._fuse_wrist(obs)
        found, masks = focus.measure(obs, self.cams, self.prompts or DEFAULT_PROMPTS)
        self.objects.merge(found, seen_xy=xy, seen_r=LOOK_R)
        self._draw_wrist_masks(masks, obs)
        self._say(f"Close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm{label}: {len(masks)} outlines, "
                  f"{len(found)} standing up: " + ("; ".join(o.describe() for o in found) or "nothing"))
        self._publish(obs, "skill_done", f"close look: {len(found)} objects measured")
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
                under = heights(tip[:2], 0.03)
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

    def _heights(self):
        """height_under(xy, r): the tallest thing (mm) the map or the measured objects put within r of xy."""
        m = self.map
        occ = (m.solid() | m.leaning()) & m.clear_of_table()
        idx = np.nonzero(occ)
        pts = m.centers(*idx)
        objs = list(self.objects.all.values())

        def under(xy, r):
            h = 0.0
            if len(pts):
                near = np.linalg.norm(pts[:, :2] - xy, axis=1) < r
                if near.any():
                    h = (pts[near, 2].max() - SURFACE_Z) * 1000
            for o in objs:
                if np.linalg.norm(o.xy - xy) < r + max(o.size_mm) / 2000:
                    h = max(h, o.top_mm)
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

    def _next_job(self):
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
        self._say(f"Request {r.id} from the ventral stream: close look at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm. It goes first.")
        obs = yield from self._close_look(np.asarray(xy), obs)
        self.board.finish(r, "done", f"close look: {len(self.objects.parts())} parts measured so far")
        return obs

    def _grip_for(self, o):
        """Jaws centred on o, closing across its narrow side (or the other way), clear of every other object."""
        others = [x for x in self.objects.all.values() if x.id != o.id]
        h = max(2.5, min(GRIP_SHARE * o.top_mm, o.top_mm - 3.0))
        long_, short = o.size_mm
        a = np.radians(o.angle)
        along = np.array([np.cos(a), np.sin(a)])
        centres = [o.xy]
        if long_ > grasp.GAP - 2:  # longer than the jaws open: likely touching things (one outline): take an end
            off = (long_ - short) / 2000
            centres = [o.xy + along * off, o.xy - along * off]
        fits_any = np.hypot(long_, short) < grasp.GAP - 2  # its diagonal fits: any jaw angle grips it
        if fits_any:
            yaws = sorted(np.arange(0, 180, 15.0), key=lambda y: min(abs((y - o.angle) % 90), 90 - abs((y - o.angle) % 90)))
        else:
            yaws = [(o.angle + 90) % 180] + ([o.angle] if long_ <= grasp.GAP - 2 else [])
        for c0 in centres:
            for yaw in yaws:
                d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
                ok = True
                for side in (-1, 1):
                    c = c0 + side * d * (grasp.GAP / 2 + grasp.FINGER / 2) / 1000
                    for x in others:
                        reach = (np.hypot(*x.size_mm) / 2 + np.hypot(grasp.FINGER, grasp.WIDTH) / 2 + NEIGHBOUR_MM) / 1000
                        if np.linalg.norm(c - x.xy) < reach and x.top_mm > h:
                            ok = False
                if ok:
                    return grasp.Grip(float(c0[0]), float(c0[1]), float(h), float(yaw), 0, 1.0)
        return None

    def _put(self, o, dest: str, obs, request=None):
        """Look closely right above o, grip it at its fresh position, drop it in the cup, look to check."""
        self.idle_since = None
        box = None
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
        if g is None:
            self._say(f"#{o.id}: no grip with both fingers clear of its neighbours; the others first.")
            self.misses.append(np.asarray(o.xy))  # tried again later (after neighbours leave), twice at most
            self.board.post("gave_up", f"#{o.id}: neighbours too close for the fingers (ask to break it up)")
            return obs
        safe = max(P.SAFE_MM, box.top_mm + 15.0 + g.height_mm)
        a = self._approach((g.x, g.y), g.yaw, safe, obs)
        if a is None:
            self.failed.add(o.id)
            self._say(f"#{o.id}: no safe way above it.")
            return obs
        approach, q_over = a
        rows, _ = P.pick_and_place(tcp_uf0(q_over), (g.x, g.y), g.height_mm, g.yaw, tuple(box.xy),
                                   drop_mm=box.top_mm + DROP_ABOVE_RIM_MM, safe_mm=safe)
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
                self.board.contents.setdefault(box.id, []).append(label)
        self._publish(obs, "skill_done", f"put #{o.id} ({label}) in the {box.color} cup" + (" (it didn't come up)" if still else ""))
        if request:
            self.board.finish(request, "done", "put away")
        return obs
