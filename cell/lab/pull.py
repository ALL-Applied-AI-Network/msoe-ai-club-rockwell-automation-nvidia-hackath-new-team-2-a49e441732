"""Pull the wrist camera and the joints from the cell as fast as it answers, and work out when each picture was taken.

    pull = Pull(url, token)          # threads start asking at once
    ...
    pull.fit()                       # the camera's beat, from when each new picture first showed up
    pull.measure_delay(t0, t1)       # the fixed picture delay, from a wrist twitch between t0 and t1 (cell clock)
    for f in pull.frames: f.tc       # capture time of every picture; pull.joints_at(f.tc) is where the arm was

The cell answers one snapshot per request: the joints read when the request arrived (its "t"), and the newest picture
the camera had ready then. That picture is already old (a fixed pipeline delay, plus a jitter, plus wherever in the
camera's 0.1 s cycle we asked), and no capture time comes with it. So:
    inflight       wrist requests kept going at once: every frame is seen soon after it's ready, none are missed
    thumbnails     tiny (64 x 36, ~1 KB) requests in between: a denser record of where the arm was, and of when each
                   new picture became ready (a thumbnail changes the moment the full picture does); each is matched to
                   its full picture by shrinking that and comparing
    new frames     a repeated picture has the same bytes; a frame's first appearance (t_first) bounds when it was ready
    the beat       new frames come every `period` s: a lower-envelope line through t_first against the frame count gives
                   g_k = a + period k, the capture times plus the smallest delay
    the delay      D: g_k - D is the capture time. Found once per run by twitching the wrist: the D for which the
                   picture-to-picture change best follows the camera's motion computed from the joints
Every time here is on the cell's clock (the answers' t), so nothing depends on our clock or the network.
"""
from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.geometry import flange_pose
from cell_client import CellClient

PERIOD = 0.1  # the wrist camera's nominal frame period (10 fps, the organizers' setting)
SIZE = (848, 480)
D_RANGE = (0.0, 0.30)  # the fixed delay searched (s)


@dataclass
class Frame:
    hash: str
    jpeg: bytes
    t_first: float  # earliest answer carrying it (cell clock)
    seen: int = 1
    sure: bool = False  # its first appearance pinned by a thumbnail that clearly matched it
    k: int = 0  # frame count along the camera's beat
    tc: float | None = None  # capture time, once fit() and measure_delay() have run
    truth: dict = field(default_factory=dict)  # sim only: what the server says it was (for checks, never for fitting)


class Pull:
    def __init__(self, url: str, token: str, ca: str | None = None, inflight: int = 3, joint_threads: int = 3,
                 quality: int = 90):
        self.url, self.token, self.ca, self.quality = url, token, ca, quality
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.frames: list[Frame] = []
        self.by_hash: dict[str, Frame] = {}
        self.jt: list[float] = []  # joint record: cell times
        self.jq: list[np.ndarray] = []  # and joints (J1-J6, gripper)
        self.thumbs: dict[str, list] = {}  # thumbnail hash -> [first time seen, small grey picture]
        self.answers = 0
        self.errors = 0
        self.last: dict = {}  # the newest answer's status fields (queued, status, t)
        self.a, self.period, self.delay = None, PERIOD, None
        self.threads = [threading.Thread(target=self._run, args=(True,), daemon=True) for _ in range(inflight)]
        self.threads += [threading.Thread(target=self._run, args=(False,), daemon=True) for _ in range(joint_threads)]
        for th in self.threads:
            th.start()

    def stop(self) -> None:
        self.stop_ev.set()
        for th in self.threads:
            th.join(timeout=5)

    def _run(self, wrist: bool) -> None:
        c = CellClient(self.url, self.token, self.ca)
        size = SIZE if wrist else (64, 36)
        params = {"cameras": "wrist", "width": size[0], "height": size[1], "format": "jpeg", "quality": self.quality}
        while not self.stop_ev.is_set():
            try:
                o = c._call("GET", "observation", params=params)
            except Exception:  # one failed request must not end the stream
                with self.lock:
                    self.errors += 1
                time.sleep(0.05)
                continue
            self._take(o, wrist)

    def _take(self, o: dict, wrist: bool) -> None:
        t = float(o["t"])
        q = np.asarray(o["joints"], float)
        img = (o.get("images") or {}).get("wrist")
        b64 = img if wrist else None
        with self.lock:
            self.answers += 1
            if img and not wrist:
                h = hashlib.md5(img.encode()).hexdigest()
                th = self.thumbs.get(h)
                if th is None:
                    small = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_GRAYSCALE)
                    self.thumbs[h] = [t, small.astype(np.float32)]
                else:
                    th[0] = min(th[0], t)
            k = int(np.searchsorted(self.jt, t))
            self.jt.insert(k, t)
            self.jq.insert(k, q)
            if t >= self.last.get("t", -1e18):
                self.last = {"t": t, "queued": o.get("queued"), "status": o.get("status")}
            if not b64:
                return
            h = hashlib.md5(b64.encode()).hexdigest()
            f = self.by_hash.get(h)
            if f is None:
                f = Frame(h, base64.b64decode(b64), t, truth=o.get("_truth") or {})
                self.by_hash[h] = f
                self.frames.append(f)
            else:
                f.seen += 1
                f.t_first = min(f.t_first, t)

    # ------------------------------------------------------------ where the arm was

    def joints_at(self, t) -> np.ndarray:
        """Joints (J1-J6) at cell time(s) t, linear between the recorded answers."""
        with self.lock:
            jt, jq = np.asarray(self.jt), np.asarray(self.jq)[:, :6]
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([np.interp(t, jt, jq[:, j]) for j in range(6)], -1)

    # ------------------------------------------------------------ when each picture was taken

    def fit(self) -> None:
        """The camera's beat. A picture first shows up a delay after it was taken: never less than the pipeline's,
        usually a little more, now and then a lot more. With frame k taken at a + period k (a: capture plus the
        smallest delay), each picture's count is chosen (_assign) so that its extra delay t_first - (a + period k) is
        never negative and as small as likely, counts strictly increase, and skipped frames are rare. a and the period
        are then refitted under the frames a thumbnail timed (their first appearance is tight), and again. The period
        stays at the nominal 0.1 s (the organizers' setting; camera and server clocks drift by parts per million)
        unless the data says otherwise by more than 0.5 ms (period_seen)."""
        self._thumb_times()
        with self.lock:
            fr = sorted(self.frames, key=lambda f: f.t_first)
        t = np.array([f.t_first for f in fr])
        sure = np.array([f.sure for f in fr])
        if sure.sum() < 20:  # too few thumbnail-timed frames: fit on all of them
            sure[:] = True
        P = PERIOD
        ph = (t[sure] - t[0]) % P  # first guess at a: the start of the bunch of arrivals folded onto the period
        cand = np.arange(0, P, 0.001)
        edge = cand[int(np.argmax([np.sum(((ph - c) % P) < 0.3 * P) for c in cand]))]
        a = t[0] + edge - 0.02 - P
        for _ in range(4):
            k = _assign(t, a, P)
            # the lower edge in six stretches of the run (10th percentile of the extra delay in each), and a line
            # through those: a steady period over the whole run (a lower envelope over all frames tilted with noise)
            r = t - (a + P * k)
            idx = np.nonzero(sure)[0]
            ks, qs = [], []
            for blk in np.array_split(idx, 6):
                if len(blk) >= 5:
                    ks.append(k[blk].mean())
                    qs.append(np.percentile(r[blk], 10))
            if len(ks) >= 2:  # the period as the data has it: kept as a check, used only if clearly not nominal
                slope, icpt = np.polyfit(ks, qs, 1)
                self.period_seen = P + slope
                if abs(slope) > 0.0005:
                    a, P = a + icpt, P + slope
            a += float(np.percentile(t[sure] - (a + P * k[sure]), 1))
        k = _assign(t, a, P)
        k0 = k[0]
        a, k = a + P * k0, k - k0
        for f, kk in zip(fr, k):
            f.k = int(kk)
        self.a, self.period = float(a), float(P)
        self._place()

    def _thumb_times(self) -> None:
        """Pull each picture's first appearance earlier with its thumbnail: the thumbnail is matched to the full
        picture (shrunk the same way) among those first seen within 0.3 s of it; only a clear match counts."""
        with self.lock:
            thumbs = sorted(self.thumbs.values(), key=lambda x: x[0])
            frames = sorted(self.frames, key=lambda f: f.t_first)
        if not thumbs or not frames:
            return
        ft = np.array([f.t_first for f in frames])
        smalls = {}
        self.thumb_stats = {"thumbs": len(thumbs), "matched": 0, "earlier": 0, "no_near": 0, "unclear": 0}
        for t, small in thumbs:
            near = np.nonzero(np.abs(ft - t) < 0.3)[0]
            if len(near) == 0:
                self.thumb_stats["no_near"] += 1
                continue
            d = []
            for i in near:
                if i not in smalls:
                    g = cv2.imdecode(np.frombuffer(frames[i].jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                    smalls[i] = cv2.resize(g, small.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
                d.append(np.abs(smalls[i] - small).mean())
            d = np.array(d)
            j = int(np.argmin(d))
            second = np.partition(d, 1)[1] if len(d) > 1 else np.inf
            if d[j] < 2.0 and d[j] < 0.4 * second:  # clearly this picture, not one of its neighbours
                f = frames[near[j]]
                self.thumb_stats["matched"] += 1
                self.thumb_stats["earlier"] += t < f.t_first
                f.t_first = min(f.t_first, t)
                f.sure = True
            else:
                self.thumb_stats["unclear"] += 1
                self.thumb_stats.setdefault("d", []).append((round(float(d[j]), 2), round(float(second), 2)))

    def grid(self, f: Frame) -> float:
        return self.a + self.period * f.k

    def _place(self) -> None:
        if self.a is None or self.delay is None:
            return
        for f in self.frames:
            f.tc = self.grid(f) - self.delay

    def measure_delay(self, t0: float, t1: float, T_fc: np.ndarray, depth_m: float = 0.3, fx: float = 410.5) -> dict:
        """The fixed delay D from a twitch between cell times t0 and t1: for each D, the camera's motion between
        consecutive frames (from the joints at g_k - D) against how much the picture changed; D where they agree best."""
        if self.a is None:
            self.fit()
        fr = sorted((f for f in self.frames if t0 <= self.grid(f) <= t1 + 0.4), key=lambda f: f.k)
        g = np.array([self.grid(f) for f in fr])
        imgs = [cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE) for f in fr]
        consec = np.array([fr[i].k - fr[i - 1].k == 1 for i in range(1, len(fr))])
        flow = np.array([_flow(imgs[i - 1], imgs[i]) for i in range(1, len(fr))])  # px the picture moved
        ok = consec & np.isfinite(flow)
        Ds = np.arange(D_RANGE[0], D_RANGE[1] + 1e-9, 0.002)
        score = []
        for D in Ds:
            T = [flange_pose(q) @ T_fc for q in self.joints_at(g - D)]
            move = []
            for i in range(1, len(T)):
                R = T[i - 1][:3, :3].T @ T[i][:3, :3]
                ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                move.append(fx * (ang + np.linalg.norm(T[i][:3, 3] - T[i - 1][:3, 3]) / depth_m))
            score.append(_corr(np.array(move)[ok], flow[ok]))
        score = np.array(score)
        best = np.nonzero(score >= score.max() - 0.002)[0]
        pick = int(np.argmax(score))
        change = flow
        self.delay = float(Ds[pick])
        self._place()
        return {"delay_s": self.delay, "score": float(score[pick]), "plateau_ms": (Ds[best[0]] * 1000, Ds[best[-1]] * 1000), "frames": len(fr), "Ds": Ds, "scores": score,
                "change": change, "consec": consec, "g": g}


def _assign(t: np.ndarray, a: float, P: float, tau: float = 0.025, miss: float = 3.0, early: float = 0.003) -> np.ndarray:
    """Frame counts for first appearances t (in order) on the beat a + P k, by Viterbi: each picture's extra delay
    r = t - (a + P k) costs r / tau (likely small), a negative one costs -r / early (unlikely, but a is only an
    estimate: forbidding it pushed a whole stretch of frames one count low), and every skipped frame between
    consecutive pictures costs `miss`. Counts strictly increase (the cell sends the newest ready
    picture, so arrival order is capture order)."""
    base = np.floor((t - a) / P).astype(int)
    opts = np.stack([base - 2, base - 1, base, base + 1], 1)
    r = t[:, None] - (a + P * opts)
    cost = np.where(r < 0, -r / early, r / tau)
    acc, back = cost[0].copy(), np.zeros(opts.shape, int)
    for i in range(1, len(t)):
        gap = opts[i][:, None] - opts[i - 1][None, :]
        step = np.where(gap >= 1, (gap - 1) * miss, 1e9) + acc[None, :]
        back[i] = np.argmin(step, 1)
        acc = step[np.arange(4), back[i]] + cost[i]
    j = int(np.argmin(acc))
    k = np.zeros(len(t), int)
    for i in range(len(t) - 1, -1, -1):
        k[i] = opts[i, j]
        j = back[i, j]
    return k


def _flow(a: np.ndarray, b: np.ndarray) -> float:
    """How far the picture moved from a to b: the median length of tracked corners (checked forward and back), px.
    0 when nothing moved; nan when there's nothing to track."""
    p0 = cv2.goodFeaturesToTrack(a, 300, 0.01, 7)
    if p0 is None or len(p0) < 8:
        return np.nan
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
    p2, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p2 - p0, axis=2).ravel() < 0.5)
    if good.sum() < 6:
        return np.nan
    return float(np.median(np.linalg.norm(p1 - p0, axis=2).ravel()[good]))


def _corr(x, y) -> float:
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


# a wrist twitch: J6 turns (deg) and holds (rows), different speeds so starts and stops fall at many camera phases
TWITCH = [(0, 8), (12, 2), (0, 6), (-12, 3), (0, 7), (8, 1), (0, 5), (-8, 4), (0, 6), (15, 3), (0, 4), (-15, 2), (0, 5),
          (6, 1), (0, 3), (-6, 1), (0, 6), (10, 2), (0, 5), (-10, 1), (0, 4), (12, 3), (0, 3), (-12, 2), (0, 7),
          (5, 1), (0, 4), (-5, 2), (0, 8)]


def twitch(q) -> list:
    """Joint rows: the wrist rolls back and forth from q (J1-J6, deg) and ends where it started, still."""
    q = np.asarray(q, float)[:6].copy()
    rows = []
    for turn, n in TWITCH:
        for i in range(n):
            if turn:
                q[5] += turn / n
            rows.append([*q.tolist(), 0.0])
    return rows
