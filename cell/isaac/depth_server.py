"""Depth from the wrist video, on the GPU box: frames with known poses come in, a grid of evidence goes out.

    ~/cosmos-env/bin/python isaac/depth_server.py            (from ~/cell-isaac; isaac/run_depth.sh does this)

    POST /reset {"fx", "k1", "size": [w, h]}         a new scene: forget every frame and every vote
    POST /frame {"id", "t", "T": 16 floats, "jpeg"}  one picture, taken at cell time t from camera pose T (4x4, camera to
                                                     base, row-major), JPEG base64. In any order; used once it arrives
    GET  /grid?sure=2.2                              voxels with evidence for something (L >= sure): centres (m), colours,
                                                     their L; and counts
    GET  /columns?sure=2.2                           per 2 mm column: the top of what's there, the bottom of the known-empty
    GET  /stats

The Mac works out when each picture was taken and where the camera was (lab/pull.py); this only does depth.
Every time the camera has moved REF_MOVE_MM or turned REF_TURN_DEG since the last reference frame, the new frame becomes
one: each colour-edge pixel in it is tried along its ray at every height from the table up, against up to MAX_NBRS
frames already here whose cameras are BASE_MM apart from it (wide eyes); a pixel counts only with a clear, narrow peak
(the rules of lab/edges3d.py, the same numbers). Each point then votes in a 2 mm grid of log odds (lab/fog3d.py): HIT x
quality where it is, FREE x quality along its ray up to 3 sigma short of it. Quality = VOX / sigma, at most 1. No match
means no vote: blank table stays fog.
All on the GPU (torch): the edge search is ~70 heights x 6 neighbours x 15 samples for each of thousands of pixels.
"""
from __future__ import annotations

import base64
import collections
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # cell-isaac/: calibration
from calibration.geometry import SURFACE_Z  # noqa: E402

PORT = int(os.getenv("DEPTH_PORT", 8767))
DEV = torch.device("cuda")

# the edge search (lab/edges3d.py)
EDGE_STRIDE = 2
BASE_MM = (15, 120)
MAX_NBRS = 6
NBR_TURN_DEG = 15.0
NBR_WINDOW_S = 2.0
APERTURE_PX = 3.0
COARSE_MM = 2.0
FINE_MM = torch.arange(-2.0, 2.01, 0.5)
PEAK, MARGIN = 0.8, 0.1
MIN_SEEN = 4
SIGMA_MM = 2.0
H_RANGE_MM = (-3.0, 120.0)
REF_MOVE_MM, REF_TURN_DEG = 8.0, 3.0
ALONG = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
ACROSS = torch.tensor([-1.0, 0.0, 1.0])
_A, _C = (m.reshape(-1).to(DEV) for m in torch.meshgrid(ALONG, ACROSS, indexing="ij"))

# the grid (lab/fog3d.py)
VOX = 0.002
REGION = ((0.15, 0.70), (-0.40, 0.40), (-0.006, 0.15))  # x, y, height above the table (m)
HIT, FREE = 0.85, -0.4
SHAPE = tuple(int(round((hi - lo) / VOX)) for lo, hi in REGION)


class Depth:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset({"fx": 410.5, "k1": 0.0, "size": [848, 480]})

    def reset(self, lens: dict) -> dict:
        with self.lock:
            self.fx, self.k1 = float(lens["fx"]), float(lens["k1"])
            self.W, self.H = (int(v) for v in lens["size"])
            self.cx, self.cy = self.W / 2, self.H / 2
            self.inbox: collections.deque = collections.deque()
            self.ids, self.t, self.T = [], [], []  # frames kept, in arrival order
            self.gray: list[torch.Tensor] = []  # (H, W) float on the GPU
            self.colour: list[np.ndarray] = []
            self.refs: list[int] = []  # reference frames (indices), searched or waiting
            self.done: set[int] = set()
            self.L = torch.zeros(SHAPE, device=DEV)
            self.rgb = torch.zeros(SHAPE + (3,), device=DEV)
            self.nhit = torch.zeros(SHAPE, device=DEV)
            self.stats = collections.Counter()
            self.ms = collections.deque(maxlen=50)
        return {"ok": True, "grid": SHAPE}

    # ------------------------------------------------------------ frames in

    def add(self, body: dict) -> dict:
        with self.lock:
            self.inbox.append(body)
            return {"queued": len(self.inbox)}

    def _take_inbox(self) -> None:
        with self.lock:
            new, self.inbox = list(self.inbox), collections.deque()
        for b in new:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
            T = np.asarray(b["T"], float).reshape(4, 4)
            self.ids.append(b["id"])
            self.t.append(float(b["t"]))
            self.T.append(T)
            self.colour.append(img)
            self.gray.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).to(DEV).float())
            k = len(self.ids) - 1
            if not self.refs:
                self.refs.append(k)
            else:
                Tr = self.T[self.refs[-1]]
                move = np.linalg.norm(T[:3, 3] - Tr[:3, 3]) * 1000
                turn = np.degrees(np.arccos(np.clip((np.trace(Tr[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
                if move >= REF_MOVE_MM or turn >= REF_TURN_DEG:
                    self.refs.append(k)
            self.stats["frames"] += 1

    def neighbours(self, k: int) -> list[int]:
        c = np.array([T[:3, 3] for T in self.T])
        dist = np.linalg.norm(c - c[k], axis=1) * 1000
        z = np.array([T[:3, 2] for T in self.T])  # viewing directions: only views within NBR_TURN_DEG are compared
        turn = np.degrees(np.arccos(np.clip(z @ z[k], -1, 1)))  # (a tilted orbit view against a straight-down one
        cand = np.nonzero((dist >= BASE_MM[0]) & (dist <= BASE_MM[1]) & (turn <= NBR_TURN_DEG)  # made phantoms)
                          & (np.abs(np.array(self.t) - self.t[k]) <= NBR_WINDOW_S))[0]
        # (only pictures from the same stretch of motion: a picture-delay error then shifts a pass sideways as a whole
        # and leaves its heights right; pairing passes that went opposite ways turned the error into wrong heights)
        pick = []
        for i in cand[np.argsort(-dist[cand])]:
            if all(np.linalg.norm(c[i] - c[j]) * 1000 > 10 for j in pick):
                pick.append(int(i))
            if len(pick) >= MAX_NBRS:
                break
        return pick

    # ------------------------------------------------------------ the edge search, on the GPU

    def rays(self, T, uv):
        xd, yd = (uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fx
        x, y = xd, yd
        for _ in range(8):
            d = 1 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        R = torch.as_tensor(T[:3, :3], device=DEV, dtype=torch.float32)
        r = torch.stack([x, y, torch.ones_like(x)], -1) @ R.T
        return r / r.norm(dim=1, keepdim=True)

    def project(self, T, P):
        R = torch.as_tensor(T[:3, :3], device=DEV, dtype=torch.float32)
        o = torch.as_tensor(T[:3, 3], device=DEV, dtype=torch.float32)
        pc = (P - o) @ R
        x, y = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
        d = 1 + self.k1 * (x * x + y * y)
        return torch.stack([self.fx * x * d + self.cx, self.fx * y * d + self.cy], -1), pc[..., 2]

    def patches(self, img, uv, nrm):
        """Samples (..., 15) along the normal and across it at each uv (...,2), bilinear, edge-clamped."""
        tan = torch.stack([-nrm[..., 1], nrm[..., 0]], -1)
        xs = uv[..., 0:1] + nrm[..., 0:1] * _A + tan[..., 0:1] * _C
        ys = uv[..., 1:2] + nrm[..., 1:2] * _A + tan[..., 1:2] * _C
        g = torch.stack([xs / (self.W - 1) * 2 - 1, ys / (self.H - 1) * 2 - 1], -1)
        shp = g.shape
        out = F.grid_sample(img[None, None], g.reshape(1, -1, 15, 2), mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out.reshape(shp[:-1])

    @staticmethod
    def ncc(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / ((a * a).sum(-1) * (b * b).sum(-1)).sqrt().clamp_min(1e-6)

    def edge_points(self, k: int):
        nb = self.neighbours(k)
        if len(nb) < MIN_SEEN:
            return None
        col = self.colour[k]
        edges = np.zeros(col.shape[:2], np.uint8)
        for ch in cv2.split(cv2.cvtColor(col, cv2.COLOR_BGR2LAB)):
            edges |= cv2.Canny(ch, 10, 30)
        edges[:10] = edges[-10:] = 0
        edges[:, :10] = edges[:, -10:] = 0
        vs, us = np.nonzero(edges)
        vs, us = vs[::EDGE_STRIDE], us[::EDGE_STRIDE]
        if len(us) == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.uint8), {"edges": 0, "kept": 0, "nbrs": len(nb)}
        g = self.gray[k]
        gn = g.cpu().numpy()
        gx = cv2.Sobel(gn, cv2.CV_32F, 1, 0, ksize=3)[vs, us]
        gy = cv2.Sobel(gn, cv2.CV_32F, 0, 1, ksize=3)[vs, us]
        uv = torch.as_tensor(np.stack([us, vs], 1), device=DEV, dtype=torch.float32)
        nrm0 = -torch.as_tensor(np.stack([gx, gy], 1), device=DEV) / (torch.as_tensor(np.hypot(gx, gy), device=DEV)[:, None] + 1e-6)
        ray = self.rays(self.T[k], uv)
        o = torch.as_tensor(self.T[k][:3, 3], device=DEV, dtype=torch.float32)

        def at_height(h_mm):  # h_mm (N, H) -> points (N, H, 3)
            z = SURFACE_Z + h_mm / 1000
            return o + ray[:, None, :] * ((z - o[2]) / ray[:, None, 2:3].squeeze(-1))[..., None]

        # the aperture: an edge lying along the way its image moves with height looks the same at every height. Keep
        # only edge points whose image moves at least APERTURE_PX across the edge, in some neighbour, from 0 to 40 mm
        # up (the table's straight tape seam came out 15-20 mm high without this, real data run Oct 30)
        P01 = at_height(torch.tensor([[0.0, 40.0]], device=DEV).expand(len(uv), -1))
        across = torch.zeros(len(uv), device=DEV)
        for j in nb:
            q, _ = self.project(self.T[j], P01)
            across = torch.maximum(across, ((q[:, 1] - q[:, 0]) * nrm0).sum(-1).abs())
        keep = (across >= APERTURE_PX).cpu().numpy()
        if not keep.all():
            idx = np.nonzero(keep)[0]
            us, vs, gx, gy = us[idx], vs[idx], gx[idx], gy[idx]
            uv, nrm0, ray = uv[idx], nrm0[idx], ray[idx]
        if len(us) == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.uint8), {"edges": 0, "kept": 0, "nbrs": len(nb)}

        def score(ref_p, nrm, h_mm):
            P = at_height(h_mm)
            tot = torch.zeros(h_mm.shape, device=DEV)
            n = torch.zeros(h_mm.shape, device=DEV)
            for j in nb:
                q, depth = self.project(self.T[j], P)
                ok = (q[..., 0] > 6) & (q[..., 0] < self.W - 6) & (q[..., 1] > 6) & (q[..., 1] < self.H - 6) & (depth > 0.03)
                s = self.ncc(ref_p[:, None, :].expand(-1, h_mm.shape[1], -1),
                             self.patches(self.gray[j], q, nrm[:, None, :].expand(-1, h_mm.shape[1], -1)))
                tot += torch.where(ok, s, torch.zeros_like(s))
                n += ok.float()
            return torch.where(n >= MIN_SEEN, tot / n.clamp_min(1), torch.full_like(tot, -1.0)), n

        coarse_h = torch.arange(H_RANGE_MM[0], H_RANGE_MM[1] + 0.01, COARSE_MM, device=DEV)
        fine_d = FINE_MM.to(DEV)
        best = None
        for side in (1.0, -1.0):
            nrm = side * nrm0
            ref_p = self.patches(g, uv, nrm)
            coarse = torch.cat([score(ref_p, nrm, coarse_h[c:c + 16].expand(len(uv), -1))[0]
                                for c in range(0, len(coarse_h), 16)], 1)
            hc = coarse_h[coarse.argmax(1)]
            fine, seen_all = score(ref_p, nrm, hc[:, None] + fine_d[None, :])
            seen = seen_all[:, len(fine_d) // 2]
            i = fine.argmax(1)
            h = hc + fine_d[i]
            peak = fine.max(1).values
            far = (coarse_h[None, :] - h[:, None]).abs() > 4.0
            second = torch.where(far, coarse, torch.full_like(coarse, -1.0)).max(1).values
            width = (fine >= peak[:, None] - 0.03).sum(1).float() * (fine_d[1] - fine_d[0]) / 2
            cand = (h, peak, second, width, seen)
            if best is None:
                best = cand
            else:
                better = peak > best[1]
                best = tuple(torch.where(better, c, b) for c, b in zip(cand, best))
        h, peak, second, width, seen = best
        sig_h = width.clamp_min(0.5) / 1000
        sig = sig_h / ray[:, 2].abs().clamp_min(0.2)
        good = (peak >= PEAK) & (peak - second >= MARGIN) & (seen >= MIN_SEEN) & (sig_h * 1000 <= SIGMA_MM)
        good &= (h > H_RANGE_MM[0] + 1) & (h < H_RANGE_MM[1] - 1)
        P = at_height(h[:, None])[:, 0][good]
        gm = good.cpu().numpy()
        rgb = col[vs[gm], us[gm], ::-1]
        return P, sig[good], rgb, {"edges": len(us), "kept": int(gm.sum()), "nbrs": len(nb)}

    # ------------------------------------------------------------ votes

    def vote(self, o, P, sig, rgb):
        """HIT x quality at each point, FREE x quality along its ray from the camera to 3 sigma short of it (each voxel
        once per ray)."""
        if len(P) == 0:
            return
        lo = torch.tensor([REGION[0][0], REGION[1][0], SURFACE_Z + REGION[2][0]], device=DEV)
        shape = torch.tensor(SHAPE, device=DEV)
        w = (VOX / sig).clamp(max=1.0)
        d = P - o
        dist = d.norm(dim=1)
        u = d / dist[:, None]
        stop = (dist - 3 * sig).clamp_min(0)
        steps = torch.arange(0, float(stop.max()) + VOX, VOX / 2, device=DEV)
        for c in range(0, len(P), 4096):
            sl = slice(c, c + 4096)
            S = o + u[sl, None, :] * steps[None, :, None]  # (n, s, 3)
            ix = ((S - lo) / VOX).floor().long()
            ok = (steps[None, :] <= stop[sl, None]) & ((ix >= 0) & (ix < shape)).all(-1)
            lin = ix[..., 0] * SHAPE[1] * SHAPE[2] + ix[..., 1] * SHAPE[2] + ix[..., 2]
            first = torch.ones_like(ok)
            first[:, 1:] = lin[:, 1:] != lin[:, :-1]
            ok &= first
            wv = (FREE * w[sl, None]).expand_as(lin)
            self.L.view(-1).index_put_((lin[ok],), wv[ok], accumulate=True)
        ix = ((P - lo) / VOX).floor().long()
        ok = ((ix >= 0) & (ix < shape)).all(-1)
        lin = ix[ok, 0] * SHAPE[1] * SHAPE[2] + ix[ok, 1] * SHAPE[2] + ix[ok, 2]
        self.L.view(-1).index_put_((lin,), HIT * w[ok], accumulate=True)
        self.nhit.view(-1).index_put_((lin,), torch.ones_like(lin, dtype=torch.float32), accumulate=True)
        col = torch.as_tensor(np.ascontiguousarray(rgb), device=DEV, dtype=torch.float32)[ok]
        self.rgb.view(-1, 3).index_put_((lin,), col, accumulate=True)

    # ------------------------------------------------------------ the worker

    def step(self) -> bool:
        self._take_inbox()
        todo = [k for k in self.refs if k not in self.done]
        if not todo:
            return False
        for k in todo:  # oldest first; a reference without enough wide neighbours yet waits for more frames
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.edge_points(k)
            if out is None:
                if len(self.ids) - k > 60:  # it never will: give up on it
                    self.done.add(k)
                continue
            P, sig, rgb, st = out
            with torch.no_grad():
                self.vote(torch.as_tensor(self.T[k][:3, 3], device=DEV, dtype=torch.float32), P, sig, rgb)
            torch.cuda.synchronize()
            self.done.add(k)
            self.ms.append((time.perf_counter() - t0) * 1000)
            self.stats["refs"] += 1
            self.stats["edges"] += st["edges"]
            self.stats["points"] += st["kept"]
            return True
        return False

    def loop(self):
        while True:
            try:
                if not self.step():
                    time.sleep(0.02)
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)

    def grid(self, sure: float) -> dict:
        with torch.no_grad():
            m = self.L >= sure
            idx = m.nonzero()
            lo = np.array([REGION[0][0], REGION[1][0], SURFACE_Z + REGION[2][0]])
            xyz = idx.cpu().numpy() * VOX + lo + VOX / 2
            rgb = (self.rgb[m] / self.nhit[m].clamp_min(1)[:, None]).clamp(0, 255).byte().cpu().numpy()
            L = self.L[m].cpu().numpy()
            nothing = int((self.L <= -sure).sum())
        return {"xyz": np.round(xyz, 4).tolist(), "rgb": rgb.tolist(), "L": np.round(L, 2).tolist(),
                "something": len(xyz), "nothing": nothing, **self.stat()}

    def columns(self, sure: float) -> dict:
        """Per 2 mm column (x, y): the highest voxel with evidence for something (-1: none) and the lowest with
        evidence for nothing (-1: none), as voxel indices up from REGION's bottom."""
        with torch.no_grad():
            z = torch.arange(SHAPE[2], device=DEV)
            top = torch.where(self.L >= sure, z, torch.full_like(z, -1)).max(2).values
            low = torch.where(self.L <= -sure, z, torch.full_like(z, 10 ** 6)).min(2).values
            low = torch.where(low == 10 ** 6, torch.full_like(low, -1), low)
        return {"x0": REGION[0][0], "y0": REGION[1][0], "z0": SURFACE_Z + REGION[2][0], "vox": VOX,
                "shape": list(SHAPE[:2]), "top": top.cpu().numpy().tolist(), "free_low": low.cpu().numpy().tolist()}

    def stat(self) -> dict:
        return {"frames": self.stats["frames"], "refs": len(self.refs), "searched": self.stats["refs"],
                "waiting": len([k for k in self.refs if k not in self.done]), "points": self.stats["points"],
                "edges": self.stats["edges"], "ms_per_ref": round(float(np.median(self.ms)), 1) if self.ms else None,
                "inbox": len(self.inbox)}


def handler(depth: Depth):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method):
            u = urlparse(self.path)
            path = u.path.strip("/")
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if path == "reset":
                    return self._send(200, depth.reset(body))
                if path == "frame":
                    return self._send(200, depth.add(body))
                if path == "grid":
                    return self._send(200, depth.grid(float(q.get("sure", 2.2))))
                if path == "columns":
                    return self._send(200, depth.columns(float(q.get("sure", 2.2))))
                if path == "stats":
                    return self._send(200, depth.stat())
                self._send(404, {"error": f"no such endpoint: {method} /{path}"})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")
    return H


def main():
    depth = Depth()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler(depth))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[depth] ready on http://127.0.0.1:{PORT} (grid {SHAPE}, {torch.cuda.get_device_name()})", flush=True)
    depth.loop()


if __name__ == "__main__":
    main()
