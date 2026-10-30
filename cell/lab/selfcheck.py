"""Does the depth agree with itself? Two depth maps from two halves of one real recording, no picture in both.

    uv run python -m lab.selfcheck runs/<time>_datarun_real [--delay-ms 140] [--port 8768]

A data run (lab/datarun.py) swept the wrist camera along y, then along x, then tilted. Half A is the passes along y,
half B the passes along x and the tilted ones. Each half goes through the depth service on its own (a second one, on
`port`, so a live run isn't disturbed), every picture posed by the joints `delay` before the cell's clock (first
appearance: lab/timing_real.py measured ~140 ms on that basis). Then, column by column of the grid: where both maps
say something stands above the table, how far apart their heights are; where only one does; and object by object
(lab/objects.py), the two tops. Two independent maps agreeing to a millimetre or two is depth to trust, without
knowing the truth. Writes <run>/selfcheck.json and selfcheck.jpg (A's heights | B's | where they differ).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose
from lab import objects as OB
from lab.depth_live import DepthLink
from lab.handling import column_heights
from lab.scene3d import cameras
from lab.timing_real import joints_at, load


def build(link, frames, jt, jq, T_fc, cam, delay, say=print):
    link.s.post(link.url + "reset", json={"fx": cam["fx"], "k1": cam["k1"], "size": list(cam["size"])},
                timeout=30).raise_for_status()
    for i, f in enumerate(frames):
        T = flange_pose(joints_at(jt, jq, f["t"] - delay)) @ T_fc
        link.send(f"f{i}", f["t"] - delay, T, f["path"].read_bytes())
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(2.0)
        st = link.stats()
        if st["inbox"] == 0 and st["waiting"] == 0 and link.q.empty():
            break
    say(f"  {st['frames']} pictures, {st['searched']} searched, {st['ms_per_ref']} ms each")
    return link.grid(), link.columns()


def heights_img(h, top_mm=60.0):
    v = np.clip(np.nan_to_num(h, nan=-1, neginf=-1) / top_mm, 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(h) | (h < OB.ABOVE_MM)] = (30, 30, 30)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--delay-ms", type=float, default=140.0)
    ap.add_argument("--port", type=int, default=8768)
    a = ap.parse_args()
    cam, T_fc, _ = cameras(None)
    jt, jq, frames = load(a.run)
    halves = {"A (passes along y)": [f for f in frames if f["phase"] == "line_y"],
              "B (along x + tilted)": [f for f in frames if f["phase"] != "line_y"]}
    link = DepthLink(port=a.port)
    maps = {}
    for name, fr in halves.items():
        print(f"{name}: {len(fr)} pictures")
        g, cols = build(link, fr, jt, jq, T_fc, cam, a.delay_ms / 1000)
        maps[name] = (g, cols, column_heights(cols))
    (nA, (gA, cA, hA)), (nB, (gB, cB, hB)) = maps.items()
    np.savez(a.run / "selfcheck_maps.npz", xyzA=np.asarray(gA["xyz"]).reshape(-1, 3), xyzB=np.asarray(gB["xyz"]).reshape(-1, 3),
             hA=hA, hB=hB)
    inA, inB = np.isfinite(hA) & (hA >= OB.ABOVE_MM), np.isfinite(hB) & (hB >= OB.ABOVE_MM)
    both = inA & inB
    with np.errstate(invalid="ignore"):
        d = np.abs(hA - hB)[both]
    res = {"delay_ms": a.delay_ms, "pictures": {k: len(v) for k, v in halves.items()},
           "columns_with_something": {"A": int(inA.sum()), "B": int(inB.sum()), "both": int(both.sum())},
           "agreement_iou": round(float(both.sum() / max(1, (inA | inB).sum())), 3),
           "height_diff_mm_where_both": {"median": round(float(np.median(d)), 2) if len(d) else None,
                                         "p90": round(float(np.percentile(d, 90)), 2) if len(d) else None,
                                         "within_2mm": round(float((d <= 2).mean()), 3) if len(d) else None}}
    def pooled(h, k=3):  # the highest thing in each k x k block of columns (6 mm): sparse edge maps rarely share cells
        X, Y = (h.shape[0] // k) * k, (h.shape[1] // k) * k
        v = np.where(np.isfinite(h), h, -np.inf)[:X, :Y].reshape(X // k, k, Y // k, k).max((1, 3))
        return v
    pA, pB = pooled(hA), pooled(hB)
    iA, iB = pA >= OB.ABOVE_MM, pB >= OB.ABOVE_MM
    pb = iA & iB
    dp = np.abs(pA - pB)[pb]
    res["pooled_6mm"] = {"iou": round(float(pb.sum() / max(1, (iA | iB).sum())), 3),
                         "height_diff_median_mm": round(float(np.median(dp)), 2) if len(dp) else None,
                         "within_3mm": round(float((dp <= 3).mean()), 3) if len(dp) else None}
    oA = OB.find(cA, gA["xyz"], gA["rgb"])
    oB = OB.find(cB, gB["xyz"], gB["rgb"])
    pairs = []
    for o in oA:
        if not oB:
            break
        dist = [np.linalg.norm(o.centre - p.centre) * 1000 for p in oB]
        j = int(np.argmin(dist))
        if dist[j] <= 15:
            pairs.append({"A": o.id, "B": oB[j].id, "centre_gap_mm": round(dist[j], 1), "top_A": o.top_mm,
                          "top_B": oB[j].top_mm, "top_diff_mm": round(o.top_mm - oB[j].top_mm, 1),
                          "container_A": o.container, "container_B": oB[j].container})
    res["objects"] = {"A": len(oA), "B": len(oB), "matched": len(pairs), "pairs": pairs,
                      "top_diff_median_mm": round(float(np.median([abs(p["top_diff_mm"]) for p in pairs])), 1) if pairs else None}
    (a.run / "selfcheck.json").write_text(json.dumps(res, indent=1, default=float))
    diff = np.full(hA.shape, np.nan)
    diff[both] = np.abs(hA - hB)[both]
    dimg = cv2.applyColorMap((np.clip(np.nan_to_num(diff, nan=0) / 10, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    dimg[~both] = (30, 30, 30)
    dimg[inA ^ inB] = (200, 200, 200)  # only one map says something's there
    tiles = []
    for img, label in ((heights_img(hA), "A: heights (0-60 mm)"), (heights_img(hB), "B: heights"),
                       (dimg, "|A-B| 0-10 mm (grey: only one)")):
        img = cv2.resize(np.rot90(img), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    cv2.imwrite(str(a.run / "selfcheck.jpg"), np.hstack(tiles))
    print(json.dumps({k: v for k, v in res.items() if k != "objects"}, indent=1))
    print("objects:", {k: v for k, v in res["objects"].items() if k != "pairs"})


if __name__ == "__main__":
    main()
