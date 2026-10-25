"""Run MapAnything on a capture (on the GPU box, in ~/mapany/venv):  python infer.py sim.npz

Tries each combination of inputs and saves the 3D points per frame (in the base frame) to <name>_<mode>.npz:
    images          images + lens only: structure from motion, scale guessed by the model
    images+poses    images + lens + the arm's camera poses (the wide baseline comes from the arm moving)
    images+poses+depth   ... + the noisy depth (sim only: the real recordings have no depth)
"""
import os
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

path = sys.argv[1]
d = np.load(path)
n = len(d["img"])
dev = "cuda"
model = MapAnything.from_pretrained("facebook/map-anything").to(dev)
modes = ["images", "images+poses"] + (["images+poses+depth"] if "depth" in d else [])
for mode in modes:
    views = []
    for i in range(n):
        v = {"img": torch.from_numpy(d["img"][i]), "intrinsics": torch.from_numpy(d["K"][i]).float()}
        if "poses" in mode:
            v["camera_poses"] = torch.from_numpy(d["pose"][i]).float()
            v["is_metric_scale"] = torch.tensor([True])
        if "depth" in mode:
            v["depth_z"] = torch.from_numpy(d["depth"][i]).float()
        views.append(v)
    t0 = time.time()
    pred = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                       apply_mask=True, mask_edges=True)
    secs = time.time() - t0
    pts = np.stack([p["pts3d"][0].float().cpu().numpy() for p in pred])
    mask = np.stack([p["mask"][0, ..., 0].cpu().numpy() for p in pred]).astype(bool)
    conf = np.stack([p["conf"][0].float().cpu().numpy() for p in pred])
    poses = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in pred])
    img = np.stack([p["img_no_norm"][0].float().cpu().numpy() for p in pred])
    out = path.replace(".npz", f"_{mode.replace('+', '_')}.npz")
    np.savez_compressed(out, pts=pts, mask=mask, conf=conf, poses=poses, img=img, secs=secs)
    print(f"{mode:22s} {secs:5.2f} s  {pts.shape}  -> {out}", flush=True)
