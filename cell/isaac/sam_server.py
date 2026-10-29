"""SAM 3 on this box's GPU, for lab/sam.py: one picture and several words per request; the picture is encoded once.

    ~/cosmos-env/bin/python isaac/sam_server.py          (listens on 127.0.0.1:8766; the Mac reaches it over SSH)

POST /segment  {"image": base64 PNG/JPEG, "prompts": ["cube", "cup"], "threshold": 0.5}
  -> {"results": {"cube": [{"box": [x0, y0, x1, y1], "score": 0.9, "mask": base64 PNG (0/255, picture size)}]},
      "ms": {"picture": 105, "words": 290}}
GET /health -> {"ok": true}
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

PORT = int(os.getenv("SAM_PORT", 8766))
MODEL = os.getenv("SAM_MODEL", "facebook/sam3")
DEV = "cuda"

model = Sam3Model.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
proc = Sam3Processor.from_pretrained(MODEL)
lock = threading.Lock()  # one GPU job at a time


def segment(img: Image.Image, prompts: list[str], threshold: float) -> tuple[dict, dict]:
    with lock, torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        inp = proc(images=img, return_tensors="pt").to(DEV)
        emb = model.get_vision_features(pixel_values=inp.pixel_values.half())
        torch.cuda.synchronize(); t1 = time.time()
        out = {}
        for w in prompts:
            ti = proc(text=w, return_tensors="pt").to(DEV)
            r = model(vision_embeds=emb, input_ids=ti.input_ids, attention_mask=ti.attention_mask)
            res = proc.post_process_instance_segmentation(r, threshold=threshold, mask_threshold=0.5,
                                                          target_sizes=inp.get("original_sizes").tolist())[0]
            found = []
            for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
                ok, png = cv2.imencode(".png", m.cpu().numpy().astype(np.uint8) * 255)
                found.append({"box": [int(v) for v in b.tolist()], "score": round(float(s), 3),
                              "mask": base64.b64encode(png.tobytes()).decode()})
            out[w] = found
        torch.cuda.synchronize(); t2 = time.time()
    return out, {"picture": round((t1 - t0) * 1000), "words": round((t2 - t1) * 1000)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json(200, {"ok": True, "model": MODEL}) if self.path == "/health" else self._json(404, {"error": "no such path"})

    def do_POST(self):
        if self.path != "/segment":
            return self._json(404, {"error": "no such path"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            buf = np.frombuffer(base64.b64decode(body["image"]), np.uint8)
            img = Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            results, ms = segment(img, [str(p) for p in body["prompts"]], float(body.get("threshold", 0.5)))
            self._json(200, {"results": results, "ms": ms})
        except Exception as e:  # a bad request must not kill the server
            self._json(400, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    segment(Image.new("RGB", (640, 480)), ["cube"], 0.5)  # warm up: the first run is slow
    print(f"[sam] {MODEL} ready on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
