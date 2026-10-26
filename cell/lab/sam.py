"""SAM 3.1 through Meta's Model API: a picture and a few words ("cube", "bolt") -> one mask per object it finds.

    POST https://api.meta.ai/v1/responses, Bearer SAM3_API_KEY (cell/.env); the masks are decoded by lab.samcodec.

Only the table area is sent (cropped from the table camera's picture, scaled up): a 10 mm part is a few pixels in the
whole frame. The key goes to api.meta.ai and nowhere else. Simulator pictures only, unless SAM_REAL=1 (sending the
lab's camera images to Meta is the team's call).
"""
from __future__ import annotations

import base64
import os
import time

import cv2
import numpy as np
import requests

from lab import samcodec, vlm  # noqa: F401  (vlm loads cell/.env)

URL = "https://api.meta.ai/v1/responses"


def configured() -> bool:
    return bool(os.getenv("SAM3_API_KEY"))


def segment(image: np.ndarray, prompt: str, crop: tuple[int, int, int, int] | None = None, scale: float = 2.0,
            timeout: float = 40.0) -> list[dict]:
    """image: RGB. crop: (x0, y0, x1, y1) of the part to send. Returns [{"box", "mask" (bool, image size), "area",
    "prompt"}], masks placed back in the whole image."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = crop or (0, 0, W, H)
    part = image[y0:y1, x0:x1]
    if scale != 1:
        part = cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(part, cv2.COLOR_RGB2BGR))
    body = {"model": "sam-3.1", "metadata": {"mask_encoding": "one_bit"},
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()}]}]}
    t0 = time.time()
    r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {os.environ['SAM3_API_KEY']}"}, timeout=timeout)
    r.raise_for_status()
    text = "".join(c.get("text", "") for o in r.json().get("output", []) for c in o.get("content", []) or [])
    out = []
    for d in samcodec.parse(text):
        m = d["mask"]
        if scale != 1:
            m = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((H, W), bool)
        full[y0:y1, x0:x1] = m
        bx = np.array(d["box"], float) / scale + [x0, y0, x0, y0]
        out.append({"box": tuple(int(v) for v in bx), "mask": full, "area": int(full.sum()), "prompt": prompt})
    for d in out:
        d["seconds"] = round(time.time() - t0, 1)
    return out
