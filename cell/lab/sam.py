"""SAM: a picture and a few words ("cube", "bolt") -> one mask per object it finds.

Two backends, the same results:
    local (default)  SAM 3 on the 4090 box (isaac/sam_server.py), over an SSH tunnel like the Isaac cell's: ~0.4 s for
                     a picture and six words, the frames stay on our machine. SAM_BACKEND=meta to skip it.
    meta             SAM 3.1 through Meta's Model API: POST https://api.meta.ai/v1/responses, Bearer SAM3_API_KEY
                     (cell/.env), one word per call (~1.5 s each); the masks are decoded by lab.samcodec. Also the
                     fallback when the box can't be reached.

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
LOCAL_PORT = int(os.getenv("SAM_PORT", 8766))
_local_down = False  # the box didn't answer once: Meta for the rest of the session (said once)


def configured() -> bool:
    return bool(os.getenv("SAM3_API_KEY")) or os.getenv("SAM_BACKEND", "local") == "local"


def _tunnel() -> None:
    """localhost:LOCAL_PORT -> the box's localhost:LOCAL_PORT (an ssh -L in the background, if it isn't up)."""
    import socket
    import subprocess

    def up() -> bool:
        with socket.socket() as so:
            so.settimeout(0.5)
            return so.connect_ex(("127.0.0.1", LOCAL_PORT)) == 0
    if up():
        return
    host = os.getenv("ISAAC_SSH", "joel@shepherd-traffic-base-server")
    subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                      "-L", f"{LOCAL_PORT}:127.0.0.1:{LOCAL_PORT}", host], stdin=subprocess.DEVNULL)
    for _ in range(50):
        if up():
            return
        time.sleep(0.2)
    raise ConnectionError(f"no tunnel to the SAM server on {host} (isaac/sam_server.py there)")


def _local(part: np.ndarray, prompts: list[str], timeout: float) -> dict[str, list[dict]]:
    """The local server: {prompt: [{"box" (part pixels), "mask" (bool, part size)}]}."""
    _tunnel()
    ok, buf = cv2.imencode(".png", cv2.cvtColor(part, cv2.COLOR_RGB2BGR))
    r = requests.post(f"http://127.0.0.1:{LOCAL_PORT}/segment", timeout=timeout,
                      json={"image": base64.b64encode(buf.tobytes()).decode(), "prompts": list(prompts)})
    r.raise_for_status()
    out = {}
    for w, found in r.json()["results"].items():
        out[w] = [{"box": d["box"], "mask": cv2.imdecode(np.frombuffer(base64.b64decode(d["mask"]), np.uint8),
                                                          cv2.IMREAD_UNCHANGED) > 0} for d in found]
    return out


def segment_many(image: np.ndarray, prompts: list[str], crop: tuple[int, int, int, int] | None = None,
                 scale: float = 2.0, timeout: float = 40.0) -> list[dict]:
    """segment() for several words at once: on the local server the picture is sent and encoded once."""
    global _local_down
    if os.getenv("SAM_BACKEND", "local") == "local" and not _local_down:
        H, W = image.shape[:2]
        x0, y0, x1, y1 = crop or (0, 0, W, H)
        part = image[y0:y1, x0:x1]
        if scale != 1:
            part = cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        t0 = time.time()
        try:
            found = _local(part, prompts, timeout)
        except (requests.RequestException, ConnectionError, OSError) as e:
            _local_down = True
            print(f"(local SAM unreachable: {e}; using Meta's API from now on)", flush=True)
        else:
            out = []
            for w in prompts:
                for d in found.get(w, []):
                    m = d["mask"]
                    if scale != 1:
                        m = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
                    full = np.zeros((H, W), bool)
                    full[y0:y1, x0:x1] = m
                    bx = np.array(d["box"], float) / scale + [x0, y0, x0, y0]
                    out.append({"box": tuple(int(v) for v in bx), "mask": full, "area": int(full.sum()), "prompt": w,
                                "seconds": round(time.time() - t0, 2)})
            return out
    return [d for w in prompts for d in _meta(image, w, crop, scale, timeout)]


def segment(image: np.ndarray, prompt: str, crop: tuple[int, int, int, int] | None = None, scale: float = 2.0,
            timeout: float = 40.0) -> list[dict]:
    """image: RGB. crop: (x0, y0, x1, y1) of the part to send. Returns [{"box", "mask" (bool, image size), "area",
    "prompt"}], masks placed back in the whole image."""
    return segment_many(image, [prompt], crop, scale, timeout)


def _meta(image: np.ndarray, prompt: str, crop, scale: float, timeout: float) -> list[dict]:
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
