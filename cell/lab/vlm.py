"""The robot's semantic side: ask a vision-language model about what the cameras see.

Any OpenAI-compatible endpoint works (NVIDIA build.nvidia.com, OpenAI, a local server). Configure in cell/.env:
    VLM_BASE_URL=https://integrate.api.nvidia.com/v1
    VLM_API_KEY=nvapi-...
    VLM_MODEL=<a vision model id from the provider>
    VLM_NEMOTRON=nvidia/nemotron-3.5-super-vl-120b-a12b   # any VLM_<NAME>=<model id> is a choosable model
    VLM_COSMOS=nvidia/cosmos-reason2-8b
VLM_MODEL (if set) is the default, else the first VLM_<NAME>. Pick another with model="cosmos" (or --vlm cosmos).
With no key, everything here returns None and the rest of the lab runs without it.

The model only ever *describes and decides*: it never gives coordinates the arm moves to.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GIST = """You are the eyes of a small robot arm at a sorting table. The image shows the same table from several cameras
(labelled). Someone has put a pile of objects on the table and said: "{instruction}".

Glance at it the way a person would before touching anything, and answer in JSON only:
{{
  "pile": "one sentence: what this pile is",
  "objects": [{{"what": "...", "color": "...", "count": 0}}],
  "total_count": 0,
  "rule": "the sorting rule you'd use (follow the instruction if it gives one)",
  "why": "one sentence: why that rule fits these objects",
  "groups": ["group name", "..."],
  "shortcut": "if you recognise this kind of pile, what can we assume (e.g. 'dice: all cubes about 16 mm'), else null",
  "unsure": "what you can't tell from here and would want a closer look at"
}}"""


RESERVED = {"VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL"}


def models() -> dict[str, str]:
    """Named models from the environment: {"nemotron": "nvidia/...", ...}, plus "default"."""
    named = {k[4:].lower(): v for k, v in os.environ.items() if k.startswith("VLM_") and k not in RESERVED and v}
    default = os.getenv("VLM_MODEL") or next(iter(named.values()), None)
    return {**named, **({"default": default} if default else {})}


def model_id(name: str | None = None) -> str | None:
    m = models()
    if name is None:
        return m.get("default")
    return m.get(name.lower(), name if "/" in name else None)


def configured(name: str | None = None) -> bool:
    return bool(os.getenv("VLM_API_KEY") and model_id(name))


def tile(images: dict[str, np.ndarray], width: int = 1280) -> np.ndarray:
    """All camera images in one labelled picture (some endpoints take only one image per request)."""
    ims = []
    for name, im in images.items():
        h = int(im.shape[0] * (width / 2) / im.shape[1])
        im = cv2.resize(im, (width // 2, h))
        cv2.putText(im, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ims.append(im)
    if len(ims) % 2:
        ims.append(np.zeros_like(ims[-1]))
    rows = [np.hstack([ims[i], ims[i + 1]]) for i in range(0, len(ims), 2)]
    return np.vstack(rows)


def ask(images: dict[str, np.ndarray], prompt: str, max_tokens: int = 8000, model: str | None = None) -> dict | None:
    """Send the images + prompt. Returns {"text", "json" (parsed or None), "thinking" (reasoning models), "finish",
    "seconds", "model"}, or None if not configured."""
    if not configured(model):
        return None
    mid = model_id(model)
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("VLM_BASE_URL") or None, api_key=os.getenv("VLM_API_KEY"))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(tile(images), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=mid, max_tokens=max_tokens, temperature=0.2,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                               {"type": "image_url", "image_url": {"url": url}}]}])
    msg = resp.choices[0].message
    text = msg.content or ""
    thinking = (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or ""
    return {"text": text, "json": _parse(text), "thinking": thinking, "finish": resp.choices[0].finish_reason,
            "seconds": time.time() - t0, "model": mid}


def gist(images: dict[str, np.ndarray], instruction: str, model: str | None = None) -> dict | None:
    return ask(images, GIST.format(instruction=instruction or "sort these"), model=model)


def _parse(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models may think out loud first
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
