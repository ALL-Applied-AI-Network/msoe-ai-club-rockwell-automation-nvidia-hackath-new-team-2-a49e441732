"""Decode SAM 3.1 masks from Meta's Model API ("mask_encoding": "one_bit").

Each detection in the reply text is `<|box;x1=;y1=;x2=;y2=;w=;h=|><|mask;x=;y=;data=H,W,<payload>|>` (box corners inclusive;
w, h = the image's size). The payload is an H x W bitmap (row-major; it covers the box, offset by x, y), compressed with an adaptive binary range coder whose
context is the 12 already-decoded neighbours, then written as base85 text with a 5-digit length prefix and a leading
"!". Ported from Meta's @meta-sam/parser 0.1.0 (dist/mask-codec.js, decodeRaster).
"""
from __future__ import annotations

import re

import numpy as np

_EXCLUDED = set('"\\,;<>|')
ALPHABET = "".join(chr(c) for c in range(0x21, 0x7F) if chr(c) not in _EXCLUDED)
_VAL = {ch: i for i, ch in enumerate(ALPHABET)}
R, PREFIX = 85, 5
M32, TOP, BOTTOM = 0xFFFFFFFF, 1 << 24, 1 << 16
INCREMENT, COUNT_LIMIT, CONTEXTS, PAD = 14, 4096, 1 << 12, 2


def _unpack(s: str) -> bytes:
    n = 0
    for ch in s[:PREFIX]:
        n = n * R + _VAL[ch]
    out = bytearray()
    src, rem = PREFIX, n % 4
    for _ in range(n // 4):
        v = 0
        for ch in s[src:src + 5]:
            v = v * R + _VAL[ch]
        out += v.to_bytes(4, "big")
        src += 5
    if rem:
        v = 0
        for i in range(5):
            v = v * R + (_VAL[s[src + i]] if i < rem + 1 else R - 1)
        out += v.to_bytes(4, "big")[:rem]
    return bytes(out)


class _Decoder:
    def __init__(self, data: bytes):
        self.d, self.p = data, 0
        self.low, self.range, self.code, self.scaled = 0, M32, 0, 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._read()) & M32

    def _read(self) -> int:
        b = self.d[self.p]
        self.p += 1
        return b

    def freq(self, total: int) -> int:
        self.scaled = self.range // total
        v = ((self.code - self.low) & M32) // self.scaled
        return min(v, total - 1)

    def decode(self, cum: int, f: int) -> None:
        self.low = (self.low + self.scaled * cum) & M32
        self.range = self.scaled * f
        while True:
            if ((self.low ^ (self.low + self.range)) & M32) < TOP:
                pass
            elif self.range < BOTTOM:
                self.range = (-self.low & M32) & (BOTTOM - 1)
            else:
                break
            self.code = ((self.code << 8) | self._read()) & M32
            self.low = (self.low << 8) & M32
            self.range = (self.range << 8) & M32


def decode(payload: str, w: int, h: int) -> np.ndarray:
    """The H x W bitmap (uint8 0/1) of one one_bit payload."""
    if not payload.startswith("!"):
        raise ValueError("not a one_bit payload")
    dec = _Decoder(_unpack(payload[1:]))
    counts = np.ones(CONTEXTS * 2, np.int64)
    pw = w + 2 * PAD
    pad = [0] * (pw * (h + PAD))  # python ints: the context shifts go up to 11 bits
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        row = (y + PAD) * pw + PAD
        for x in range(w):
            p = row + x
            q, e = p - pw, p - 2 * pw
            ctx = (pad[p - 1] | pad[p - 2] << 1 | pad[q - 2] << 2 | pad[q - 1] << 3 | pad[q] << 4 | pad[q + 1] << 5
                   | pad[q + 2] << 6 | pad[e - 2] << 7 | pad[e - 1] << 8 | pad[e] << 9 | pad[e + 1] << 10 | pad[e + 2] << 11)
            o = int(ctx) << 1
            z, one = int(counts[o]), int(counts[o + 1])
            bit = 0 if dec.freq(z + one) < z else 1
            dec.decode(0, z) if bit == 0 else dec.decode(z, one)
            pad[p] = bit
            out[y, x] = bit
            counts[o + bit] += INCREMENT
            if counts[o] + counts[o + 1] >= COUNT_LIMIT:
                counts[o] = max(counts[o] >> 1, 1)
                counts[o + 1] = max(counts[o + 1] >> 1, 1)
    return out


DETECTION = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
                       r"<\|mask;x=(\d+);y=(\d+);data=(\d+),(\d+),([^|]*)\|>")


def parse(text: str) -> list[dict]:
    """Every detection in a reply: its box and its mask placed in the full image (bool, image size)."""
    out = []
    for m in DETECTION.finditer(text):
        x1, y1, x2, y2, W, H, mx, my, h, w = map(int, m.groups()[:10])  # data= is height, then width
        bm = decode(m.group(11), w, h)
        full = np.zeros((H, W), bool)
        ox, oy = x1 + mx, y1 + my
        hh, ww = min(h, H - oy), min(w, W - ox)
        full[oy:oy + hh, ox:ox + ww] = bm[:hh, :ww].astype(bool)
        out.append({"box": (x1, y1, x2, y2), "mask": full, "area": int(bm.sum())})
    return out
