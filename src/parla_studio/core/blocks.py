"""Block processing for long audio: real progress, time left and abort between blocks.

Models run on overlapping blocks joined with a crossfade, so an hours-long inference reports its progress
(and can be aborted within seconds) instead of being one opaque call.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def fmt_dur(s: float) -> str:
    s = max(0, int(round(s)))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02}:{sec:02}" if h else f"{m:02}:{sec:02}"


class Eta:
    """Progress messages with an estimated time left, without flooding the log."""

    def __init__(self, label: str, total_s: float, log=None, every_s: float = 5.0) -> None:
        self.label, self.total_s, self.log, self.every_s = label, total_s, log, every_s
        self.t0 = time.perf_counter()
        self._last = -1e9

    def message(self, frac: float) -> str:
        el = time.perf_counter() - self.t0
        rest = el / frac * (1 - frac) if frac > 0.01 else None
        msg = f"{self.label}: {frac * 100:.0f}% · {fmt_dur(frac * self.total_s)} of {fmt_dur(self.total_s)} of audio"
        return msg + (f" · ~{fmt_dur(rest)} left" if rest is not None else "")

    def tick(self, frac: float, force: bool = False) -> None:
        now = time.perf_counter()
        if self.log and (force or now - self._last >= self.every_s):
            self._last = now
            self.log(self.message(frac), 0)

    def done(self) -> None:
        if self.log:
            self.log(f"{self.label}: done in {fmt_dur(time.perf_counter() - self.t0)} ({self.total_s / max(1e-6, time.perf_counter() - self.t0):.0f}x real time)", 0)


def _oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def process_blocks(x: np.ndarray, sr: int, fn: Callable[[np.ndarray, Callable[[float], None]], np.ndarray],
                   block_s: float, overlap_s: float, progress: Callable[[float], None] | None = None,
                   check: Callable[[], None] | None = None, log=None, min_block_s: float = 8.0) -> np.ndarray:
    """Apply `fn(block, inner_progress)` to overlapping blocks of x [frames, channels] and join the result with a
    linear crossfade on the overlaps. `fn` returns a block of the same shape. `check()` is called between
    blocks (and may raise Cancelled); `progress(frac)` receives the overall progress 0..1."""
    n = x.shape[0]
    block = max(1, int(block_s * sr))
    ov = min(int(overlap_s * sr), block // 2)
    step = block - ov
    starts = list(range(0, max(n - ov, 1), step)) if n > block else [0]
    out = np.zeros_like(x, dtype=np.float32)
    for k, a in enumerate(starts):
        if check:
            check()
        b = min(n, a + block)

        def inner(f: float, k=k) -> None:
            if progress:
                progress((k + min(max(f, 0.0), 1.0)) / len(starts))

        try:
            y = fn(x[a:b], inner).astype(np.float32, copy=False)
        except Exception as e:  # noqa: BLE001
            if not _oom(e) or block_s / 2 < min_block_s:
                raise
            # did not fit in VRAM: this block is processed in halves (and so on)
            from . import models
            models.free_vram()
            if log:
                log(f"Not enough VRAM for a {block_s:.0f} s block: processing in {block_s / 2:.0f} s parts", 1)
            y = process_blocks(x[a:b], sr, fn, block_s / 2, overlap_s, inner, check, log, min_block_s)
        if y.shape[0] != b - a:  # some models return a few samples more or less
            y = y[: b - a] if y.shape[0] > b - a else np.pad(y, ((0, b - a - y.shape[0]),) + ((0, 0),) * (y.ndim - 1))
        if k == 0:
            out[a:b] = y
        else:
            m = min(ov, b - a)
            w = np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
            out[a:a + m] = out[a:a + m] * (1 - w) + y[:m] * w
            out[a + m:b] = y[m:]
        if progress:
            progress((k + 1) / len(starts))
    return out
