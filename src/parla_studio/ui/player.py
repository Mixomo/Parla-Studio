"""Player with instant switching between sources (A/B/voice/background) and a real-time level meter.

The audio callback only copies samples already in memory; the meter measures exactly what is playing
(RMS 300 ms, momentary LUFS 400 ms, short-term LUFS 3 s and peak with decay), so it falls on silence,
pause or end.
"""
from __future__ import annotations

import collections
import math
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, lfilter_zi

from ..core.leveler import kweight_coeffs


class Player:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sources: dict[str, np.ndarray] = {}
        self.active = ""
        self.sr = 48000
        self.pos = 0
        self.playing = False
        self.stop_at: int | None = None  # stops by itself on reaching this frame (segment preview)
        self.stream: sd.OutputStream | None = None
        self._meter_reset()

    # ------------------------------------------------------------ sources
    def set_sources(self, sr: int, sources: dict[str, np.ndarray], active: str | None = None) -> None:
        """Replace the sources (same length and sample rate). Reopens the stream if the sample rate changes."""
        with self._lock:
            self.sources = {k: (v if v.ndim == 2 else v[:, None]).astype(np.float32, copy=False) for k, v in sources.items()}
            if active in self.sources:
                self.active = active
            elif self.active not in self.sources:
                self.active = next(iter(self.sources), "")
            self.pos = min(self.pos, self.length)
        if self.stream is None or self.sr != sr:
            self._open(sr)

    def add_source(self, name: str, data: np.ndarray) -> None:
        with self._lock:
            self.sources[name] = (data if data.ndim == 2 else data[:, None]).astype(np.float32, copy=False)

    def remove_source(self, name: str) -> None:
        with self._lock:
            self.sources.pop(name, None)
            if self.active == name:
                self.active = next(iter(self.sources), "")

    def set_active(self, name: str) -> None:
        with self._lock:
            if name in self.sources:
                self.active = name

    @property
    def length(self) -> int:
        src = self.sources.get(self.active)
        return 0 if src is None else src.shape[0]

    # ------------------------------------------------------------ transporte
    def play(self) -> None:
        with self._lock:
            if self.pos >= self.length:
                self.pos = 0
            self.playing = bool(self.sources)

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            self.pos = 0

    def seek(self, frame: int) -> None:
        with self._lock:
            self.pos = int(np.clip(frame, 0, self.length))

    def position(self) -> int:
        latency = 0
        if self.stream is not None and self.playing:
            latency = int(self.stream.latency * self.sr) if np.isscalar(self.stream.latency) else 0
        return max(0, self.pos - latency)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def device_name(self) -> str:
        try:
            d = sd.query_devices(sd.default.device[1])
            return f"{d['name']} · {self.sr} Hz"
        except Exception:
            return "no device"

    # ------------------------------------------------------------ interno
    def _open(self, sr: int) -> None:
        self.close()
        self.sr = sr
        self._meter_reset()
        self.stream = sd.OutputStream(samplerate=sr, channels=2, dtype="float32", callback=self._callback,
                                      blocksize=0, latency="low")
        self.stream.start()

    def _callback(self, out: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            src = self.sources.get(self.active)
            if not self.playing or src is None:
                out.fill(0.0)
                block = None
            else:
                a = self.pos
                end = src.shape[0] if self.stop_at is None else min(src.shape[0], self.stop_at)
                b = max(a, min(a + frames, end))
                block = src[a:b]
                n = b - a
                if block.shape[1] == 1:
                    out[:n] = block
                else:
                    out[:n] = block[:, :2]
                out[n:] = 0.0
                self.pos = b
                if b >= end:
                    self.playing = False
                    self.stop_at = None
        self._meter_feed(block if block is not None and block.size else np.zeros((frames, 1), np.float32), frames)

    # ------------------------------------------------------------ level meter
    def _meter_reset(self) -> None:
        (b1, a1), (b2, a2) = kweight_coeffs(self.sr)
        self._kf = (b1, a1, b2, a2)
        self._zi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._hist: collections.deque = collections.deque()  # (frames, raw_sum, k_sum)
        self._hist_frames = 0
        self.peak_hold = 0.0
        self.meter = {"rms": -200.0, "m": -200.0, "s": -200.0, "peak": -200.0}

    def _meter_feed(self, block: np.ndarray, frames: int) -> None:
        if block.shape[0] < frames:
            block = np.vstack([block, np.zeros((frames - block.shape[0], block.shape[1]), np.float32)])
        b1, a1, b2, a2 = self._kf
        ch = block.shape[1]
        raw = float((block.astype(np.float64) ** 2).sum()) / ch
        ks = 0.0
        for c in range(ch):
            z = self._zi.get(c)
            if z is None:
                z = (lfilter_zi(b1, a1) * 0.0, lfilter_zi(b2, a2) * 0.0)
            y, z1 = lfilter(b1, a1, block[:, c].astype(np.float64), zi=z[0])
            w, z2 = lfilter(b2, a2, y, zi=z[1])
            self._zi[c] = (z1, z2)
            ks += float((w * w).sum())
        self._hist.append((frames, raw, ks))
        self._hist_frames += frames
        while self._hist and self._hist_frames - self._hist[0][0] >= 3 * self.sr:
            self._hist_frames -= self._hist.popleft()[0]

        def window(seconds: float, idx: int) -> float:
            need, n, s = seconds * self.sr, 0, 0.0
            for item in reversed(self._hist):
                n += item[0]
                s += item[idx]
                if n >= need:
                    break
            return s / max(need, 1)

        def db(e: float, off: float = 0.0) -> float:
            return off + 10 * math.log10(e) if e > 1e-20 else -200.0

        pk = float(np.abs(block).max()) if block.size else 0.0
        self.peak_hold = max(pk, self.peak_hold * 10 ** (-20.0 * frames / self.sr / 20.0))
        self.meter = {
            "rms": db(window(0.3, 1)),
            "m": db(window(0.4, 2), -0.691),
            "s": db(window(3.0, 2), -0.691),
            "peak": 20 * math.log10(self.peak_hold) if self.peak_hold > 1e-10 else -200.0,
        }
