"""Classic leveler: vectorized (numpy) port of the algorithm validated in Python/Rust/C++.

Stage 1: pull loud and quiet parts toward a reference (smooth attack/release).
Stage 2: measure (RMS or LUFS) and apply a single offset.
Stage 3: 1 ms peak limiter (one-block lookahead, linear release in dB).

It also exposes the pieces (curves, limiter, measurement) reused by the voice-aware mode.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import lfilter

HOP_S = 0.005
SAFETY_DB = 60.0
REF_PCT = 50.0
LIM_S = 0.001
LIM_CEIL_DB = -0.3
LIM_REL_S = 0.08
KNEE = np.float32(0.89)
CHUNK = 1 << 20  # frames per block when rendering (bounded memory)


class Cancelled(Exception):
    """Process cancelled by the user."""


@dataclass
class Params:
    window_s: float = 0.25
    attack_s: float = 0.05
    release_s: float = 0.25
    lufs: bool = False
    final_db: float = -18.0


@dataclass
class GainCurve:
    """Gain in dB, one value per `hop` frames (sample i falls at k = i/hop - 0.5)."""
    db: np.ndarray
    hop: int

    def at(self, idx: np.ndarray) -> np.ndarray:
        if self.db.size == 0:
            return np.zeros(len(idx))
        xp = (np.arange(self.db.size) + 0.5) * self.hop
        return np.interp(np.asarray(idx, dtype=np.float64), xp, self.db.astype(np.float64))


@dataclass
class Ctx:
    cancel: threading.Event = field(default_factory=threading.Event)
    progress: Callable[[float], None] = lambda v: None
    log: Callable[[str, int], None] = lambda m, lvl: None

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def sub(self, p0: float, p1: float) -> "Ctx":
        """Same abort and logging, with a stage's progress mapped to the [p0, p1] range of the bar."""
        return Ctx(self.cancel, lambda v: self.progress(p0 + (p1 - p0) * v), self.log)


@dataclass
class Result:
    gain: GainCurve
    level_db: float
    offset_db: float
    limited_blocks: int
    max_reduction_db: float
    output: np.ndarray


def hop_frames(sr: int, seconds: float) -> int:
    return max(1, int(round(sr * seconds)))


# ---------------------------------------------------------------- measurement

def hop_meansq(x: np.ndarray, sr: int) -> np.ndarray:
    """Mean energy per 5 ms hop (channel average); complete hops only."""
    hop = hop_frames(sr, HOP_S)
    n = x.shape[0] // hop
    out = np.empty(n, dtype=np.float64)
    step = max(1, CHUNK // hop)
    for a in range(0, n, step):
        b = min(n, a + step)
        seg = x[a * hop:b * hop].astype(np.float64)
        out[a:b] = (seg * seg).mean(axis=1).reshape(b - a, hop).mean(axis=1)
    return out


def loudness_db(e: np.ndarray, win_s: float) -> np.ndarray:
    """Centered short-term loudness (moving average with zeros outside), in dB."""
    n = e.size
    w = max(1, int(round(win_s / HOP_S)))
    p = np.concatenate(([0.0], np.cumsum(e)))
    i = np.arange(n)
    a = np.maximum(i - w // 2, 0)
    b = np.minimum(i - w // 2 + w, n)
    s = np.where(b > a, p[b] - p[a], 0.0)
    return 10.0 * np.log10(s / w + 1e-12)


def smooth_gain(desired: np.ndarray, attack_s: float, release_s: float) -> np.ndarray:
    """Follower with attack (when lowering gain) and release (when raising it); float32 output."""
    a = math.exp(-HOP_S / max(attack_s, 1e-4))
    r = math.exp(-HOP_S / max(release_s, 1e-4))
    out = np.empty(desired.size, dtype=np.float32)
    cur = 0.0
    ia, ir = 1.0 - a, 1.0 - r
    for i, d in enumerate(desired.tolist()):  # recursive: Python lists are the fastest option without a JIT
        cur = a * cur + ia * d if d < cur else r * cur + ir * d
        out[i] = cur
    return out


def kweight_coeffs(sr: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    a = 10.0 ** (g / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    c, s = math.cos(w0), math.sin(w0)
    al = s / (2.0 * q)
    sa = math.sqrt(a)
    b0 = a * ((a + 1) + (a - 1) * c + 2 * sa * al)
    b1 = -2 * a * ((a - 1) + (a + 1) * c)
    b2 = a * ((a + 1) + (a - 1) * c - 2 * sa * al)
    a0 = (a + 1) - (a - 1) * c + 2 * sa * al
    a1 = 2 * ((a - 1) - (a + 1) * c)
    a2 = (a + 1) - (a - 1) * c - 2 * sa * al
    s1 = (np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))
    f0, q = 38.13547087602444, 0.5003270373238773
    w0 = 2.0 * math.pi * f0 / sr
    c, s = math.cos(w0), math.sin(w0)
    al = s / (2.0 * q)
    ha0 = 1 + al
    s2 = (np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2]) / ha0, np.array([1.0, -2 * c / ha0, (1 - al) / ha0]))
    return s1, s2


def k_subblocks(x: np.ndarray, sr: int) -> np.ndarray:
    """K-weighted energy per 100 ms sub-block (sum of channels); complete blocks only."""
    (b1, a1), (b2, a2) = kweight_coeffs(sr)
    h = max(1, int(round(0.1 * sr)))
    acc = np.zeros(x.shape[0] // h, dtype=np.float64)
    for k in range(x.shape[1]):
        w = lfilter(b2, a2, lfilter(b1, a1, x[:, k].astype(np.float64)))
        acc += (w[: acc.size * h] ** 2).reshape(acc.size, h).mean(axis=1)
    if acc.size == 0:
        raise ValueError("audio too short to measure LUFS")
    return acc


def lufs_integrated(s: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Integrated LUFS (-70 absolute and -10 relative gating). `mask` (per 400 ms block) limits the
    measurement to those blocks (e.g. voice only: dialogue loudness)."""
    if s.size < 4:
        raise ValueError("audio too short to measure LUFS (minimum 0.4 s)")
    z = (s[:-3] + s[1:-2] + s[2:-1] + s[3:]) / 4.0
    l = -0.691 + 10.0 * np.log10(z + 1e-20)
    gate = l > -70.0
    if mask is not None:
        gate &= mask[: z.size]
    if not gate.any():
        raise ValueError("silence: there is no level to measure")
    rel = -0.691 + 10.0 * math.log10(z[gate].mean()) - 10.0
    gate &= l > rel
    if not gate.any():
        raise ValueError("silence: there is no level to measure")
    return float(-0.691 + 10.0 * math.log10(z[gate].mean()))


def rms_db(e: np.ndarray) -> float:
    if e.size == 0:
        raise ValueError("no samples to measure")
    return float(10.0 * math.log10(e.mean() + 1e-20))


# ---------------------------------------------------------------- render and limiter

def soft_clip(v: np.ndarray) -> np.ndarray:
    """Soft saturation above the knee (0.89)."""
    a = np.abs(v)
    over = a > KNEE
    if over.any():
        t = np.tanh((a[over] - KNEE) / (np.float32(1.0) - KNEE))
        v[over] = np.copysign(KNEE + (np.float32(1.0) - KNEE) * t, v[over]).astype(np.float32)
    return v


def apply_gain(x: np.ndarray, curve: GainCurve, offset_db: float = 0.0, ctx: Ctx | None = None,
               p0: float = 0.0, p1: float = 1.0, clip: bool = True) -> np.ndarray:
    """x * 10^((curve + offset)/20) per sample, in blocks; with optional soft saturation."""
    n = x.shape[0]
    y = np.empty_like(x, dtype=np.float32)
    for a in range(0, n, CHUNK):
        if ctx:
            ctx.check()
        b = min(n, a + CHUNK)
        g = (10.0 ** ((curve.at(np.arange(a, b)) + offset_db) / 20.0)).astype(np.float32)
        seg = x[a:b] * g[:, None]
        y[a:b] = soft_clip(seg) if clip else seg
        if ctx:
            ctx.progress(p0 + (p1 - p0) * b / max(1, n))
    return y


def peak_blocks_db(x: np.ndarray, block: int) -> np.ndarray:
    """Peak per block (maximum across channels) in dBFS, float32; includes the final partial block."""
    n = x.shape[0]
    nb = (n + block - 1) // block
    pk = np.zeros(nb, dtype=np.float32)
    full = n // block
    for a in range(0, full, max(1, CHUNK // block)):
        b = min(full, a + max(1, CHUNK // block))
        pk[a:b] = np.abs(x[a * block:b * block]).max(axis=1).reshape(b - a, block).max(axis=1)
    if full < nb:
        pk[full] = np.abs(x[full * block:]).max()
    return (20.0 * np.log10(pk.astype(np.float64) + 1e-12)).astype(np.float32)


def limiter_curve(peak_db: np.ndarray, gain_db: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Reduction per 1 ms block: ±1 block lookahead and a linear release of 12 dB per 80 ms.
    `peak_db` is the unprocessed peak and `gain_db` the planned gain per block."""
    m = peak_db.size
    r = np.maximum(0.0, peak_db.astype(np.float64) + gain_db - LIM_CEIL_DB)
    ra = r.copy()
    ra[1:] = np.maximum(ra[1:], r[:-1])
    ra[:-1] = np.maximum(ra[:-1], r[1:])
    cr = 12.0 * LIM_S / LIM_REL_S
    t = np.arange(m) * cr
    red = np.maximum.accumulate(ra + t) - t
    limited = red > 0.05
    return red, int(limited.sum()), float(red[limited].max()) if limited.any() else 0.0


# ---------------------------------------------------------------- classic pipeline

def stage1(x: np.ndarray, sr: int, prm: Params, ctx: Ctx) -> GainCurve:
    e = hop_meansq(x, sr)
    if e.size < 50:
        raise ValueError("audio too short to process")
    db = loudness_db(e, prm.window_s)
    reference = float(np.percentile(db, REF_PCT))
    desired = np.clip(reference - db, -SAFETY_DB, SAFETY_DB)
    g = smooth_gain(desired, prm.attack_s, prm.release_s)
    ctx.log(f"Stage 1: {e.size} frames · reference {reference:.1f} dB · gain {g.min():+.1f} to {g.max():+.1f} dB", 0)
    return GainCurve(g, hop_frames(sr, HOP_S))


def measure(y: np.ndarray, sr: int, lufs: bool) -> float:
    return lufs_integrated(k_subblocks(y, sr)) if lufs else rms_db(hop_meansq(y, sr))


def run_pipeline(x: np.ndarray, sr: int, prm: Params, ctx: Ctx | None = None) -> Result:
    ctx = ctx or Ctx()
    gain1 = stage1(x, sr, prm, ctx)
    ctx.progress(0.3)
    tmp = apply_gain(x, gain1, 0.0, ctx, 0.3, 0.55)
    level = measure(tmp, sr, prm.lufs)
    del tmp
    offset = prm.final_db - level
    ctx.log(f"Measurement {'LUFS' if prm.lufs else 'RMS'}: {level:.2f} dB · target {prm.final_db:.1f} · offset {offset:+.2f} dB", 0)
    ctx.progress(0.6)

    lim_hop = hop_frames(sr, LIM_S)
    peak = peak_blocks_db(x, lim_hop)
    centers = np.arange(peak.size) * lim_hop + 0.5 * lim_hop
    gblk = gain1.at(centers) + offset
    red, limited, max_red = limiter_curve(peak, gblk)
    _log_limiter(ctx, limited, max_red, offset)
    total = GainCurve((gblk - red).astype(np.float32), lim_hop)
    ctx.progress(0.65)
    y = apply_gain(x, total, 0.0, ctx, 0.65, 1.0)
    return Result(total, level, offset, limited, max_red, y)


def _log_limiter(ctx: Ctx, limited: int, max_red: float, offset: float) -> None:
    if limited:
        ctx.log(f"Peak limiter: {limited} 1 ms blocks reduced · maximum {max_red:.2f} dB · ceiling {LIM_CEIL_DB:.1f} dBFS", 0)
        if offset > 0:
            ctx.log("the limiter acted on part of the audio; lower the final level if you notice lost dynamics", 1)
    else:
        ctx.log("Peak limiter: no reduction needed", 0)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -20.0,
                       peak_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Linear internal normalization (no compression): integrated LUFS to `target_lufs` without exceeding
    `peak_db` of peak. Returns (audio, applied gain in dB). So the models always see an even level."""
    x2 = x if x.ndim == 2 else x[:, None]
    try:
        lufs = lufs_integrated(k_subblocks(x2, sr))
    except ValueError:
        return x, 0.0
    peak = 20.0 * np.log10(float(np.max(np.abs(x2))) + 1e-12)
    g = float(np.clip(min(target_lufs - lufs, peak_db - peak), -30.0, 40.0))
    return (x * np.float32(10.0 ** (g / 20.0))).astype(np.float32), g
