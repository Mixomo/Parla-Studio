"""Voice-aware leveling.

1. voice-isolator separates the voice from the background (background = original − voice).
2. SAVA marks where there is voice (probability per 40 ms frame, on the isolated voice).
3. Stage 1 levels ONLY the voice: the reference is computed on voice frames and in pauses the
   gain freezes (noise is not raised and does not "breathe").
4. The final measurement is dialogue loudness: voice frames only (RMS or LUFS).
5. The background is NOT leveled: it keeps its original level (plus an optional offset), so noise is not raised.
6. Mix and 1 ms peak limiter on the final mix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import leveler as lv
from . import models


PAUSE_FADE_S = 0.12  # minimum fade between voice and pause when there is no separation


@dataclass
class SmartParams(lv.Params):
    speech_threshold: float = 0.5   # P(voice) to consider a frame as voice
    hangover_s: float = 0.15        # margin around the voice (word edges)
    background_db: float = 0.0      # background relative to its ORIGINAL level (0 = noise is not raised)
    remove_background: bool = False
    engine: str = models.DEFAULT_ENGINE  # voice separation engine (see models.ENGINES)
    separate: bool = True           # False: no separation; SAVA on the original (audio with little or no background)
    sava_window_s: float | None = None  # SAVA window; None = automatic (the longest that fits in the free VRAM)
    use_sava: bool = True           # False: voice activity from the energy of the (separated) voice, no SAVA model
    sep_block_s: float | None = None    # separation block; None = automatic (free RAM / VRAM)


@dataclass
class SmartResult:
    output: np.ndarray
    voice: np.ndarray           # leveled voice (no limiter)
    background: np.ndarray      # background with its gain
    voice_gain: lv.GainCurve    # voice gain (dB, includes the offset)
    speech: np.ndarray          # P(voice) per 40 ms frame
    level_db: float
    offset_db: float
    limited_blocks: int
    max_reduction_db: float
    speech_ratio: float


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or mask.size == 0:
        return mask
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    i = np.arange(mask.size)
    return (c[np.minimum(i + n + 1, mask.size)] - c[np.maximum(i - n, 0)]) > 0


def energy_speech(mono: np.ndarray, sr: int) -> np.ndarray:
    """P(voice) per SAVA frame (40 ms) from energy alone: a soft gate above the estimated noise floor.

    Works well on a separated voice stem (pauses are near silence); on the original it only separates
    speech from quiet noise. Same format as models.speech_probability, so the rest of the pipeline is unchanged."""
    f = max(1, int(round(sr * models.SAVA_FRAME_S)))
    n = mono.size // f
    if n == 0:
        return np.zeros(0, np.float32)
    e = np.mean(np.square(mono[: n * f].astype(np.float64).reshape(n, f)), axis=1)
    db = 10.0 * np.log10(e + 1e-12)
    k = max(1, int(round(0.2 / models.SAVA_FRAME_S)))          # 200 ms smoothing
    db = np.convolve(db, np.ones(k) / k, mode="same")
    floor = float(np.percentile(db, 10))
    top = float(np.percentile(db, 95))
    thr = max(floor + 10.0, top - 30.0)                        # between the noise floor and loud speech
    return (1.0 / (1.0 + np.exp(-(db - thr) / 2.0))).astype(np.float32)


def voice_gain_curve(voice: np.ndarray, sr: int, speech_hops: np.ndarray, prm: SmartParams, ctx: lv.Ctx) -> lv.GainCurve:
    """Stage 1 on voice only: reference on voice frames and gain frozen in pauses."""
    e = lv.hop_meansq(voice, sr)
    if e.size < 50:
        raise ValueError("audio too short to process")
    db = lv.loudness_db(e, prm.window_s)
    sp = speech_hops[: e.size]
    if sp.sum() < 20:
        raise ValueError("not enough voice was detected to level")
    reference = float(np.percentile(db[sp], lv.REF_PCT))
    desired = np.clip(reference - db, -lv.SAFETY_DB, lv.SAFETY_DB)
    # In pauses: hold the last desired voice gain (no noise pumping)
    idx = np.where(sp, np.arange(sp.size), -1)
    idx = np.maximum.accumulate(idx)
    first = int(np.argmax(sp))
    idx[idx < 0] = first
    desired = desired[idx]
    g = lv.smooth_gain(desired, prm.attack_s, prm.release_s)
    ctx.log(f"Stage 1 (voice only): {int(sp.sum())} of {sp.size} frames with voice · reference {reference:.1f} dB · gain {g[sp].min():+.1f} to {g[sp].max():+.1f} dB", 0)
    return lv.GainCurve(g, lv.hop_frames(sr, lv.HOP_S))


def dialog_level(y: np.ndarray, sr: int, speech_hops: np.ndarray, lufs: bool) -> float:
    """Level measured only where there is voice."""
    if lufs:
        s = lv.k_subblocks(y, sr)
        # 400 ms block j = 100 ms sub-blocks j..j+3 = hops [20j, 20j+80)
        per_sub = np.add.reduceat(speech_hops[: s.size * 20].astype(np.int64), np.arange(0, s.size * 20, 20)) > 10
        mask = per_sub[:-3] & per_sub[1:-2] & per_sub[2:-1] & per_sub[3:] if per_sub.size >= 4 else per_sub
        return lv.lufs_integrated(s, mask)
    e = lv.hop_meansq(y, sr)
    return lv.rms_db(e[speech_hops[: e.size]])


def run_smart(x: np.ndarray, sr: int, prm: SmartParams, ctx: lv.Ctx | None = None,
              voice: np.ndarray | None = None, speech: np.ndarray | None = None) -> SmartResult:
    ctx = ctx or lv.Ctx()
    # 1) separation
    if voice is None and not prm.separate:
        # No separation: the whole audio is "voice"; SAVA marks where there is speech and gain freezes in pauses,
        # so the background (if any, little) is not raised. Much faster: only SAVA uses the GPU.
        ctx.log("No voice separation: the voice is detected directly in the original", 0)
        voice = x
    elif voice is None:
        ctx.log(f"Separating the voice: {models.ENGINES.get(prm.engine, prm.engine)} ({models.device_description()})", 0)
        voice = models.isolate_voice(x, sr, ctx.sub(0.0, 0.35), prm.engine, prm.sep_block_s)
    ctx.check()
    background = x - voice
    ctx.progress(0.35)
    # 2) voice activity
    vad = "SAVA" if prm.use_sava else "energy"
    if speech is None and prm.use_sava:
        ctx.log("Detecting voice activity with SAVA...", 0)
        p0 = 0.35 if prm.separate else 0.0
        speech = models.speech_probability(voice.mean(axis=1), sr, ctx.sub(p0, 0.45), prm.sava_window_s)
    elif speech is None:
        ctx.log("SAVA off: voice activity from the energy of the separated voice" if prm.separate else
                "SAVA off: voice activity from the energy of the original", 0)
        speech = energy_speech(voice.mean(axis=1), sr)
    ctx.check()
    n_hops = x.shape[0] // lv.hop_frames(sr, lv.HOP_S)
    sp = models.speech_on_hops(speech, n_hops, lv.HOP_S) >= prm.speech_threshold
    sp = _dilate(sp, int(round(prm.hangover_s / lv.HOP_S)))
    ratio = float(sp.mean()) if sp.size else 0.0
    ctx.log(f"{vad}: voice in {100 * ratio:.0f}% of the audio", 0)
    ctx.progress(0.45)
    # 3) stage 1 on the voice
    g1 = voice_gain_curve(voice, sr, sp, prm, ctx)
    v1 = lv.apply_gain(voice, g1, 0.0, ctx, 0.45, 0.6, clip=False)
    # 4) dialogue loudness
    level = dialog_level(v1, sr, sp, prm.lufs)
    del v1
    offset = prm.final_db - level
    ctx.log(f"Dialogue measurement {'LUFS' if prm.lufs else 'RMS'} (voice only): {level:.2f} dB · target {prm.final_db:.1f} · offset {offset:+.2f} dB", 0)
    # 5) voice and background with their gains
    vg = lv.GainCurve((g1.db + offset).astype(np.float32), g1.hop)
    if not prm.separate:
        # Without separation the "voice" is the whole audio: in pauses (no voice according to SAVA) the gain returns to
        # the background level (0 dB + offset) instead of keeping the voice gain; so background noise is not raised.
        n = min(vg.db.size, sp.size)
        # "Remove the background" without separation: pauses fade smoothly to silence (−120 dB ≈ gain 0)
        pause_db = -120.0 if prm.remove_background else prm.background_db
        desired = np.where(sp[:n], vg.db[:n].astype(np.float64), pause_db)
        fade = max(prm.attack_s, prm.release_s, PAUSE_FADE_S)  # smooth fade, no abrupt cuts
        smoothed = lv.smooth_gain(desired, fade, fade)
        vg = lv.GainCurve(np.concatenate([smoothed, vg.db[n:]]).astype(np.float32), g1.hop)
        ctx.log("Voiceless pauses: faded to silence" if prm.remove_background else
                f"Voiceless pauses: gain {pause_db:+.1f} dB (the background keeps its original level)", 0)
    v = lv.apply_gain(voice, vg, 0.0, ctx, 0.6, 0.75, clip=False)
    if prm.remove_background:
        bg = np.zeros_like(background)
        ctx.log("Background removed", 0)
    else:
        bg = (background * np.float32(10.0 ** (prm.background_db / 20.0))).astype(np.float32)
        ctx.log(f"Background: {prm.background_db:+.1f} dB vs. original (not leveled; noise is not raised)", 0)
    mix = v + bg
    ctx.progress(0.8)
    # 6) limiter on the mix
    lim_hop = lv.hop_frames(sr, lv.LIM_S)
    peak = lv.peak_blocks_db(mix, lim_hop)
    red, limited, max_red = lv.limiter_curve(peak, np.zeros(peak.size))
    lv._log_limiter(ctx, limited, max_red, offset)
    out = lv.apply_gain(mix, lv.GainCurve((-red).astype(np.float32), lim_hop), 0.0, ctx, 0.8, 1.0)
    return SmartResult(out, v, bg, vg, speech, level, offset, limited, max_red, ratio)
