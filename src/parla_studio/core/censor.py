"""Word censoring: search in the transcript (CrisperWhisper) and beep, silence or cut.

With the voice isolated (voice-isolator) the beep or silence replaces only the VOICE: music and ambience
keep playing underneath, with no gaps. A cut removes the whole segment with a crossfade.
Each word's boundaries snap to the nearby energy minimum of the voice, so no audible "tails" of the
censored word remain.
"""
from __future__ import annotations

import csv
import fnmatch
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models

PUNCT = re.compile(r"[^\w*\s'-]", re.UNICODE)


def normalize(s: str, accents: bool = True) -> str:
    """lowercase, no punctuation or brackets and (if `accents`) no accents; keeps ñ."""
    s = s.lower().replace("ñ", "\0")
    if accents:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.replace("\0", "ñ")
    return PUNCT.sub(" ", s).strip()


@dataclass
class Match:
    i: int            # first word
    j: int            # one past the last
    start: float      # seconds (editable)
    end: float
    text: str
    context: str
    enabled: bool = True
    source: str = "search"   # "search", "DB" (preventive censoring from the lexicon) or "DB + search"


def search(words: list[models.Word], query: str, ignore_accents: bool = True, whole_word: bool = True) -> list[Match]:
    """All matches of the terms (separated by commas or newlines). A term can be a multi-word phrase;
    `*` is a wildcard (e.g. `put*`). Without `whole_word` it also finds the term inside longer words."""
    terms = [t for t in (normalize(x, ignore_accents) for x in re.split(r"[,\n;]+", query)) if t]
    norm = [normalize(w.text, ignore_accents) for w in words]
    found: dict[tuple[int, int], Match] = {}
    for term in terms:
        toks = term.split()
        n = len(toks)
        for i in range(len(words) - n + 1):
            ok = True
            for k, tok in enumerate(toks):
                w = norm[i + k]
                if "*" in tok:
                    ok = fnmatch.fnmatchcase(w, tok) if whole_word else fnmatch.fnmatchcase(w, f"*{tok}*")
                else:
                    ok = (w == tok) if whole_word else (tok in w)
                if not ok:
                    break
            if ok and (i, i + n) not in found:
                ctx = " ".join(x.text for x in words[max(0, i - 6):i]) + " [" + \
                      " ".join(x.text for x in words[i:i + n]) + "] " + " ".join(x.text for x in words[i + n:i + n + 6])
                found[(i, i + n)] = Match(i, i + n, words[i].start, words[i + n - 1].end,
                                          " ".join(x.text for x in words[i:i + n]), ctx.strip())
    return sorted(found.values(), key=lambda m: m.start)


def union_spans(cw: list[models.Word], aligned: list[models.Word], max_ext_s: float = 0.15) -> list[models.Word]:
    """Words to censor: the union of CrisperWhisper's span and the aligner's (each side extended at most
    `max_ext_s` beyond CrisperWhisper). Leaving part of a word audible is worse than censoring a few ms more:
    when the two disagree, both are covered, and `refine` trims the silence at the edges against the audio."""
    out = []
    for c, a in zip(cw, aligned):
        s = min(c.start, max(a.start, c.start - max_ext_s))
        e = max(c.end, min(a.end, c.end + max_ext_s))
        out.append(models.Word(c.text, s, max(e, s + 0.01), c.prob, c.speaker))
    return out


def refine(matches: list[Match], voice_mono: np.ndarray, sr: int, pad_s: float = 0.03, snap_s: float = 0.08) -> None:
    """Widen each match by `pad_s` and snap it to the energy minimum within ±`snap_s`."""
    hop = max(1, sr // 1000)  # 1 ms
    n = voice_mono.size // hop
    e = (voice_mono[: n * hop].astype(np.float64) ** 2).reshape(n, hop).mean(axis=1)
    e = np.convolve(e, np.ones(5) / 5, mode="same")

    def snap(t: float, lo: float, hi: float) -> float:
        a, b = max(0, int(lo * 1000)), min(n, int(hi * 1000))
        return t if b <= a + 1 else (a + int(np.argmin(e[a:b]))) / 1000.0

    edb = 10 * np.log10(np.convolve(e, np.ones(10) / 10, mode="same") + 1e-12)
    for m in matches:
        # 1) trim to where the voice has real energy: sentence-final words carry the pause with them
        a, b = int(m.start * 1000), min(n, int(m.end * 1000))
        if b > a + 20:
            seg = edb[a:b]
            loud = np.nonzero(seg > seg.max() - 30.0)[0]
            if loud.size:
                m.start, m.end = (a + loud[0]) / 1000.0, (a + loud[-1] + 1) / 1000.0
        # 2) margin and snap to the nearby energy minimum
        s0, e0 = m.start - pad_s, m.end + pad_s
        m.start = max(0.0, snap(s0, s0 - snap_s, min(s0 + snap_s, m.start)))
        m.end = min(voice_mono.size / sr, snap(e0, max(e0 - snap_s, m.end), e0 + snap_s))


def _ramp(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, max(n, 1), dtype=np.float32))


def render(x: np.ndarray, sr: int, matches: list[Match], mode: str = "beep", voice: np.ndarray | None = None,
           beep_hz: float = 1000.0, beep_rel_db: float = -3.0, fade_s: float = 0.008) -> np.ndarray:
    """mode: 'beep', 'silence' or 'cut' (remove the segment).
    With `voice` (same shape as x) the beep/silence replaces only the voice and the background continues."""
    active = sorted((m for m in matches if m.enabled and m.end > m.start), key=lambda m: m.start)
    if not active:
        return x.copy()
    if mode == "cut":
        return _cut(x, sr, active, fade_s)
    y = x.copy()
    nf = max(1, int(fade_s * sr))
    for m in active:
        a, b = int(m.start * sr), min(x.shape[0], int(m.end * sr))
        if b - a < 2:
            continue
        # 0→1→0 envelope with fades: how much is replaced at each sample
        env = np.ones(b - a, np.float32)
        k = min(nf, (b - a) // 2)
        env[:k] = _ramp(k)
        env[b - a - k:] = _ramp(k)[::-1]
        env = env[:, None]
        removed = voice[a:b] if voice is not None else x[a:b]
        y[a:b] -= env * removed
        if mode == "beep":
            ref = removed if removed.size else x[a:b]
            rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))) or 1e-3
            amp = min(0.5, rms * np.sqrt(2) * 10 ** (beep_rel_db / 20))  # sine peak ≈ level of the word
            t = np.arange(b - a) / sr
            tone = (amp * np.sin(2 * np.pi * beep_hz * t)).astype(np.float32)[:, None]
            y[a:b] += env * tone
    return y


def _cut(x: np.ndarray, sr: int, active: list[Match], fade_s: float) -> np.ndarray:
    nf = max(1, int(fade_s * sr))
    parts, pos = [], 0
    for m in active:
        a, b = int(m.start * sr), min(x.shape[0], int(m.end * sr))
        if a < pos:
            a = pos
        if b <= a:
            continue
        parts.append(x[pos:a])
        pos = b
    parts.append(x[pos:])
    out = parts[0].copy()
    for p in parts[1:]:
        k = min(nf, out.shape[0], p.shape[0])
        if k > 0:
            r = _ramp(k)[:, None]
            out[-k:] = out[-k:] * r[::-1] + p[:k] * r  # crossfade at the joint
            out = np.concatenate([out, p[k:]])
        else:
            out = np.concatenate([out, p])
    return out


def map_time_after_cut(t: float, matches: list[Match], fade_s: float = 0.008) -> float:
    """Equivalent position of `t` (s in the original) in the audio with the segments cut out."""
    removed, last_end = 0.0, 0.0
    for m in sorted((m for m in matches if m.enabled and m.end > m.start), key=lambda m: m.start):
        a, b = max(m.start, last_end), m.end
        if b <= a:
            continue
        if t >= b:
            removed += (b - a) + fade_s  # every crossfaded joint shortens it by another `fade_s`
        elif t > a:
            return a - removed
        last_end = b
    return max(0.0, t - removed)


def export_list(matches: list[Match], path: Path, mode: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "word", "source", "context", "censor"])
        for m in matches:
            if m.enabled:
                w.writerow([f"{m.start:.3f}", f"{m.end:.3f}", m.text, m.source, m.context, mode])
