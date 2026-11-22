"""Voice clips for datasets (audio + text) with natural cuts.

Per-file flow:
0. Internal normalization (LUFS -20, peak -1 dB, linear) so the models see an even level.
1. (optional) separation (voice / music / noise): clean clips without background.
2. CrisperWhisper 2.0: verbatim transcription with word timestamps.
3. (optional) forced aligner (Qwen3 / MMS / wav2vec2): fixes each word's edges against the audio.
4. SAVA + energy (broadband and highs): a pause counts only if there is really no voice; clip edges
   extend while the word's tail continues (weak syllables, a plural "s").
5. Segmentation: words are grouped into clips between `min_s` and `max_s`, cutting at real pauses;
   sentence ends and long pauses are preferred. Each cut lands at the lowest-energy point of the
   pause, with padding and a 10 ms fade.
6. Export as a ZIP with the layout of the chosen TTS project (LJSpeech, audio + .txt, Fish Speech,
   Qwen3-TTS, VoxCPM, XTTS-v2 / Coqui or Kaldi) + metadata.jsonl.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio_io, leveler as lv, models
from ..i18n import tr as _tr

SENT_END = re.compile(r"[.!?…。！？]['\"”»)]*$")
CLAUSE_END = re.compile(r"[,;:—–]['\"”»)]*$")
FILLER = re.compile(r"^\[[^\]]*\]$")  # CrisperWhisper tags: [UM], [UH], [laughter], [lipsmack]...


@dataclass
class DatasetParams:
    language: str = "en"
    hallucination_mitigation: bool = True
    isolate: bool = True
    engine: str = models.DEFAULT_ENGINE  # separation engine (models.ENGINES)
    sep_block_s: float | None = None   # None = automatic (free RAM / VRAM)
    verbatim: bool = True
    keep_fillers: bool = False    # keep CrisperWhisper's tags ([UM], [UH], [laughter]...) in the text
    min_s: float = 2.0
    max_s: float = 12.0
    min_pause_s: float = 0.20     # minimum pause to cut
    pad_s: float = 0.12           # silence padding around the voice
    fade_s: float = 0.010
    out_sr: int = 24000
    min_word_prob: float = 0.2    # flag clips with any word below this confidence for review
    min_speech_ratio: float = 0.5 # minimum mean P(voice) within the clip
    weak_word_db: float = 15.0    # a word (≥150 ms) this much weaker than the strong words of the clip = echo or background voice
    speaker: str = "speaker"      # name when not diarizing
    diarize: bool = False         # cut at speaker changes and flag overlapping voices
    max_overlap: float = 0.10     # fraction of overlapping voices tolerated in a clip
    id_threshold: float = 0.80    # minimum similarity to recognize an enrolled person
    normalize: bool = True        # internal normalization before processing
    align: bool = True            # forced aligner on top of CrisperWhisper's timestamps
    use_sava: bool = False        # SAVA voice detection for pauses and the "little voice" check; off: pauses come
                                  # from the word timestamps (better with music or effects under the voice)
    aligner: str = "qwen"         # aligner engine (aligner.ALIGNERS); outside its 11 languages it uses MMS
    edge_max_s: float = 0.35      # how far a clip's end may extend following the word's tail
    write_log: bool = True        # diagnostic log (CrisperWhisper words and corrections)


@dataclass
class Clip:
    source: str
    start: float
    end: float
    text: str
    words: int
    min_prob: float
    speech_ratio: float
    keep: bool = True
    reason: str = ""              # why it was excluded (empty if valid)
    text_edited: bool = False     # the text was corrected by hand (not recomputed when the clip moves)
    speaker: str = ""             # dominant speaker (recognized person or "Speaker N")
    overlap: float = 0.0          # fraction of the clip with overlapping voices
    name: str = ""
    audio: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clean_text(words: list[models.Word], keep_fillers: bool) -> str:
    toks = [w.text for w in words if keep_fillers or not FILLER.match(w.text)]
    text = " ".join(toks)
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _energy_db(mono: np.ndarray, sr: int, hop_s: float = 0.01) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr * hop_s))
    n = mono.size // hop
    e = (mono[: n * hop].astype(np.float64) ** 2).reshape(n, hop).mean(axis=1)
    return 10 * np.log10(e + 1e-12), hop_s


def _best_cut(a: float, b: float, edb: np.ndarray, hop_s: float) -> float:
    """Lowest-energy instant within the pause [a, b] (seconds)."""
    i0, i1 = int(a / hop_s), int(np.ceil(b / hop_s))
    if i1 <= i0 + 1 or i0 >= edb.size:
        return (a + b) / 2
    seg = edb[i0:min(i1, edb.size)]
    # short smoothing so it doesn't land in a 10 ms gap inside a consonant
    k = np.ones(3) / 3
    seg = np.convolve(seg, k, mode="same") if seg.size >= 3 else seg
    return (i0 + int(np.argmin(seg)) + 0.5) * hop_s


def segment(words: list[models.Word], prm: DatasetParams, owners: list | None = None,
            gaps: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Word groups [i, j) for every clip. With `owners` (speaker of each word) two speakers are never mixed
    in one clip: the words are first split into turns and segmented within each turn.
    `gaps[j]`: real pause (s) between word j-1 and word j (no voice according to SAVA and no energy); if missing,
    the gap between the word timestamps is used."""
    def gap_at(j: int) -> float:
        if j >= n:
            return 10.0
        raw = words[j].start - words[j - 1].end
        return float(gaps[j]) if gaps is not None else raw
    n = len(words)
    if owners is not None and len(owners) == len(words):
        groups, start, cur = [], 0, None
        for k, o in enumerate(owners + [object()]):
            if o is None:          # word without a clear speaker: stays with the current turn
                continue
            if cur is None:
                cur = o
            if o != cur or k == len(words):
                sub = gaps[start:k] if gaps is not None else None
                groups += [(a + start, b + start) for a, b in segment(words[start:k], prm, None, sub)]
                start, cur = k, o
        return groups
    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        best_j, best_score = -1, -1e9
        j = i + 1
        while j <= n:
            dur = words[j - 1].end - words[i].start
            if dur > prm.max_s and j - 1 > i:
                break
            gap = gap_at(j)
            if dur >= prm.min_s and gap >= prm.min_pause_s:
                score = min(gap, 1.5) * 2.0
                if SENT_END.search(words[j - 1].text):
                    score += 3.0
                elif CLAUSE_END.search(words[j - 1].text):
                    score += 1.0
                score += 0.3 * min(dur / prm.max_s, 1.0)  # slight preference for longer clips
                if score > best_score:
                    best_j, best_score = j, score
            j += 1
        if best_j < 0:
            # no valid pause in range: cut at the best pause available. Ties (e.g. every measured pause is 0)
            # are broken by the pause between the timestamps, then a sentence or clause end, then reaching the
            # minimum duration and then by length: never a one-word clip just because the first word came first.
            k, best = i + 1, None
            m = i + 1
            while m < n and words[m - 1].end - words[i].start <= prm.max_s:
                dur = words[m - 1].end - words[i].start
                raw = words[m].start - words[m - 1].end
                punct = 2 if SENT_END.search(words[m - 1].text) else 1 if CLAUSE_END.search(words[m - 1].text) else 0
                key = (dur >= prm.min_s, round(gap_at(m), 2), round(raw, 2), punct, dur)
                if best is None or key > best:
                    k, best = m, key
                m += 1
            best_j = k if m < n else n
        groups.append((i, best_j))
        i = best_j
    return groups


@dataclass
class Source:
    """Analysis of one file, reused when clips are edited by hand."""
    path: Path
    mono: np.ndarray            # voice (isolated if requested), mono, at `sr`
    sr: int
    words: list[models.Word]
    speech: np.ndarray          # P(voice) per 40 ms frame
    edb: np.ndarray             # energy every 10 ms (dB)
    word_db: np.ndarray         # level of each word (dB)
    diar: object | None = None  # speakers.Diarization (on the original)
    names: dict = field(default_factory=dict)  # diarizer index → name
    owners: list | None = None  # speaker of each word
    active: np.ndarray | None = None  # voice energy every 10 ms (broadband or highs: includes the "s")
    quiet: np.ndarray | None = None   # real pause every 10 ms: no voice energy and SAVA < 0.5
    pause_from_mask: bool = True      # False: the mask found no clean silences, pauses come from the timestamps
    align_rows: list | None = None    # aligner report (for the log)
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.mono.size / self.sr


def _activity(mono: np.ndarray, sr: int, edb: np.ndarray, speech: np.ndarray,
              hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """(voice energy, real pause) every 10 ms. Energy is checked in broadband and in the highs (derivative):
    a final "s", "f" or "ch" has almost no low-frequency energy but does have high-frequency energy."""
    hdb, _ = _energy_db(np.diff(mono, prepend=mono[:1]), sr, hop_s)
    n = min(edb.size, hdb.size)
    idx = np.minimum((np.arange(n) * hop_s / models.SAVA_FRAME_S).astype(int), max(0, speech.size - 1))
    sp = speech[idx] >= 0.5 if speech.size else np.zeros(n, bool)

    def on(db: np.ndarray) -> np.ndarray:
        floor = np.percentile(db, 10)
        ref = np.percentile(db[sp] if sp.sum() > 50 else db, 60)
        return db > max(floor + 10.0, ref - 38.0)

    act = on(edb[:n]) | on(hdb[:n])
    return act, ~act & ~sp


def _extend_end(t: float, limit: float, act: np.ndarray, hop_s: float = 0.01) -> float:
    """Follow the word's tail from `t` while there is voice energy (tolerates 30 ms gaps)."""
    i, lim = int(t / hop_s), min(int(limit / hop_s), act.size)
    last, gap = i, 0
    while i < lim:
        if act[i]:
            last, gap = i + 1, 0
        else:
            gap += 1
            if gap > 3:
                break
        i += 1
    return max(t, last * hop_s)


def _extend_start(t: float, limit: float, act: np.ndarray, hop_s: float = 0.01) -> float:
    """Same, backwards: soft attacks (aspiration, initial "s")."""
    i, lim = int(t / hop_s) - 1, max(int(limit / hop_s), 0)
    first, gap = i + 1, 0
    while i >= lim and i < act.size:
        if act[i]:
            first, gap = i, 0
        else:
            gap += 1
            if gap > 3:
                break
        i -= 1
    return min(t, first * hop_s)


def analyze_source(path: Path, prm: DatasetParams, ctx: lv.Ctx, store=None,
                   audio: tuple[np.ndarray, int] | None = None) -> Source:
    """`audio`: (x, sr) already loaded (e.g. separated with "Separate now"); otherwise the file is read."""
    from . import speakers
    x, sr = audio if audio is not None else audio_io.load(path)
    notes = [f"File: {path}", f"Duration: {x.shape[0] / sr:.2f} s · {sr} Hz · {x.shape[1]} channel(s)",
             f"Language: {prm.language} · verbatim: {prm.verbatim} · hallucination mitigation: "
             f"{prm.hallucination_mitigation}"]
    if prm.normalize:
        x, g = lv.normalize_loudness(x, sr)
        ctx.log(f"{path.name}: internal normalization {g:+.1f} dB (LUFS -20, peak -1 dB)", 0)
        notes.append(f"Internal normalization: {g:+.1f} dB")
    diar = None
    if prm.diarize:
        ctx.log(f"{path.name}: diarizing (Nemotron 3 Diarization)...", 0)
        diar = speakers.diarize(audio_io.to_mono(x), sr)  # on the original, before isolating
    if prm.isolate:
        ctx.log(f"{path.name}: isolating the voice ({models.ENGINES.get(prm.engine, prm.engine)})...", 0)
        x = models.isolate_voice(x, sr, ctx.sub(0.0, 0.4), prm.engine, prm.sep_block_s)
        notes.append(f"Separation: {models.ENGINES.get(prm.engine, prm.engine)}")
        if prm.normalize:  # separation changes the level: the voice is normalized again
            x, g = lv.normalize_loudness(x, sr)
            notes.append(f"Normalization after separation: {g:+.1f} dB")
    ctx.check()
    mono = audio_io.to_mono(x)
    ctx.log(f"{path.name}: transcribing with CrisperWhisper 2.0 turbo ({prm.language})...", 0)
    _, words = models.transcribe(mono, sr, prm.language, prm.verbatim, prm.hallucination_mitigation,
                                 ctx.sub(0.4 if prm.isolate else 0.0, 0.7))
    ctx.check()
    dur = mono.size / sr
    outside = [w for w in words if w.start >= dur - 0.05]  # typical Whisper hallucination at the end ("Thank you!")
    if outside:
        words = [w for w in words if w.start < dur - 0.05]
        msg = f"dropped {len(outside)} word(s) outside the audio: {' '.join(w.text for w in outside)}"
        ctx.log(f"{path.name}: {msg}", 1)
        notes.append(msg.capitalize())
    for w in words:
        w.end = min(w.end, dur)
    rows = None
    if prm.align and words:
        from . import aligner
        desc = aligner.describe(prm.aligner, prm.language)
        ctx.log(f"{path.name}: aligning {len(words)} words with {desc}...", 0)
        notes.append(f"Aligner: {desc}")
        words, rows = aligner.align(mono, sr, words, ctx.sub(0.7, 0.85), prm.aligner, prm.language)
        ok = sum(r.status == "aligned" for r in rows)
        ctx.log(f"{path.name}: aligner: {ok} of {len(rows)} words corrected against the audio", 0)
    ctx.check()
    if prm.use_sava:
        speech = models.speech_probability(mono, sr, ctx.sub(0.85, 0.95))
    else:
        speech = np.ones(int(np.ceil(mono.size / sr / models.SAVA_FRAME_S)) + 1, np.float32)
        notes.append("SAVA off: pauses from the word timestamps, no voice-ratio check")
        ctx.log(f"{path.name}: SAVA off: pauses taken from the word timestamps", 0)
    edb, _ = _energy_db(mono, sr)
    word_db = np.array([20 * np.log10(np.sqrt(np.mean(mono[int(w.start * sr):max(int(w.end * sr), int(w.start * sr) + 1)]
                                                      .astype(np.float64) ** 2)) + 1e-9) for w in words])
    src = Source(path, mono, sr, words, speech, edb, word_db)
    src.active, src.quiet = _activity(mono, sr, edb, speech)
    if not prm.use_sava:
        src.quiet = None   # no silence mask: segment() uses the pauses between the word timestamps
    src.align_rows, src.notes = rows, notes
    if diar is not None:
        spk = speakers.describe(diar, mono, sr, store, prm.id_threshold)
        src.diar, src.names = diar, {s.idx: s.name for s in spk}
        src.owners = [o for o, _ in speakers.assign_words(words, diar)]
        ctx.log(f"{path.name}: {len(spk)} speaker(s): " + ", ".join(
            f"{s.name} ({s.seconds:.0f} s)" for s in spk), 0)
    return src


def words_in(src: Source, start: float, end: float) -> tuple[int, int]:
    """Indices [i, j) of the words whose center falls within [start, end]."""
    mids = np.array([(w.start + w.end) / 2 for w in src.words])
    if mids.size == 0:
        return 0, 0
    i = int(np.searchsorted(mids, start, "left"))
    j = int(np.searchsorted(mids, end, "right"))
    return i, max(i, j)


def clip_audio(src: Source, start: float, end: float, prm: DatasetParams) -> np.ndarray:
    """Audio of the clip [start, end] (s) at `out_sr`, with short fades (also used to rebuild clips of a project)."""
    seg = audio_io.resample(src.mono[int(start * src.sr):int(end * src.sr)], src.sr, prm.out_sr).copy()
    nf = min(int(prm.fade_s * prm.out_sr), seg.size // 2)
    if nf > 0:
        ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
        seg[:nf] *= ramp
        seg[-nf:] *= ramp[::-1]
    return seg


def make_clip(src: Source, start: float, end: float, prm: DatasetParams, text: str | None = None) -> Clip:
    """Build a clip [start, end] (s): audio at `out_sr` with fades, text of the words it contains and
    quality checks. Used for detected clips and for clips edited by hand."""
    start, end = max(0.0, start), min(src.duration, end)
    i, j = words_in(src, start, end)
    ws = src.words[i:j]
    seg = clip_audio(src, start, end, prm)
    f0 = int(start / models.SAVA_FRAME_S)
    f1 = max(f0 + 1, int(np.ceil(end / models.SAVA_FRAME_S)))
    sr_ratio = float(src.speech[f0:f1].mean()) if f1 <= src.speech.size else 1.0
    min_prob = min((w.prob for w in ws), default=0.0)
    clip = Clip(str(src.path), start, end, text if text is not None else _clean_text(ws, prm.keep_fillers),
                len(ws), min_prob, sr_ratio, audio=seg)
    reasons = []
    if not clip.text:
        reasons.append("no text")
    if clip.duration < prm.min_s * 0.5:
        reasons.append("too short")
    if clip.duration > prm.max_s + 2 * prm.pad_s:
        reasons.append("too long")
    if ws and min_prob < prm.min_word_prob:
        reasons.append("low confidence")
    if sr_ratio < prm.min_speech_ratio:
        reasons.append("little voice")
    if src.diar is not None:
        k, clip.overlap = src.diar.who(start, end)
        clip.speaker = src.names.get(k, _tr("Unknown")) if k is not None else _tr("Unknown")
        if clip.overlap > prm.max_overlap:
            reasons.append("overlapping voices")
    else:
        clip.speaker = prm.speaker
    lv_words = src.word_db[i:j]
    # only words of at least 150 ms: very short ones ("y", "te") sometimes land on an instant without energy.
    # Reference = 75th percentile (strong words): with echo, half of the words are weak copies.
    long_w = np.array([w.end - w.start >= 0.15 for w in ws], dtype=bool)
    if long_w.sum() >= 2 and (np.percentile(lv_words, 75) - lv_words[long_w].min()) > prm.weak_word_db:
        reasons.append("weak words (echo or background voice)")
    clip.reason = ", ".join(reasons)
    clip.keep = not reasons
    return clip


def split_clip(src: Source, clip: Clip, t: float, prm: DatasetParams) -> tuple[Clip, Clip]:
    """Two clips [start, t] and [t, end]. Each gets the transcript of its own words; a text corrected by hand is
    split at the same word if it still has as many words as the clip, otherwise each part takes its words'."""
    if not (clip.start + 0.05 < t < clip.end - 0.05):
        raise ValueError("the split point must be inside the clip")
    a_text = b_text = None
    if clip.text_edited:
        i, j = words_in(src, clip.start, clip.end)
        k, _ = words_in(src, t, clip.end)
        parts = clip.text.split()
        if len(parts) == j - i and i <= k <= j:
            a_text, b_text = " ".join(parts[:k - i]), " ".join(parts[k - i:])
    a = make_clip(src, clip.start, t, prm, a_text)
    b = make_clip(src, t, clip.end, prm, b_text)
    a.text_edited, b.text_edited = a_text is not None, b_text is not None
    a.name = clip.name
    return a, b


def merge_clips(src: Source, clips: list[Clip], prm: DatasetParams) -> Clip:
    """One clip from the first start to the last end of `clips` (same source file). The transcript is the words in
    that span; if any of the clips had its text corrected by hand, the texts are joined in time order instead."""
    if len(clips) < 2 or any(c.source != str(src.path) for c in clips):
        raise ValueError("merge needs two or more clips of the same file")
    ordered = sorted(clips, key=lambda c: c.start)
    edited = any(c.text_edited for c in ordered)
    text = " ".join(c.text.strip() for c in ordered if c.text.strip()) if edited else None
    new = make_clip(src, ordered[0].start, max(c.end for c in ordered), prm, text)
    new.text_edited, new.name = edited, ordered[0].name
    return new


def detect_clips(src: Source, prm: DatasetParams) -> list[Clip]:
    """Automatic clips: word groups cut at the energy minimum of each pause."""
    words, total, hop_s = src.words, src.duration, 0.01
    act = src.active
    gaps = None
    src.pause_from_mask = True
    if src.quiet is not None:
        gaps = np.zeros(len(words))
        for j in range(1, len(words)):
            a, b = int(words[j - 1].end / hop_s), int(np.ceil(words[j].start / hop_s))
            gaps[j] = float(src.quiet[a:max(a, b)].sum()) * hop_s
        # the silence mask (no voice for SAVA and no energy) fails when music or effects stay under the voice:
        # if it finds well under a third of the pauses the timestamps show, the timestamps' pauses are used
        raw = np.array([0.0] + [words[j].start - words[j - 1].end for j in range(1, len(words))])
        if (raw >= prm.min_pause_s).sum() >= 10 and (gaps >= prm.min_pause_s).sum() < 0.3 * (raw >= prm.min_pause_s).sum():
            gaps = None
            src.pause_from_mask = False

    rows = src.align_rows if src.align_rows and len(src.align_rows) == len(words) else None

    def end_of(k: int) -> float:   # real end of word k (with its tail)
        # the later of CrisperWhisper and the aligner (CTC sometimes closes the final letter too early)
        e = max(words[k].end, rows[k].cw_end) if rows else words[k].end
        nxt = words[k + 1].start if k + 1 < len(words) else total
        e = min(e, nxt - 0.02) if k + 1 < len(words) else e
        return _extend_end(e, min(nxt - 0.02, e + prm.edge_max_s), act) if act is not None else e

    def start_of(k: int) -> float:  # real start of word k (with its attack)
        st = min(words[k].start, rows[k].cw_start) if rows else words[k].start
        prv = words[k - 1].end if k > 0 else 0.0
        st = max(st, prv + 0.02) if k > 0 else st
        return _extend_start(st, max(prv + 0.02, st - 0.2), act) if act is not None else st

    clips = []
    for i, j in segment(words, prm, src.owners, gaps):
        ws = words[i:j]
        s0, e0 = start_of(i), end_of(j - 1)
        lo = _best_cut(end_of(i - 1), s0, src.edb, hop_s) if i > 0 else 0.0
        hi = _best_cut(e0, start_of(j), src.edb, hop_s) if j < len(words) else total
        clips.append(make_clip(src, max(lo, s0 - prm.pad_s), min(hi, e0 + prm.pad_s), prm,
                               _clean_text(ws, prm.keep_fillers)))
    return clips


def write_log(src: Source, clips: list[Clip], prm: DatasetParams) -> Path | None:
    """Diagnostic log: every CrisperWhisper word (start, end, confidence, tag), what the aligner corrected
    and every clip with its end edge relative to its last word."""
    from . import aligner
    rows = src.align_rows or [aligner.AlignRow(w.text, w.start, w.end, w.prob, w.start, w.end, "not aligned")
                              for w in src.words]
    tail = ["CLIPS", f"{'#':>5}  {'start':>9} {'end':>9} {'dur':>6}  {'word_end':>9} {'tail ms':>7}  "
            f"{'status':<28} text"]
    for k, c in enumerate(clips):
        i, j = words_in(src, c.start, c.end)
        last = src.words[j - 1].end if j > i else c.end
        tail.append(f"{k:>5}  {c.start:9.3f} {c.end:9.3f} {c.duration:6.2f}  {last:9.3f} "
                    f"{(c.end - last) * 1000:+7.0f}  {('OK' if c.keep else c.reason)[:28]:<28} {c.text}")
    try:
        return aligner.write_log(src.path.name, rows, src.notes, tail, "dataset",
                                 aligner.describe(prm.aligner, prm.language) if src.align_rows else "not aligned")
    except OSError:
        return None


def build_clips(path: Path, prm: DatasetParams, ctx: lv.Ctx, store=None,
                audio: tuple[np.ndarray, int] | None = None) -> tuple[list[Clip], Source]:
    """Analyze a file and return its clips and the analysis (for later editing)."""
    src = analyze_source(path, prm, ctx, store, audio)
    if not src.words:
        ctx.log(f"{path.name}: no words detected", 1)
        return [], src
    clips = detect_clips(src, prm)
    if not getattr(src, "pause_from_mask", True):
        ctx.log(f"{path.name}: the voice has no clean silences (music or noise under it): pauses taken from "
                f"the word timestamps", 1)
    ctx.log(f"{path.name}: {len(src.words)} words → {len(clips)} clips ({sum(c.keep for c in clips)} valid)", 0)
    if prm.write_log:
        lp = write_log(src, clips, prm)
        if lp:
            ctx.log(f"{path.name}: word and cut log → {lp}", 0)
    return clips, src


# Export layouts for the usual TTS projects. Almost every training GUI imports
# "audio + .txt with the same name" and builds its own manifest; Qwen3-TTS and VoxCPM document their JSONL.
EXPORT_LAYOUTS = {
    "ljspeech": "LJSpeech (generic)",
    "pairs": "Audio + .txt",
    "audio": "Audio only",
    "fish": "Fish Speech / S2",
    "qwen": "Qwen3-TTS SFT",
    "voxcpm": "VoxCPM",
    "xtts": "XTTS-v2 / Coqui",
    "kaldi": "Official CosyVoice (Kaldi)",
}
LAYOUT_HINTS = {
    "ljspeech": "wavs/clip.wav + metadata.csv (clip|text) and metadata_speakers.csv (clip|speaker|text).",
    "pairs": "All clips at the root, each with its .txt: what the \"dataset/\" folders of FireRedTTS3, MOSS-TTS, Higgs, IndexTTS, CosyVoice3, Qwen and OmniVoice expect (they build their own manifest).",
    "audio": "Only the audio clips, without transcripts or metadata (e.g. for voice cloning references or listening).",
    "fish": "One folder per speaker with clip.wav + clip.lab, as fish-speech expects.",
    "qwen": "train_raw.jsonl / eval_raw.jsonl (audio, text, ref_audio). One ref_audio per speaker (the clearest 4-10 s clip). Full Qwen3-TTS SFT is single-speaker and trains on 24 kHz mono audio.",
    "voxcpm": "train.jsonl / val.jsonl (audio, text, duration). Paths relative to the manifest folder; VoxCPM recommends 3-30 s WAV clips.",
    "xtts": "metadata_train.csv + metadata_eval.csv (audio_file|text|speaker_name). \"coqui\" format with header and lang.txt with the language; XTTS uses 22050 Hz WAV.",
    "kaldi": "wav.scp, text, utt2spk, spk2utt. Utterance ids prefixed with the speaker (Kaldi order). Then run the official preparation.",
}
LICENSE_TEXT = ("The transcripts and timestamps in this dataset were generated with CrisperWhisper 2.0 "
                "(nyra health GmbH) under the nyra health Non-Commercial Research License: non-commercial use "
                "only; they may not be used to train commercially used models without a license from nyra health.\n")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()) or "speaker"


def _csv(rows: list[list[str]], header: list[str] | None = None) -> str:
    import io
    buf = io.StringIO(newline="")
    w = csv.writer(buf, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
    if header:
        w.writerow(header)
    for r in rows:
        w.writerow([v.replace("|", " ") for v in r])
    return buf.getvalue()


def _jsonl(items: list[dict]) -> str:
    return "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in items)


def _reference(group: list[Clip]) -> Clip:
    """Reference clip: the most reliable one between 4 and 10 s (if none, the longest)."""
    mid = [c for c in group if 4.0 <= c.duration <= 10.0]
    return max(mid, key=lambda c: (c.min_prob, c.speech_ratio)) if mid else max(group, key=lambda c: c.duration)


def export(clips: list[Clip], out: Path, prm: DatasetParams, ctx: lv.Ctx | None = None,
           fmt: str = "wav16", layout: str = "ljspeech", eval_pct: float = 5.0, package: str = "zip") -> int:
    """Full dataset with layout `layout` (see EXPORT_LAYOUTS) and the clips in format `fmt`.

    package="zip" writes `out` (.zip); package="folder" creates the folder `out` uncompressed. Every layout except
    "audio" (clips only) also includes metadata.jsonl with all the data of every clip. Returns the clips exported."""
    import io
    import shutil
    import zipfile
    ctx = ctx or lv.Ctx()
    audio_only = layout == "audio"
    out = Path(out).with_suffix(".zip") if package == "zip" else Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if package != "zip" and out.exists():
        raise FileExistsError(f"The folder already exists: {out}")
    ext = audio_io.ext_for(fmt)
    rows = [c for c in clips if c.keep and c.audio is not None and (c.text or audio_only)]
    for k, c in enumerate(rows):
        c.name = c.name or f"{Path(c.source).stem}_{k:05d}"
    spk_of = {id(c): _safe(c.speaker or prm.speaker) for c in rows}
    every = max(2, int(round(100 / eval_pct))) if eval_pct > 0 and len(rows) >= 10 else 0
    is_eval = {id(c): bool(every) and k % every == every - 1 for k, c in enumerate(rows)}  # reparto parejo

    def rel(c: Clip) -> str:
        if layout in ("pairs", "audio"):
            return f"{c.name}{ext}"
        if layout == "fish":
            return f"{spk_of[id(c)]}/{c.name}{ext}"
        return f"wavs/{c.name}{ext}"

    tmp = out.with_name(out.name + ".part")  # if aborted, nothing half-written is left under the final name
    z = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) if package == "zip" else None

    def put(name: str, data: str | bytes, stored: bool = False) -> None:
        if z is not None:
            z.writestr(name, data, zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            return
        f = tmp / name
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            f.write_bytes(data.encode("utf-8"))  # same bytes as in the ZIP (\n line endings)
        else:
            f.write_bytes(data)

    try:
        if z is None:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
        meta = []
        for k, c in enumerate(rows):
            ctx.check()
            buf = io.BytesIO()
            audio_io.write_audio(buf, c.audio, prm.out_sr, fmt)
            # already-compressed formats are stored as is (compressing them again only costs time)
            put(rel(c), buf.getvalue(), stored=fmt in ("mp3", "ogg", "flac16", "flac24"))
            if layout == "pairs":
                put(f"{c.name}.txt", c.text + "\n")
            elif layout == "fish":
                put(f"{spk_of[id(c)]}/{c.name}.lab", c.text)
            m = {k2: v for k2, v in asdict(c).items() if k2 != "audio"}
            m.update(audio=rel(c), duration=round(c.duration, 3), speaker=c.speaker or prm.speaker,
                     sample_rate=prm.out_sr, language=prm.language, format=audio_io.EXPORT_FORMATS[fmt][0],
                     split="eval" if is_eval[id(c)] else "train")
            meta.append(m)
            ctx.progress((k + 1) / max(1, len(rows)) * 0.98)
        train = [c for c in rows if not is_eval[id(c)]]
        evals = [c for c in rows if is_eval[id(c)]]
        groups: dict[str, list[Clip]] = {}
        for c in rows:
            groups.setdefault(spk_of[id(c)], []).append(c)
        if layout == "ljspeech":
            put("metadata.csv", _csv([[c.name, c.text] for c in rows]))
            put("metadata_speakers.csv", _csv([[c.name, c.speaker or prm.speaker, c.text] for c in rows]))
        elif layout == "qwen":
            ref = {s: f"./{rel(_reference(g))}" for s, g in groups.items()}
            line = lambda c: {"audio": f"./{rel(c)}", "text": c.text, "ref_audio": ref[spk_of[id(c)]]}
            put("train_raw.jsonl", _jsonl([line(c) for c in train]))
            if evals:
                put("eval_raw.jsonl", _jsonl([line(c) for c in evals]))
        elif layout == "voxcpm":
            line = lambda c: {"audio": rel(c), "text": c.text, "duration": round(c.duration, 3)}
            put("train.jsonl", _jsonl([line(c) for c in train]))
            if evals:
                put("val.jsonl", _jsonl([line(c) for c in evals]))
        elif layout == "xtts":
            head = ["audio_file", "text", "speaker_name"]
            put("metadata_train.csv", _csv([[rel(c), c.text, spk_of[id(c)]] for c in train], head))
            put("metadata_eval.csv", _csv([[rel(c), c.text, spk_of[id(c)]] for c in evals or train[-1:]], head))
            put("lang.txt", prm.language + "\n")
        elif layout == "kaldi":
            utt = {id(c): f"{spk_of[id(c)]}-{c.name}" for c in rows}
            order = sorted(rows, key=lambda c: utt[id(c)])
            put("wav.scp", "".join(f"{utt[id(c)]} {rel(c)}\n" for c in order))
            put("text", "".join(f"{utt[id(c)]} {c.text}\n" for c in order))
            put("utt2spk", "".join(f"{utt[id(c)]} {spk_of[id(c)]}\n" for c in order))
            put("spk2utt", "".join(f"{s} {' '.join(sorted(utt[id(c)] for c in g))}\n"
                                          for s, g in sorted(groups.items())))
        if not audio_only:
            put("metadata.jsonl", _jsonl(meta))
            # .md so GUIs that read "every .txt is a transcript" don't mistake it for a clip
            put("LICENSE_NOTE.md", LICENSE_TEXT)
        if z is not None:
            z.close()
            z = None
        tmp.replace(out)
    finally:
        if z is not None:
            z.close()
        if tmp.exists():
            shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
    ctx.progress(1.0)
    ctx.log(f"Dataset exported: {len(rows)} clips · {EXPORT_LAYOUTS[layout].split(' · ')[0]} · {audio_io.EXPORT_FORMATS[fmt][0]} · {len(evals)} for evaluation → {out}", 0)
    return len(rows)
