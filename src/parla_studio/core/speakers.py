"""Speakers: diarization (Nemotron 3 Diarization), identification by voice print and exports.

Rules:
- The original is never modified: per-person tracks and the isolated voice are derivatives.
- Diarization runs on the original (before isolating or separating), to keep the time references.
- An identity is assigned only if the similarity exceeds the threshold; otherwise it stays "unknown".
- Voice prints are built only from segments without overlap.
"""
from __future__ import annotations

import datetime
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio_io, models
from ..i18n import tr as _tr

COLORS = ["#60a5fa", "#f59e0b", "#a78bfa", "#f472b6", "#34d399", "#f87171", "#22d3ee", "#facc15"]


@dataclass
class Segment:
    start: float
    end: float
    speaker: int


@dataclass
class Speaker:
    idx: int                 # diarizer index (order of appearance)
    name: str                # "Speaker 1" or the identified / renamed person
    identity: str | None = None   # recognized enrolled person
    score: float = 0.0       # similarity with that person
    seconds: float = 0.0     # speaking time
    turns: int = 0
    embedding: np.ndarray | None = field(default=None, repr=False)
    seconds_clean: float = 0.0  # seconds without overlap used for the voice print
    color: str = ""          # color of the assigned slot
    slot: int = -1           # interface slot (-1 = unassigned)

    def __post_init__(self) -> None:
        if not self.color:
            self.color = COLORS[self.idx % len(COLORS)]


@dataclass
class Diarization:
    prob: np.ndarray                    # [T, 8] P(speech) per speaker, every `frame_s`
    frame_s: float = models.DIAR_FRAME_S
    threshold: float = 0.5
    min_seg_s: float = 0.25             # shorter islands and gaps are discarded

    def __post_init__(self) -> None:
        act = self.prob > self.threshold
        n = max(1, int(round(self.min_seg_s / self.frame_s)))
        self.active = np.stack([_clean(act[:, k], n) for k in range(act.shape[1])], axis=1)
        self.speakers = [k for k in range(self.active.shape[1]) if self.active[:, k].sum() * self.frame_s >= 1.0]

    @property
    def overlap(self) -> np.ndarray:
        return self.active.sum(axis=1) >= 2

    def segments(self) -> list[Segment]:
        out = []
        for k in self.speakers:
            d = np.diff(np.concatenate(([0], self.active[:, k].astype(np.int8), [0])))
            out += [Segment(a * self.frame_s, b * self.frame_s, k) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]
        return sorted(out, key=lambda s: (s.start, s.speaker))

    def _span(self, t0: float, t1: float) -> slice:
        a = int(max(0.0, t0) / self.frame_s)
        return slice(a, max(a + 1, int(np.ceil(t1 / self.frame_s))))

    def who(self, t0: float, t1: float) -> tuple[int | None, float]:
        """(dominant speaker in [t0, t1], fraction of the span with overlapping voices)."""
        sl = self._span(t0, t1)
        act = self.active[sl]
        if act.size == 0:
            return None, 0.0
        # probability decides when nobody exceeds the threshold (isolated words, breaths)
        score = act.sum(axis=0) + 1e-3 * self.prob[sl].sum(axis=0)
        k = int(np.argmax(score))
        return (k if score[k] > 0 else None), float(self.overlap[sl].mean())


def _runs(m: np.ndarray, value: bool) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], (m == value).astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _clean(mask: np.ndarray, n: int) -> np.ndarray:
    """Close gaps shorter than n frames between voice frames, then remove islands shorter than n."""
    m = mask.copy()
    for a, b in _runs(m, False):
        if b - a < n and a > 0 and b < m.size:
            m[a:b] = True
    for a, b in _runs(m, True):
        if b - a < n:
            m[a:b] = False
    return m


def diarize(mono: np.ndarray, sr: int) -> Diarization:
    return Diarization(models.speaker_probabilities(mono, sr))


def gate_with_speech(diar: Diarization, speech: np.ndarray, threshold: float = 0.5, margin_s: float = 0.2) -> Diarization:
    """Diarization with every speaker silenced where SAVA hears no voice (music, noise, effects),
    with a margin around the voice so word edges are kept."""
    T = diar.prob.shape[0]
    if speech.size == 0 or T == 0:
        return diar
    v = models.speech_on_hops(speech, T, diar.frame_s) >= threshold
    n = int(round(margin_s / diar.frame_s))
    if n > 0:
        c = np.concatenate(([0], np.cumsum(v.astype(np.int64))))
        i = np.arange(T)
        v = (c[np.minimum(i + n + 1, T)] - c[np.maximum(i - n, 0)]) > 0
    prob = np.where(v[:, None], diar.prob, 0.0).astype(np.float32)
    return Diarization(prob, diar.frame_s, diar.threshold, diar.min_seg_s)


# ---------------------------------------------------------------- enrolled people

@dataclass
class Profile:
    name: str
    embedding: list[float]
    samples: int = 1
    updated: str = ""
    seconds: float = 0.0     # accumulated seconds of reference voice


class ProfileStore:
    """Voice prints of known people, in data/profiles.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.profiles: dict[str, Profile] = {}
        if path.exists():
            for d in json.loads(path.read_text(encoding="utf-8")):
                self.profiles[d["name"]] = Profile(**d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(p) for p in self.profiles.values()], ensure_ascii=False, indent=1),
                             encoding="utf-8")

    def enroll(self, name: str, emb: np.ndarray, seconds: float = 0.0) -> Profile:
        """Add a voice print; if the person exists, it is averaged with the previous ones."""
        name = name.strip()
        p = self.profiles.get(name)
        if p is None:
            p = Profile(name, emb.tolist(), 1, seconds=seconds)
        else:
            m = np.asarray(p.embedding) * p.samples + emb
            p.embedding, p.samples = (m / np.linalg.norm(m)).tolist(), p.samples + 1
            p.seconds += seconds
        p.updated = datetime.datetime.now().isoformat(timespec="seconds")
        self.profiles[name] = p
        self.save()
        return p

    def remove(self, name: str) -> None:
        self.profiles.pop(name, None)
        self.save()

    def rename(self, old: str, new: str) -> None:
        if old in self.profiles and new and new not in self.profiles:
            p = self.profiles.pop(old)
            p.name = new
            self.profiles[new] = p
            self.save()

    def identify(self, emb: np.ndarray, threshold: float) -> tuple[str | None, float]:
        best, score = None, -1.0
        for p in self.profiles.values():
            s = float(np.dot(emb, np.asarray(p.embedding)))
            if s > score:
                best, score = p.name, s
        return (best, score) if best is not None and score >= threshold else (None, max(score, 0.0))


def clean_pieces(mono: np.ndarray, sr: int, diar: Diarization, k: int, max_total_s: float = 60.0,
                 min_s: float = 1.0, max_piece_s: float = 10.0) -> list[np.ndarray]:
    """Fragments of one speaker without overlap (longest first), for their voice print."""
    solo = diar.active[:, k] & ~diar.overlap
    d = np.diff(np.concatenate(([0], solo.astype(np.int8), [0])))
    spans = sorted(((b - a, a, b) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])), reverse=True)
    out, total = [], 0.0
    for length, a, b in spans:
        dur = length * diar.frame_s
        if dur < min_s or total >= max_total_s:
            break
        t, end = a * diar.frame_s, b * diar.frame_s
        while t + min_s <= end and total < max_total_s:
            e = min(end, t + max_piece_s)
            out.append(mono[int(t * sr):int(e * sr)])
            total += e - t
            t = e
    return out


def describe(diar: Diarization, mono: np.ndarray, sr: int, store: ProfileStore | None, threshold: float) -> list[Speaker]:
    """Speakers with time, turns, voice print and identity (if an enrolled person is recognized)."""
    segs = diar.segments()
    out = []
    for n, k in enumerate(diar.speakers):
        sp = Speaker(k, _tr(f"Speaker {n + 1}"), seconds=float(diar.active[:, k].sum() * diar.frame_s),
                     turns=sum(s.speaker == k for s in segs))
        pieces = clean_pieces(mono, sr, diar, k)
        sp.seconds_clean = float(sum(p.size for p in pieces) / sr)
        if pieces:
            sp.embedding = models.voice_embedding(pieces, sr)
            if store is not None and store.profiles:
                who, score = store.identify(sp.embedding, threshold)
                sp.score = score
                if who:
                    sp.identity, sp.name = who, who
        out.append(sp)
    return out


# ---------------------------------------------------------------- transcript per speaker

@dataclass
class Turn:
    start: float
    end: float
    speaker: int | None
    text: str
    overlap: float = 0.0
    words: list = field(default_factory=list, repr=False)


def assign_words(words: list[models.Word], diar: Diarization, separated: list | None = None) -> list[tuple[int | None, float]]:
    """Speaker of each word. Words transcribed from a separated voice already carry their speaker;
    in separated segments without that, the separated voice with the most energy during the word decides."""
    out = []
    for w in words:
        if getattr(w, "speaker", None) is not None:
            out.append((w.speaker, 1.0))
            continue
        k, ov = diar.who(w.start, w.end)
        for sp in separated or []:
            if ov > 0 and sp.start <= w.start and w.end <= sp.end:
                a, b = int((w.start - sp.start) * sp.sr), int((w.end - sp.start) * sp.sr)
                energy = {kk: float(np.mean(v[a:max(b, a + 1)].astype(np.float64) ** 2)) for kk, v in sp.voices.items()}
                k = max(energy, key=energy.get)
                break
        out.append((k, ov))
    return out


def turns(words: list[models.Word], diar: Diarization, max_s: float = 6.0, max_chars: int = 84, pause_s: float = 1.2,
          separated: list | None = None) -> list[Turn]:
    """Group words into turns/subtitles: a new one starts on a speaker change, after `max_s` or at the end of a
    long sentence; used for the transcript table and the SRT."""
    # Each speaker builds their turns separately (their words interleave with the other's when
    # they overlap); a turn is cut by a pause of that person, by length or when a long sentence ends.
    out: list[Turn] = []
    owners = assign_words(words, diar, separated)
    cur: dict = {}
    last_other: dict = {}  # end of the last word of ANOTHER speaker without overlap, per speaker
    for w, (spk, ov) in zip(words, owners):
        t = cur.get(spk)
        closes = w.text.endswith((".", "?", "!", "…"))  # if the word ends the sentence, a bit more is tolerated
        new = (t is None or w.start - t.end > pause_s
               or last_other.get(spk, -1.0) > t.end    # the other one spoke in between (alternation)
               or w.end - t.start > max_s + (1.5 if closes else 0.0)
               or len(t.text) + len(w.text) + 1 > max_chars + (24 if closes else 0)
               or (t.text.endswith((".", "?", "!", "…")) and w.end - t.start > 2.0))
        if new:
            t = Turn(w.start, w.end, spk, w.text, ov, [w])
            cur[spk] = t
            out.append(t)
        else:
            t.end, t.text = w.end, f"{t.text} {w.text}"
            t.words.append(w)
            t.overlap = max(t.overlap, ov)
        if ov == 0:
            for other in cur:
                if other != spk:
                    last_other[other] = w.end
    return sorted(out, key=lambda t: (t.start, -1 if t.speaker is None else t.speaker))


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(tr: list[Turn], names: dict[int, str]) -> str:
    lines = []
    for i, t in enumerate(tr, 1):
        who = names.get(t.speaker, _tr("Unknown")) if t.speaker is not None else _tr("Unknown")
        lines += [str(i), f"{_srt_time(t.start)} --> {_srt_time(t.end)}", f"{who}: {t.text}", ""]
    return "\n".join(lines)


def to_rttm(diar: Diarization, names: dict[int, str], file_id: str) -> str:
    rows = []
    for s in diar.segments():
        name = names.get(s.speaker, f"spk{s.speaker}").replace(" ", "_")
        rows.append(f"SPEAKER {file_id} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {name} <NA> <NA>")
    return "\n".join(rows) + "\n"


def speaker_track(x: np.ndarray, sr: int, diar: Diarization, k: int, fade_s: float = 0.03,
                  margin_s: float = 0.08, separated: list | None = None) -> np.ndarray:
    """Track of one person: the audio only where they speak (with margin and fades); silence elsewhere.
    With `separated` (TIGER), each overlapped segment uses that person's separated voice."""
    y = _gated(x, sr, diar, k, fade_s, margin_s)
    for sp in separated or []:
        if k not in sp.voices:
            continue
        a, b = int(sp.start * sr), min(y.shape[0], int(sp.end * sr))
        v = sp.voices[k][: b - a]
        b = a + v.size
        w = np.ones(b - a, np.float32)
        nf = min(int(fade_s * sr), (b - a) // 2)
        if nf > 0:
            ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
            w[:nf], w[b - a - nf:] = ramp, ramp[::-1]
        y[a:b] = y[a:b] * (1 - w[:, None]) + v[:, None] * w[:, None]
    return y


def _gated(x: np.ndarray, sr: int, diar: Diarization, k: int, fade_s: float, margin_s: float) -> np.ndarray:
    act = diar.active[:, k].astype(np.float32)
    n = int(round(margin_s / diar.frame_s))
    if n:
        act = np.convolve(act, np.ones(2 * n + 1), mode="same") > 0
    gain = act.astype(np.float32)
    f = max(1, int(round(fade_s / diar.frame_s)))
    gain = np.convolve(gain, np.ones(f) / f, mode="same")  # rampas suaves
    t = np.arange(x.shape[0]) / sr
    g = np.interp(t, (np.arange(gain.size) + 0.5) * diar.frame_s, gain).astype(np.float32)
    return x * g[:, None]


# ---------------------------------------------------------------- overlapping voices (TIGER)

@dataclass
class Separated:
    start: float                     # processed segment (with margin), s
    end: float
    voices: dict = field(default_factory=dict)   # speaker index → separated voice (mono, at sr)
    confidence: float = 0.0          # difference between the chosen assignment and the opposite one
    sr: int = 16000


def overlap_regions(diar: Diarization, min_s: float = 0.2, pad_s: float = 1.0, merge_s: float = 0.6,
                    max_s: float = 30.0) -> list[tuple[float, float]]:
    """Overlapped segments of at least `min_s`, with margin and merged when close."""
    spans = [(a * diar.frame_s, b * diar.frame_s) for a, b in _runs(diar.overlap, True)
             if (b - a) * diar.frame_s >= min_s]
    merged: list[list[float]] = []
    for a, b in spans:
        a, b = max(0.0, a - pad_s), b + pad_s
        if merged and a - merged[-1][1] <= merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:  # very long segments in parts (bounded memory)
        t = a
        while t < b:
            out.append((t, min(b, t + max_s)))
            t += max_s
    return out


def _envelope(v: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    hop = max(1, int(sr * frame_s))
    n = v.size // hop
    return np.sqrt((v[: n * hop].astype(np.float64) ** 2).reshape(n, hop).mean(axis=1) + 1e-12)


def separate_overlaps(mono: np.ndarray, sr: int, diar: Diarization, speakers_: list, ctx=None) -> list:
    """Separate the two-voice segments with TIGER and assign each separated voice to its speaker by combining:
    - how well its energy matches the activity the diarizer gives each speaker, and
    - the similarity of its voice print with each speaker's voice print (if it lasts at least 1 s)."""
    regions = overlap_regions(diar)
    emb = {s.idx: s.embedding for s in speakers_ if s.embedding is not None}
    out = []
    for n, (a, b) in enumerate(regions):
        if ctx:
            ctx.check()
            ctx.progress((n + 1) / max(1, len(regions)))
        sl = diar._span(a, b)
        talk = diar.active[sl].sum(axis=0)
        pair = [int(k) for k in np.argsort(talk)[::-1][:2] if talk[k] > 0]
        if len(pair) < 2:
            continue
        seg = mono[int(a * sr):int(b * sr)]
        voices = models.separate_two(seg, sr)
        env = [_envelope(v, sr, diar.frame_s) for v in voices]
        m = min(env[0].size, sl.stop - sl.start)

        def score(i: int, k: int) -> float:
            e, p = env[i][:m], diar.prob[sl][:m, k]
            c = float(np.corrcoef(e, p)[0, 1]) if e.std() > 0 and p.std() > 0 else 0.0
            if k in emb and voices[i].size >= sr:
                c += float(np.dot(models.voice_embedding([voices[i]], sr), emb[k]))
            return c

        straight = score(0, pair[0]) + score(1, pair[1])
        swapped = score(0, pair[1]) + score(1, pair[0])
        order = (0, 1) if straight >= swapped else (1, 0)
        out.append(Separated(a, b, {pair[0]: voices[order[0]].astype(np.float32),
                                    pair[1]: voices[order[1]].astype(np.float32)},
                             confidence=abs(straight - swapped), sr=sr))
    return out


def identify_with_separated(speakers_: list[Speaker], separated: list, store: "ProfileStore | None",
                            threshold: float, min_clean_s: float = 3.0) -> None:
    """Speakers who almost always talk over others have no clean segments for their voice print: it is computed
    (or reinforced) from their TIGER-separated voice and they are identified again."""
    for sp in speakers_:
        pieces = [s.voices[sp.idx] for s in separated if sp.idx in s.voices]
        if not pieces or (sp.embedding is not None and sp.seconds_clean >= min_clean_s):
            continue
        sr = separated[0].sr
        try:
            e_sep = models.voice_embedding(pieces, sr)
        except ValueError:
            continue
        if sp.embedding is not None:  # average weighted by the seconds of each source
            w = min(1.0, sp.seconds_clean / min_clean_s)
            m = w * sp.embedding + (1 - w) * e_sep
            sp.embedding = (m / np.linalg.norm(m)).astype(np.float32)
        else:
            sp.embedding = e_sep
        if store is not None and store.profiles and not sp.identity:
            who, score = store.identify(sp.embedding, threshold)
            sp.score = score
            if who:
                sp.identity, sp.name = who, who


def merge_separated_words(words: list[models.Word], separated: list, transcribe_fn, edge_s: float = 0.25,
                          ctx=None) -> list[models.Word]:
    """Transcribe each voice separated by TIGER on its own and merge it into the transcript:
    in each separated segment, the words of the main transcript (which only hears the dominant voice)
    are replaced by those of each voice, with a fixed speaker. So both voices end up transcribed.
    `transcribe_fn(audio_mono, sr) -> list[Word]` with times relative to the segment."""
    inner = [(sp.start + edge_s, sp.end - edge_s) for sp in separated]
    mid = lambda w: (w.start + w.end) / 2
    out = [w for w in words if not any(a <= mid(w) <= b for a, b in inner)]
    for n, sp in enumerate(separated):
        if ctx:
            ctx.check()
            ctx.progress((n + 1) / max(1, len(separated)))
        a, b = inner[n]
        for k, v in sp.voices.items():
            for w in transcribe_fn(v, sp.sr):
                w2 = models.Word(w.text, w.start + sp.start, w.end + sp.start, w.prob, k)
                if a <= mid(w2) <= b:  # edges: covered by the main transcript
                    out.append(w2)
    return sorted(out, key=lambda w: (w.start, -1 if w.speaker is None else w.speaker))


# ---------------------------------------------------------------- speaker slots and speaker limit

MAX_SPEAKERS = 8  # diarizer maximum


@dataclass
class Slot:
    name: str
    color: str


class SlotStore:
    """Interface slots (name and color per speaker), in data/slots.json."""

    def __init__(self, path: Path, initial: int = 4) -> None:
        self.path = path
        self.slots: list[Slot] = []
        if path.exists():
            self.slots = [Slot(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
        while len(self.slots) < initial:
            self.add()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(s) for s in self.slots], ensure_ascii=False, indent=1), encoding="utf-8")

    def add(self, name: str | None = None) -> Slot | None:
        if len(self.slots) >= MAX_SPEAKERS:
            return None
        n = len(self.slots)
        used = {s.color for s in self.slots}
        color = next((c for c in COLORS if c not in used), COLORS[n % len(COLORS)])
        s = Slot(name or _tr(f"Speaker {n + 1}"), color)
        self.slots.append(s)
        self.save()
        return s

    def remove(self, i: int) -> None:
        if 0 <= i < len(self.slots) and len(self.slots) > 1:
            del self.slots[i]
            self.save()


def limit_speakers(diar: Diarization, speakers_: list[Speaker], n: int) -> list[Speaker]:
    """Keep at most `n` speakers: those who speak least merge into the most similar one (voice print)."""
    if len(speakers_) <= n:
        return speakers_
    keep = sorted(speakers_, key=lambda s: -s.seconds)[:n]
    for s in speakers_:
        if s in keep:
            continue
        if s.embedding is not None and any(k.embedding is not None for k in keep):
            target = max((k for k in keep if k.embedding is not None), key=lambda k: float(np.dot(k.embedding, s.embedding)))
        else:
            target = keep[0]
        diar.prob[:, target.idx] = np.maximum(diar.prob[:, target.idx], diar.prob[:, s.idx])
        diar.prob[:, s.idx] = 0.0
    diar.__post_init__()  # recompute activity, overlaps and speakers
    segs = diar.segments()
    for k in keep:
        k.seconds = float(diar.active[:, k.idx].sum() * diar.frame_s)
        k.turns = sum(g.speaker == k.idx for g in segs)
    return sorted(keep, key=lambda s: s.idx)


def apply_slots(speakers_: list[Speaker], store: SlotStore, auto_add: bool) -> None:
    """Assign every detected speaker to a slot (name and color):
    1) recognized person → the slot with their name (if none, a free one that takes that name);
    2) the rest → free slots in order of appearance; in automatic mode slots are created when missing."""
    used: set[int] = set()
    by_name = {sl.name: i for i, sl in enumerate(store.slots)}
    pending = []
    for sp in speakers_:
        i = by_name.get(sp.identity) if sp.identity else None
        if i is not None and i not in used:
            sp.slot = i
            used.add(i)
        else:
            pending.append(sp)
    for sp in pending:
        free = [i for i in range(len(store.slots)) if i not in used]
        if not free and auto_add and store.add() is not None:
            free = [len(store.slots) - 1]
        if free:
            i = free[0]
            if sp.identity:  # recognized person without their own slot: the free slot takes their name
                store.slots[i].name = sp.identity
            sp.slot = i
            used.add(i)
        else:
            sp.slot = -1
    store.save()
    for sp in speakers_:
        if sp.slot >= 0:
            sl = store.slots[sp.slot]
            sp.name, sp.color = sl.name, sl.color


# ---------------------------------------------------------------- people from reference clips

@dataclass
class ClipReport:
    file: str
    seconds: float = 0.0        # usable voice used for the voice print
    speakers: int = 0           # voices detected in the clip
    similarity: float = 0.0     # similarity with the rest of the clips
    accepted: bool = False
    note: str = ""


def enroll_from_clips(store: ProfileStore, name: str, files: list, isolate: bool = True,
                      min_similarity: float = 0.70, ctx=None) -> tuple:
    """Create or improve a person's voice print from reference clips.

    Each clip is isolated (optional) and diarized: if it has several voices only the main one is used, and only
    segments without overlap. Clips that don't match the rest (another person, heavy noise) are
    discarded with a notice, so they don't contaminate the voice print."""
    reports, embs = [], []
    for n, f in enumerate(files):
        if ctx:
            ctx.check()
            ctx.progress(n / max(1, len(files)))
        rep = ClipReport(Path(f).name)
        try:
            x, sr = audio_io.load(f)
            if isolate:
                x = models.isolate_voice(x, sr, ctx.sub(n / max(1, len(files)), (n + 0.8) / max(1, len(files))) if ctx is not None else None)
            mono = audio_io.to_mono(x)
            diar = diarize(mono, sr)
            rep.speakers = len(diar.speakers)
            if not diar.speakers:
                rep.note = "no voice detected"
            else:
                main = max(diar.speakers, key=lambda k: diar.active[:, k].sum())
                pieces = clean_pieces(mono, sr, diar, main, max_total_s=120.0, min_s=0.8)
                rep.seconds = float(sum(p.size for p in pieces) / sr)
                if rep.seconds < 2.0:
                    rep.note = "too little voice without overlap (< 2 s)"
                else:
                    embs.append((rep, models.voice_embedding(pieces, sr)))
                    if rep.speakers > 1:
                        rep.note = f"{rep.speakers} voices: the main one was used"
        except Exception as e:  # noqa: BLE001 - reported per clip
            rep.note = f"error: {e}"
        reports.append(rep)
    if not embs:
        return None, reports
    E = np.array([e for _, e in embs])
    for i, (rep, _) in enumerate(embs):
        others = np.delete(E, i, axis=0)
        rep.similarity = float((others @ E[i]).mean()) if others.size else 1.0
    good = [(r, e) for r, e in embs if r.similarity >= min_similarity or len(embs) == 1]
    for r, _ in embs:
        r.accepted = any(r is g for g, _ in good)
        if not r.accepted:
            r.note = (r.note + "; " if r.note else "") + f"does not match the rest ({r.similarity:.2f}): discarded"
    prof = None
    for r, e in good:
        prof = store.enroll(name, e, r.seconds)
    return prof, reports


# ---------------------------------------------------------------- project

def export_project(out_dir: Path, source: Path, x: np.ndarray, sr: int, diar: Diarization, spk: list[Speaker],
                   words: list[models.Word], voice: np.ndarray | None, language: str,
                   tracks_from_voice: bool = False, ctx=None, separated: list | None = None,
                   fmt: str = "wav24") -> Path:
    """Post-production project layout:
    original/ (untouched copy) · processed/isolated/ · speakers/<person>.wav · transcripts/ (SRT, JSON, RTTM)
    · project.json (segments with speaker, identity, confidence, overlap, source and models)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original").mkdir(exist_ok=True)
    orig = out_dir / "original" / source.name
    if not orig.exists():
        shutil.copy2(source, orig)
    ext = audio_io.ext_for(fmt)
    if voice is not None:
        audio_io.write_audio(out_dir / "processed" / "isolated" / f"{source.stem}_voice{ext}", voice, sr, fmt)
    for n, sp in enumerate(separated or []):  # TIGER derivatives: each overlapped segment, one voice per person
        for k, v in sp.voices.items():
            nm = next((s.name for s in spk if s.idx == k), f"spk{k}")
            audio_io.write_audio(out_dir / "processed" / "separated" /
                                 f"{n + 1:03d}_{sp.start:08.2f}s_{_safe(nm)}{ext}", v, sr, fmt)
    names = {s.idx: s.name for s in spk}
    base = voice if (tracks_from_voice and voice is not None) else x
    files = {}
    for i, s in enumerate(spk):
        if ctx:
            ctx.check()
            ctx.progress((i + 1) / max(1, len(spk)))
        fn = f"speaker_{i + 1:02d}_{_safe(s.name)}{ext}"
        audio_io.write_audio(out_dir / "speakers" / fn,
                             speaker_track(base, sr, diar, s.idx, separated=separated), sr, fmt)
        files[s.idx] = f"speakers/{fn}"
    tr = turns(words, diar, separated=separated)
    tdir = out_dir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "transcript.srt").write_text(to_srt(tr, names), encoding="utf-8")
    (tdir / "diarization.rttm").write_text(to_rttm(diar, names, source.stem), encoding="utf-8")
    segs = [{
        "start": round(t.start, 3), "end": round(t.end, 3), "text": t.text,
        "speaker_id": t.speaker, "speaker": names.get(t.speaker, _tr("Unknown")) if t.speaker is not None else _tr("Unknown"),
        "identity": next((s.identity for s in spk if s.idx == t.speaker), None),
        "identity_confidence": round(next((s.score for s in spk if s.idx == t.speaker), 0.0), 3),
        "overlap": round(t.overlap, 3),
        "audio_source": ("TIGER (separated voices)" if separated and t.overlap > 0 and
                         any(sp.start <= t.start and t.end <= sp.end for sp in separated)
                         else "isolated voice" if voice is not None else "original"),
        "words": [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.prob, 3)} for w in t.words],
    } for t in tr]
    (tdir / "transcript.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    project = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": {"file": f"original/{source.name}", "sample_rate": sr, "channels": int(x.shape[1]),
                   "duration_s": round(x.shape[0] / sr, 3)},
        "language": language,
        "models": {"asr": models.CW_MODEL, "diarization": models.DIAR_ID, "speaker_id": models.SV_ID,
                   "overlap_separation": models.TIGER_ID if separated else None,
                   "voice_isolation": "inworld/voice-isolator" if voice is not None else None},
        "speakers": [{"id": s.idx, "name": s.name, "identity": s.identity, "identity_confidence": round(s.score, 3),
                      "seconds": round(s.seconds, 2), "turns": s.turns, "track": files.get(s.idx)} for s in spk],
        "overlap_seconds": round(float(diar.overlap.sum() * diar.frame_s), 2),
        "separated_regions": [{"start": round(sp.start, 3), "end": round(sp.end, 3),
                               "speakers": [next((s.name for s in spk if s.idx == k), k) for k in sp.voices],
                               "assignment_confidence": round(sp.confidence, 3)} for sp in separated or []],
        "segments": segs,
    }
    (out_dir / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir / "project.json"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "speaker"
